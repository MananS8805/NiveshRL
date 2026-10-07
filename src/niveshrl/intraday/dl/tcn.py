"""Temporal convolutional network (TCN) meta-labeler for intraday entries.

**Why a TCN.** The decision depends on the *shape* of the last couple of hours (a pullback into VWAP after a trend, a
squeeze before a break, volume drying up): local patterns at several time scales. Dilated causal convolutions see
1, 2, 4 and 8 bars at once with few parameters (~25k), train in minutes on a CPU, never look ahead, and are the
standard strong baseline for sequence classification in finance and elsewhere (Bai, Kolter & Koltun 2018). An
LSTM/Transformer is slower to train on this hardware for no measured gain at a 24-bar horizon.

**Heads (multi-task).** For a long and a short entered at the next bar's open: P(the trade ends in profit after
costs) and E[R after costs] under the triple-barrier rules of ``dataset.py``. Both are calibrated (isotonic) on the
last days before the test period. The 32-number embedding before the heads is reused by the neural bandit.

**Walk-forward.** ``walk_forward`` trains on all earlier days only, refits every ``refit_every`` days, and returns
out-of-sample predictions for every later stock-bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .dataset import CONTEXT, CHANNELS, W, DayBlock, windows

R_CLIP = (-1.5, 2.5)


class _Block(nn.Module):
    def __init__(self, cin, cout, dil):
        super().__init__()
        self.pad = 2 * dil
        self.conv1 = nn.Conv1d(cin, cout, 3, dilation=dil)
        self.conv2 = nn.Conv1d(cout, cout, 3, dilation=dil)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):                                  # causal: left padding only
        y = self.act(self.conv1(nn.functional.pad(x, (self.pad, 0))))
        y = self.act(self.conv2(nn.functional.pad(y, (self.pad, 0))))
        return y + self.skip(x)


class TCN(nn.Module):
    def __init__(self, c_in: int = len(CHANNELS), k_ctx: int = len(CONTEXT), d: int = 32, emb: int = 32):
        super().__init__()
        self.tcn = nn.Sequential(_Block(c_in, d, 1), _Block(d, d, 2), _Block(d, d, 4), _Block(d, d, 8))
        self.mix = nn.Sequential(nn.Linear(d + k_ctx, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, emb), nn.GELU())
        self.head = nn.Linear(emb, 4)                       # logit long, logit short, E[R] long, E[R] short

    def embed(self, seq, ctx):
        h = self.tcn(seq.transpose(1, 2))[:, :, -1]
        return self.mix(torch.cat([h, ctx], -1))

    def forward(self, seq, ctx):
        e = self.embed(seq, ctx)
        return self.head(e), e


def samples(blocks: list[DayBlock], need_label: bool = True, max_n: int | None = None, seed: int = 0):
    """(block index, bar index) pairs of bars that have a label for at least one side."""
    bi, ii = [], []
    for k, b in enumerate(blocks):
        ok = np.isfinite(b.rL) | np.isfinite(b.rS) if need_label else np.ones(len(b.close), bool)
        idx = np.flatnonzero(ok)
        bi.append(np.full(len(idx), k))
        ii.append(idx)
    bi, ii = (np.concatenate(bi), np.concatenate(ii)) if bi else (np.array([], int), np.array([], int))
    if max_n and len(bi) > max_n:
        sel = np.random.default_rng(seed).choice(len(bi), max_n, replace=False)
        bi, ii = bi[sel], ii[sel]
    return bi, ii


def tensors(blocks, bi, ii):
    """Windows, context and labels for the chosen samples (grouped by block for speed)."""
    seq = np.zeros((len(bi), W, len(CHANNELS)), np.float32)
    ctx = np.zeros((len(bi), len(CONTEXT)), np.float32)
    rl = np.full(len(bi), np.nan, np.float32)
    rs = np.full(len(bi), np.nan, np.float32)
    order = np.argsort(bi, kind="stable")
    bs = bi[order]
    starts = np.flatnonzero(np.r_[True, bs[1:] != bs[:-1]])
    ends = np.r_[starts[1:], len(bs)]
    for s0, e0 in zip(starts, ends):
        pos = order[s0:e0]
        b = blocks[bs[s0]]
        seq[pos] = windows(b.seq, ii[pos])
        ctx[pos] = b.ctx[ii[pos]]
        rl[pos], rs[pos] = b.rL[ii[pos]], b.rS[ii[pos]]
    return np.clip(np.nan_to_num(seq), -20, 20), np.clip(ctx, -20, 20), rl, rs


class Model:
    """A trained TCN plus its isotonic calibrators."""

    def __init__(self, net: TCN, cal: dict, n_train: int, val_days: list):
        self.net, self.cal, self.n_train, self.val_days = net, cal, n_train, val_days

    def raw(self, seq, ctx, batch: int = 8192):
        self.net.eval()
        outs, embs = [], []
        with torch.no_grad():
            for i in range(0, len(seq), batch):
                o, e = self.net(torch.from_numpy(seq[i:i + batch]), torch.from_numpy(ctx[i:i + batch]))
                outs.append(o.numpy())
                embs.append(e.numpy())
        o = np.concatenate(outs) if outs else np.zeros((0, 4), np.float32)
        e = np.concatenate(embs) if embs else np.zeros((0, 32), np.float32)
        return o, e

    def predict(self, seq, ctx) -> tuple[pd.DataFrame, np.ndarray]:
        o, e = self.raw(seq, ctx)
        p_l, p_s = 1 / (1 + np.exp(-o[:, 0])), 1 / (1 + np.exp(-o[:, 1]))
        out = pd.DataFrame({"pL": self.cal["pL"].predict(p_l), "pS": self.cal["pS"].predict(p_s),
                            "eL": self.cal["eL"].predict(o[:, 2]), "eS": self.cal["eS"].predict(o[:, 3]),
                            "eL_raw": o[:, 2], "eS_raw": o[:, 3]})
        return out, e


def fit(blocks: list[DayBlock], val_days: int = 3, max_n: int = 400_000, epochs: int = 4, seed: int = 0,
        batch: int = 1024, lr: float = 2e-3, verbose: bool = False) -> Model:
    """Train on all but the last ``val_days`` days of ``blocks``; early-stop and calibrate on those days."""
    from sklearn.isotonic import IsotonicRegression
    torch.manual_seed(seed)
    days = sorted({b.day for b in blocks})
    vset = set(days[-val_days:])
    tr = [b for b in blocks if b.day not in vset]
    va = [b for b in blocks if b.day in vset]
    bi, ii = samples(tr, max_n=max_n, seed=seed)
    S, C, rl, rs = tensors(tr, bi, ii)
    vbi, vii = samples(va, max_n=150_000, seed=seed + 1)
    Sv, Cv, vrl, vrs = tensors(va, vbi, vii)
    net = TCN()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    bce, hub = nn.BCEWithLogitsLoss(reduction="none"), nn.HuberLoss(reduction="none", delta=1.0)

    def loss_on(o, yl, ys):
        ml, ms = torch.isfinite(yl), torch.isfinite(ys)
        yl0, ys0 = torch.nan_to_num(yl), torch.nan_to_num(ys)
        lb = (bce(o[:, 0], (yl0 > 0).float()) * ml + bce(o[:, 1], (ys0 > 0).float()) * ms).sum()
        lr_ = (hub(o[:, 2], yl0.clamp(*R_CLIP)) * ml + hub(o[:, 3], ys0.clamp(*R_CLIP)) * ms).sum()
        return (lb + 0.5 * lr_) / (ml.sum() + ms.sum()).clamp(min=1)

    St, Ct, yl, ys = map(torch.from_numpy, (S, C, rl, rs))
    Svt, Cvt, vyl, vys = map(torch.from_numpy, (Sv, Cv, vrl, vrs))
    best, state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(yl))
        for i in range(0, len(yl), batch):
            j = perm[i:i + batch]
            o, _ = net(St[j], Ct[j])
            loss = loss_on(o, yl[j], ys[j])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(np.mean([loss_on(net(Svt[i:i + 8192], Cvt[i:i + 8192])[0], vyl[i:i + 8192], vys[i:i + 8192]).item()
                                for i in range(0, len(vyl), 8192)]))
        if verbose:
            print(f"    epoch {ep + 1}: val loss {vl:.4f}", flush=True)
        if vl < best - 1e-4:
            best, state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 2:
                break
    net.load_state_dict(state)
    m = Model(net, {}, len(yl), [str(d.date()) for d in sorted(vset)])
    o, _ = m.raw(Sv, Cv)
    cal = {}
    for key, col, y in (("pL", 0, vrl), ("pS", 1, vrs)):
        ok = np.isfinite(y)
        cal[key] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(1 / (1 + np.exp(-o[ok, col])), (y[ok] > 0))
    for key, col, y in (("eL", 2, vrl), ("eS", 3, vrs)):
        ok = np.isfinite(y)
        cal[key] = IsotonicRegression(out_of_bounds="clip").fit(o[ok, col], y[ok])
    m.cal = cal
    return m


def predict_blocks(m: Model, blocks: list[DayBlock], with_embedding: bool = False):
    """Predictions for every bar of the given blocks (labels attached when known)."""
    bi, ii = samples(blocks, need_label=False)
    S, C, rl, rs = tensors(blocks, bi, ii)
    pr, emb = m.predict(S, C)
    pr["ticker"] = [blocks[k].ticker for k in bi]
    pr["ts"] = [blocks[k].ts[i] for k, i in zip(bi, ii)]
    pr["bar"] = ii
    pr["rL"], pr["rS"] = rl, rs
    pr["dist"] = [blocks[k].dist[i] for k, i in zip(bi, ii)]
    pr["close"] = [blocks[k].close[i] for k, i in zip(bi, ii)]
    return (pr, emb) if with_embedding else pr


def walk_forward(blocks: list[DayBlock], first_test: int = 15, refit_every: int = 5, verbose: bool = True,
                 keep_embedding_for: set | None = None, **fit_kw):
    """Out-of-sample predictions for every day from the ``first_test``-th: each block of ``refit_every`` days is
    predicted by a model trained on all earlier days only."""
    days = sorted({b.day for b in blocks})
    out, embs = [], []
    for s in range(first_test, len(days), refit_every):
        train = [b for b in blocks if b.day < days[s]]
        test_days = set(days[s:s + refit_every])
        test = [b for b in blocks if b.day in test_days]
        m = fit(train, **fit_kw)
        pr, e = predict_blocks(m, test, with_embedding=True)
        pr["refit_day"] = days[s]
        out.append(pr)
        if keep_embedding_for is not None:
            keep = pr["ticker"].isin(keep_embedding_for).to_numpy()
            embs.append(pd.DataFrame(e[keep].astype(np.float16), index=pr.index[keep]))
        if verbose:
            print(f"  [tcn] test {days[s]:%Y-%m-%d}+{len(test_days) - 1}d: trained on {m.n_train:,} samples",
                  flush=True)
    res = pd.concat(out, ignore_index=True)
    return res
