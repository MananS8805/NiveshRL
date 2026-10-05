"""Next-day stock models: which NIFTY 200 stocks will beat the median tomorrow?

Target: label = 1 if the stock's return from today's close to tomorrow's close
beats that day's cross-sectional median. The same idea as the monthly rankers,
at a 1-day horizon.

Features (``FEATURES``, all causal, all known at today's close):
- price and volume behaviour from ``technicals.indicator_frames``;
- market context (VIX, breadth, NIFTY return) and day of week.

Stock features are ranked across stocks each day (to [-0.5, 0.5]). That
removes the market level and makes the models scale-free. Market features are
z-scored with training-window statistics only.

Models:
- ``lgbm``: LightGBM gradient-boosted trees (strong tabular baseline).
- ``seq``: deep sequence model. ``WindowConv`` (the CNN layer from the RL
  encoder) plus a 2-layer Transformer over each stock's last 20 days of features.
- ``logreg``: logistic regression (linear baseline).
- ``reversal``: score = -today's return (classic 1-day reversal, no learning).
- ``ensemble``: mean of the isotonic-calibrated lgbm and seq probabilities.

Validation: walk-forward by calendar year. Train on the previous 5 years minus
the last one, early-stop and calibrate on that last year (isotonic regression),
then predict the test year. A 1-day purge gap separates train/validation/test,
because a label uses the next day's close. Hyperparameters are fixed a priori.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Panel
from .technicals import indicator_frames

STOCK_FEATS = ["ret_1d", "ret_2d", "ret_1w", "ret_10d", "ret_1m", "gap_pct", "range_pct", "close_loc", "rsi14",
               "macd_hist_n", "atr_pct", "adx14", "di_spread", "bb_pctb", "bb_width", "vs_sma20", "vs_sma50",
               "vs_sma200", "from_52w_high", "from_52w_low", "vol_ratio", "log_turnover", "rs_nifty_1m",
               "rs_sector_3m", "vol_60d", "beta_1y", "supertrend", "donchian_breakout", "ret_vs_sector_1d",
               "ret_3m", "ret_6m"]
MARKET_FEATS = ["vix", "vix_chg_5d", "breadth", "nifty_ret_1d", "nifty_ret_5d", "dow_0", "dow_1", "dow_2", "dow_3",
                "dow_4"]
FEATURES = STOCK_FEATS + MARKET_FEATS
SEQ_FEATS = ["ret_1d", "gap_pct", "range_pct", "close_loc", "rsi14", "vol_ratio", "vs_sma20", "atr_pct",
             "ret_vs_sector_1d", "bb_pctb", "macd_hist_n", "from_52w_high"]
SEQ_LEN = 20


@dataclass
class DayData:
    dates: pd.DatetimeIndex
    tickers: list[str]
    X: np.ndarray          # (T, N, F) ranked stock features + market features, float32, NaN = unavailable
    y: np.ndarray          # (T, N) 1/0/NaN: beats tomorrow's median
    ret_next: np.ndarray   # (T, N) next-day return
    valid: np.ndarray      # (T, N) bool: features + price available today


def build(p: Panel, frames: dict | None = None) -> DayData:
    f = frames or indicator_frames(p)
    c = p.close
    f = dict(f)
    f["ret_2d"] = c / c.shift(2) - 1
    f["ret_10d"] = c / c.shift(10) - 1
    f["macd_hist_n"] = f["macd_hist"] / c
    f["di_spread"] = f["plus_di"] - f["minus_di"]
    f["log_turnover"] = np.log1p(f["turnover_cr"])
    sec = p.sectors.reindex(c.columns)
    f["ret_vs_sector_1d"] = f["ret_1d"] - f["ret_1d"].T.groupby(sec).transform("mean").T
    # Point-in-time universe: indicators use each stock's full history, but ranks, labels and breadth
    # are computed only among stocks that were index members that day.
    m = p.member.reindex(index=c.index, columns=c.columns).fillna(False).to_numpy() if p.member is not None else None
    mk = (lambda df: df.where(m)) if m is not None else (lambda df: df)
    ranked = []
    for k in STOCK_FEATS:
        ranked.append((mk(f[k]).rank(axis=1, pct=True) - 0.5).to_numpy(np.float32))
    stock = np.stack(ranked, -1)                                     # (T, N, Fs)
    b = p.bench.reindex(c.index)
    ma200 = c.rolling(200, min_periods=150).mean()
    mkt = pd.DataFrame({
        "vix": p.vix.reindex(c.index) / 100,
        "vix_chg_5d": np.log(p.vix.reindex(c.index)).diff(5),
        "breadth": mk((c > ma200).where(ma200.notna())).mean(axis=1),
        "nifty_ret_1d": b.pct_change(fill_method=None),
        "nifty_ret_5d": b / b.shift(5) - 1,
    }, index=c.index)
    for d in range(5):
        mkt[f"dow_{d}"] = (c.index.dayofweek == d).astype(float)
    M = mkt[MARKET_FEATS].to_numpy(np.float32)                      # (T, Fm)
    X = np.concatenate([stock, np.repeat(M[:, None, :], len(c.columns), 1)], -1)
    nxt = mk(c.shift(-1) / c - 1).to_numpy(np.float32)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # the last day has no next-day prices
        med = np.nanmedian(nxt, axis=1, keepdims=True)
    y = np.where(np.isfinite(nxt), (nxt > med).astype(np.float32), np.nan)
    valid = np.isfinite(stock[..., :5]).all(-1) & np.isfinite(c.to_numpy())
    if m is not None:
        valid &= m
    return DayData(dates=c.index, tickers=list(c.columns), X=X, y=y, ret_next=nxt, valid=valid)


# --------------------------------------------------------------------------- models
def _rows(dd: DayData, t_idx: np.ndarray, need_label: bool = True):
    tt, nn = np.nonzero(dd.valid[t_idx] & (np.isfinite(dd.y[t_idx]) if need_label else True))
    return t_idx[tt], nn


def _tab(dd: DayData, t, n, mu=None, sd=None):
    X = dd.X[t, n].astype(np.float32)
    nf = len(STOCK_FEATS)
    if mu is None:
        mu, sd = np.nanmean(X[:, nf:], 0), np.nanstd(X[:, nf:], 0) + 1e-6
    X[:, nf:] = (X[:, nf:] - mu) / sd
    return np.nan_to_num(X), mu, sd


def _seq(dd: DayData, t, n):
    cols = [FEATURES.index(k) for k in SEQ_FEATS]
    idx = t[:, None] - np.arange(SEQ_LEN - 1, -1, -1)[None, :]        # (B, L)
    S = dd.X[np.clip(idx, 0, None)[..., None], n[:, None, None], np.array(cols)[None, None, :]]
    return np.nan_to_num(S.astype(np.float32))


class SeqNet:
    """WindowConv + Transformer over each stock's last SEQ_LEN days, plus today's full feature row."""

    def __init__(self, n_seq: int, n_tab: int, d: int = 32, seed: int = 0):
        import torch
        import torch.nn as nn

        from ..models.encoder import WindowConv

        torch.manual_seed(seed)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(WindowConv(n_seq, d), nn.GELU(), WindowConv(d, d), nn.GELU())
                self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, d))
                layer = nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True, norm_first=True)
                self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
                self.head = nn.Sequential(nn.Linear(2 * d + n_tab, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

            def forward(self, s, x):
                h = self.enc(self.conv(s) + self.pos)
                return self.head(torch.cat([h.mean(1), h[:, -1], x], -1)).squeeze(-1)

        self.torch = torch
        self.net = Net()
        self._args = (n_seq, n_tab, d, seed)

    def __getstate__(self):
        """Picklable: keep the constructor arguments and the weights, not the torch module or the local class."""
        return {"args": self._args, "state": {k: v.detach().cpu() for k, v in self.net.state_dict().items()}}

    def __setstate__(self, st):
        self.__init__(*st["args"])
        self.net.load_state_dict(st["state"])
        self.net.eval()

    def fit(self, S, X, y, Sv, Xv, yv, epochs: int = 6, batch: int = 1024, lr: float = 1e-3):
        torch = self.torch
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        lossf = torch.nn.BCEWithLogitsLoss()
        S, X, y = map(torch.from_numpy, (S, X, y))
        Sv, Xv, yv = map(torch.from_numpy, (Sv, Xv, yv))
        best, state, bad = np.inf, None, 0
        for _ in range(epochs):
            self.net.train()
            perm = torch.randperm(len(y))
            for i in range(0, len(y), batch):
                j = perm[i:i + batch]
                loss = lossf(self.net(S[j], X[j]), y[j])
                opt.zero_grad()
                loss.backward()
                opt.step()
            v = self._loss(Sv, Xv, yv, lossf)
            if v < best - 1e-5:
                best, state, bad = v, {k: t.clone() for k, t in self.net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 2:
                    break
        self.net.load_state_dict(state)
        return self

    def _loss(self, S, X, y, lossf):
        self.net.eval()
        with self.torch.no_grad():
            return float(np.mean([lossf(self.net(S[i:i + 8192], X[i:i + 8192]), y[i:i + 8192]).item()
                                  for i in range(0, len(y), 8192)]))

    def predict(self, S, X) -> np.ndarray:
        torch = self.torch
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(S), 8192):
                out.append(torch.sigmoid(self.net(torch.from_numpy(S[i:i + 8192]), torch.from_numpy(X[i:i + 8192]))).numpy())
        return np.concatenate(out) if out else np.zeros(0)


def walk_forward(dd: DayData, first_test_year: int = 2015, window: int = 5, models=("lgbm", "seq", "logreg"),
                 verbose: bool = True, train_subsample: float = 0.5, seed: int = 0) -> pd.DataFrame:
    """Out-of-sample next-day probabilities for every stock-day from ``first_test_year`` on.

    Returns a long frame indexed (date, ticker) with one column per model plus
    'reversal', 'ensemble', 'y' and 'ret_next'.
    """
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    years = sorted({d.year for d in dd.dates if d.year >= first_test_year})
    years_arr = dd.dates.year.to_numpy()
    out = []
    for Y in years:
        tr_t = np.flatnonzero((years_arr >= Y - window) & (years_arr <= Y - 2))[:-1]        # purge the last day
        va_t = np.flatnonzero(years_arr == Y - 1)[:-1]
        te_t = np.flatnonzero(years_arr == Y)
        t_tr, n_tr = _rows(dd, tr_t)
        if train_subsample < 1:
            keep = rng.random(len(t_tr)) < train_subsample
            t_tr, n_tr = t_tr[keep], n_tr[keep]
        t_va, n_va = _rows(dd, va_t)
        t_te, n_te = _rows(dd, te_t, need_label=False)
        Xtr, mu, sd = _tab(dd, t_tr, n_tr)
        Xva, _, _ = _tab(dd, t_va, n_va, mu, sd)
        Xte, _, _ = _tab(dd, t_te, n_te, mu, sd)
        ytr, yva = dd.y[t_tr, n_tr], dd.y[t_va, n_va]
        res = {"date": dd.dates[t_te], "ticker": np.array(dd.tickers)[n_te],
               "y": dd.y[t_te, n_te], "ret_next": dd.ret_next[t_te, n_te],
               "reversal": -dd.X[t_te, n_te, FEATURES.index("ret_1d")]}
        cal = {}
        if "lgbm" in models:
            m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                   random_state=seed, verbose=-1)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
            pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
            cal["lgbm"] = IsotonicRegression(out_of_bounds="clip").fit(pv, yva)
            res["lgbm"] = cal["lgbm"].predict(pt)
        if "seq" in models:
            net = SeqNet(len(SEQ_FEATS), Xtr.shape[1], seed=seed)
            net.fit(_seq(dd, t_tr, n_tr), Xtr, ytr.astype(np.float32), _seq(dd, t_va, n_va), Xva, yva.astype(np.float32))
            pv, pt = net.predict(_seq(dd, t_va, n_va), Xva), net.predict(_seq(dd, t_te, n_te), Xte)
            cal["seq"] = IsotonicRegression(out_of_bounds="clip").fit(pv, yva)
            res["seq"] = cal["seq"].predict(pt)
        if "logreg" in models:
            lr = LogisticRegression(C=0.1, max_iter=300).fit(Xtr, ytr)
            res["logreg"] = lr.predict_proba(Xte)[:, 1]
        parts = [res[k] for k in ("lgbm", "seq") if k in res]
        if parts:
            res["ensemble"] = np.mean(parts, axis=0)
        out.append(pd.DataFrame(res))
        if verbose:
            print(f"  [nextday] {Y}: train {len(t_tr):,}  val {len(t_va):,}  test {len(t_te):,}", flush=True)
    return pd.concat(out).set_index(["date", "ticker"]).sort_index()


def evaluate(pred: pd.DataFrame, model: str, top_n: int = 10, cost: float = 0.0025) -> dict:
    """Out-of-sample skill of one model's next-day scores."""
    from sklearn.metrics import roc_auc_score

    d = pred.dropna(subset=["y", "ret_next", model])
    s = d[model]
    acc = float(((s > (0.5 if model != "reversal" else 0)) == (d["y"] > 0.5)).mean()) if model != "reversal" else np.nan
    g = d.groupby(level=0)
    ic = g.apply(lambda x: x[model].rank().corr(x["ret_next"].rank()) if len(x) > 20 else np.nan).dropna()
    top = g.apply(lambda x: x.nlargest(top_n, model)["ret_next"].mean() - x["ret_next"].mean())
    hit = g.apply(lambda x: x.nlargest(top_n, model)["y"].mean())
    topret = g.apply(lambda x: x.nlargest(top_n, model)["ret_next"].mean())
    return {"AUC": float(roc_auc_score(d["y"], s)), "Accuracy": acc, "IC mean": float(ic.mean()),
            "IC t-stat": float(ic.mean() / ic.std() * np.sqrt(len(ic))), f"Top{top_n} hit rate": float(hit.mean()),
            f"Top{top_n} excess / day": float(top.mean()),
            f"Top{top_n} net of costs / day": float(topret.mean() - cost),
            "Days": int(len(ic))}
