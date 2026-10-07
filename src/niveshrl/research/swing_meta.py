"""Meta-labeling the swing picks (Track record v2).

The daily 'watch for strength' list is the *primary* model; this is the *secondary* one (López de Prado's
meta-labeling): given that a stock was picked, how good is this particular trade likely to be? It learns from the
~9,000 point-in-time historical trades in ``monitor_history_tagged.parquet``: every pick and every random-control
stock from 2015 to 2026, followed with the exact plan rules (entry next open, stop, a third at T1, 3×ATR trail,
60-day cap, delivery costs), so the label is each trade's real R after costs.

Features, all known at the signal date's close: the stock's recent returns, trend (vs its 50/200-day averages), RSI,
ATR %, relative volume, distance from its 20-day high, 60-day volatility; the market's 1-month return, trend, breadth
and VIX; the research models' views (next-day ensemble probability, next-day range forecast, and the FFNN / LSTM /
Transformer / momentum monthly ranks, each as a cross-sectional percentile); and the context tags used by 'similar
setups'.

Models: LightGBM and a small neural network (2 × 64 units, dropout, early stopping), averaged. Walk-forward by
year: the models predicting year Y are trained only on trades whose outcome was known (exit date) before 1 January Y.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ROOT
from .data import Panel

HIST = ROOT / "report" / "results" / "monitor_history_tagged.parquet"
PRED = ROOT / "data" / "predictions"
NUM = ["ret_1d", "ret_5d", "ret_1m", "ret_3m", "vs_sma50", "vs_sma200", "rsi14", "atr_pct", "vol_ratio", "from_20d_high",
       "vol_60d", "mkt_ret_1m", "mkt_vs_200", "breadth50", "vix", "p_nextday", "range_rank", "ffnn_rank", "lstm_rank",
       "transformer_rank", "momentum_rank"]
TAGS = {"trend": ["above", "below"], "rsi_zone": ["<40", "40-60", "60-70", ">70"],
        "near_high": ["within 2%", "2-8% below", "over 8% below"], "market": ["above", "below"]}


def _rsi(c: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _lookup(frame: pd.DataFrame, dates, tickers) -> np.ndarray:
    """frame[date, ticker] for many (date, ticker) pairs, using the last available date ≤ each date."""
    pos = frame.index.searchsorted(pd.DatetimeIndex(dates), side="right") - 1
    col = frame.columns.get_indexer(list(tickers))
    vals = frame.to_numpy(dtype=float)
    out = np.full(len(pos), np.nan)
    ok = (pos >= 0) & (col >= 0)
    out[ok] = vals[pos[ok], col[ok]]
    return out


def _rank_frame(path, col: str, date_name: str = "date", ticker_name: str = "ticker") -> pd.DataFrame | None:
    f = PRED / path
    if not f.exists():
        return None
    s = pd.read_parquet(f)[col]
    s.index = s.index.set_names([date_name, ticker_name])
    w = s.unstack()
    return w.rank(axis=1, pct=True)


def features(p: Panel, hist: pd.DataFrame) -> pd.DataFrame:
    c, h = p.close, p.high if p.high is not None else p.close
    d, t = hist["signal_date"], hist["ticker"]
    r = c.pct_change(fill_method=None)
    tr = pd.concat([h - (p.low if p.low is not None else c), (h - c.shift()).abs(),
                    ((p.low if p.low is not None else c) - c.shift()).abs()]).groupby(level=0).max()
    f = pd.DataFrame(index=hist.index)
    f["ret_1d"] = _lookup(r, d, t)
    for k, n in (("ret_5d", 5), ("ret_1m", 21), ("ret_3m", 63)):
        f[k] = _lookup(c / c.shift(n) - 1, d, t)
    f["vs_sma50"] = _lookup(c / c.rolling(50, min_periods=40).mean() - 1, d, t)
    f["vs_sma200"] = _lookup(c / c.rolling(200, min_periods=150).mean() - 1, d, t)
    f["rsi14"] = _lookup(_rsi(c), d, t)
    f["atr_pct"] = _lookup(tr.reindex(c.index).rolling(14).mean() / c, d, t)
    f["vol_ratio"] = _lookup(p.volume / p.volume.rolling(20).mean(), d, t)
    f["from_20d_high"] = _lookup(c / h.rolling(20).max() - 1, d, t)
    f["vol_60d"] = _lookup(r.rolling(60).std() * np.sqrt(252), d, t)
    b = p.bench
    bm = pd.DataFrame({"m": b / b.shift(21) - 1, "t": b / b.rolling(200).mean() - 1,
                       "br": (c > c.rolling(50, min_periods=40).mean()).where(c.notna()).mean(axis=1),
                       "vix": p.vix.reindex(c.index).ffill()})
    pos = bm.index.searchsorted(pd.DatetimeIndex(d), side="right") - 1
    for k, col in (("mkt_ret_1m", "m"), ("mkt_vs_200", "t"), ("breadth50", "br"), ("vix", "vix")):
        f[k] = bm[col].to_numpy()[np.maximum(pos, 0)]
    for name, path, col, dn, tn in (("p_nextday", "nextday_pit.parquet", "ensemble", "date", "ticker"),
                                    ("range_rank", "range_pit.parquet", "pred", "Date", "Ticker"),
                                    ("ffnn_rank", "ffnn_pit.parquet", "score", "date", "ticker"),
                                    ("lstm_rank", "lstm_pit.parquet", "score", "date", "ticker"),
                                    ("transformer_rank", "transformer_pit.parquet", "score", "date", "ticker"),
                                    ("momentum_rank", "momentum_pit.parquet", "score", "date", "ticker")):
        w = _rank_frame(path, col, dn, tn)
        f[name] = _lookup(w, d, t) if w is not None else np.nan
    for tag, levels in TAGS.items():
        for lv in levels:
            f[f"{tag}={lv}"] = (hist[tag] == lv).astype(float).to_numpy() if tag in hist else 0.0
    return f


class _MLP:
    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X, y, Xv, yv):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        self.mu, self.sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-6
        z = lambda a: torch.from_numpy(np.nan_to_num((a - self.mu) / self.sd).astype(np.float32))  # noqa: E731
        self.net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 64), nn.GELU(),
                                 nn.Dropout(0.2), nn.Linear(64, 1))
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-3)
        Xt, yt, Xvt, yvt = z(X), torch.from_numpy(y.astype(np.float32)), z(Xv), torch.from_numpy(yv.astype(np.float32))
        lossf = nn.HuberLoss()
        best, state, bad = np.inf, None, 0
        for _ in range(200):
            self.net.train()
            perm = torch.randperm(len(yt))
            for i in range(0, len(yt), 256):
                j = perm[i:i + 256]
                loss = lossf(self.net(Xt[j]).squeeze(-1), yt[j])
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = lossf(self.net(Xvt).squeeze(-1), yvt).item()
            if v < best - 1e-5:
                best, state, bad = v, {k: t.clone() for k, t in self.net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 10:
                    break
        self.net.load_state_dict(state)
        self._z = z
        return self

    def predict(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            return self.net(self._z(X)).squeeze(-1).numpy()


def walk_forward(hist: pd.DataFrame, F: pd.DataFrame, first_year: int = 2017, seed: int = 0) -> pd.DataFrame:
    """Out-of-sample predicted R for every trade from ``first_year``: models for year Y see only trades that had
    closed before 1 January Y."""
    import lightgbm as lgb
    cols = list(F.columns)
    y = hist["r"].clip(-1.5, 5).to_numpy(float)
    out = pd.DataFrame(index=hist.index, columns=["pred_lgbm", "pred_mlp"], dtype=float)
    for Y in range(first_year, hist["signal_date"].dt.year.max() + 1):
        start = pd.Timestamp(f"{Y}-01-01")
        tr = (hist["exit_date"] < start).to_numpy()
        te = (hist["signal_date"].dt.year == Y).to_numpy()
        if tr.sum() < 1000 or not te.any():
            continue
        X = F[cols].to_numpy(float)
        cut = hist.loc[tr, "signal_date"].quantile(0.85)             # last 15% of the training trades for early stopping
        fit_i = tr & (hist["signal_date"] < cut).to_numpy()
        val_i = tr & (hist["signal_date"] >= cut).to_numpy()
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=80, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, verbose=-1)
        m.fit(X[fit_i], y[fit_i], eval_set=[(X[val_i], y[val_i])], callbacks=[lgb.early_stopping(50, verbose=False)])
        out.loc[te, "pred_lgbm"] = m.predict(X[te])
        nn_ = _MLP(seed).fit(X[fit_i], y[fit_i], X[val_i], y[val_i])
        out.loc[te, "pred_mlp"] = nn_.predict(X[te])
    out["pred"] = out[["pred_lgbm", "pred_mlp"]].mean(axis=1)
    return out


def evaluate(hist: pd.DataFrame, pred: pd.DataFrame, keep: float = 0.5) -> dict:
    """Monitor-list trades in the test years: all of them vs keeping the best ``keep`` share of each day's picks by
    predicted R (and vs the random control)."""
    d = hist.join(pred).dropna(subset=["pred"])
    mon, rnd = d[d["group"] == "monitor list"], d[d["group"] == "random control"]
    rk = mon.groupby("signal_date")["pred"].rank(pct=True, ascending=False)
    kept = mon[rk <= keep]
    dropped = mon[rk > keep]
    def stats(x):
        by = x.groupby("signal_date")["r"].mean()
        return {"trades": int(len(x)), "avg R": float(x["r"].mean()), "win": float((x["r"] > 0).mean()),
                "t (per signal day)": float(by.mean() / by.std() * np.sqrt(len(by))) if by.std() > 0 else np.nan}
    a, b = kept.groupby("signal_date")["r"].mean(), mon.groupby("signal_date")["r"].mean()
    j = pd.concat([a.rename("k"), b.rename("a")], axis=1).dropna()
    diff = j["k"] - j["a"]
    return {"all picks": stats(mon), f"best {keep:.0%} by meta-label": stats(kept), "dropped picks": stats(dropped),
            "random control": stats(rnd),
            "kept − all": {"diff R": float(diff.mean()), "t": float(diff.mean() / diff.std() * np.sqrt(len(diff)))
                           if diff.std() > 0 else np.nan, "days": int(len(diff))},
            "corr(pred, R)": float(d["pred"].corr(d["r"], method="spearman"))}
