"""Next-month volatility forecasting: LSTM vs GARCH(1,1), EWMA and historical.

Target: at month-end t, the annualised realised volatility of the stock's
daily returns over the next 21 trading days. Models are scored on log
volatility (RMSE), on QLIKE (the standard robust loss for variance
forecasts), and by the correlation between forecast and realised.

- ``hist``: last 21-day realised vol (random-walk forecast).
- ``ewma``: RiskMetrics exponentially weighted vol, lambda = 0.94.
- ``garch``: GARCH(1,1) with constant mean, fit per stock once a year on the
  4 years before the test year (``arch``). The fixed parameters are then run
  forward through the data up to t, and the 21-day-ahead variance is
  summed. The recursion is done in NumPy for speed; only the parameter fits
  call ``arch``.
- ``lstm``: pooled LSTM over the last 60 daily returns (r and |r|), plus
  static context: log realised vol over 21 and 63 days, India VIX, and
  NIFTY 20-day vol. Trained walk-forward with a rolling window, exactly
  like the rankers.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import Panel
from .rankers import PRED_DIR

H = 21        # forecast horizon, trading days
L = 60        # LSTM lookback, trading days
ANN = np.sqrt(252)
VOL_MODELS = ["hist", "ewma", "garch", "lstm"]


def build_vol_data(p: Panel, live: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows (date, ticker): static features + target. Plus a sequence array (n, L, 2).

    ``live=True`` also returns rows whose next 21 days have not happened yet (the latest month-end(s) and the last
    trading day) with ``rv_next`` = NaN, so a forecast can be made for *now*. Without it, the newest forecast is
    always at least a month old, because a month-end only gets a row once its future is known."""
    r = p.returns()
    me = p.month_ends()
    if live and r.index[-1] not in me:
        me = me.append(pd.DatetimeIndex([r.index[-1]]))
    pos = r.index.get_indexer(me)
    bench_vol = np.log(p.bench).diff().rolling(20).std() * ANN
    rows, seqs = [], []
    R = r.to_numpy()
    for i, t in enumerate(me):
        k = pos[i]
        has_future = k + H < len(r)
        if k < 70 or (not has_future and not live):
            continue
        past = R[k - L + 1:k + 1]                  # (L, N)
        fut = R[k + 1:k + 1 + H] if has_future else np.full((H, R.shape[1]), np.nan)
        ok = (np.isfinite(past).sum(0) >= L - 3) & ((np.isfinite(fut).sum(0) >= H - 3) | (not has_future))
        if ok.sum() < 20:
            continue
        cols = np.flatnonzero(ok)
        past_c = np.nan_to_num(past[:, cols])
        fut_c = fut[:, cols]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rv_next = np.nanstd(fut_c, axis=0, ddof=0) * ANN if has_future else np.full(len(cols), np.nan)
        rv21 = past_c[-21:].std(0) * ANN
        rv63 = np.nanstd(R[k - 62:k + 1, cols], axis=0) * ANN
        df = pd.DataFrame({
            "rv21": rv21, "rv63": rv63, "rv_next": rv_next,
            "vix": float(p.vix.iloc[k]) / 100 if np.isfinite(p.vix.iloc[k]) else 0.18,
            "mkt_vol": float(bench_vol.iloc[k]) if np.isfinite(bench_vol.iloc[k]) else 0.18,
        }, index=pd.MultiIndex.from_product([[t], r.columns[cols]], names=["date", "ticker"]))
        keep = (((df["rv_next"] > 0) | df["rv_next"].isna()) & (df["rv21"] > 0)).to_numpy()   # drop frozen rows
        rows.append(df[keep])
        seqs.append(np.stack([past_c.T * 100, np.abs(past_c.T) * 100], -1)[keep])   # (n, L, 2), in %
    return pd.concat(rows), np.concatenate(seqs)


def ewma_forecast(p: Panel, frame: pd.DataFrame, lam: float = 0.94) -> pd.Series:
    r = p.returns()
    var = (r.fillna(0) ** 2).ewm(alpha=1 - lam, adjust=False).mean()
    v = var.stack().reindex(frame.index)
    return np.sqrt(v) * ANN


def _garch_path(eps: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    s2 = np.empty(len(eps) + 1)
    s2[0] = np.var(eps) if len(eps) > 1 else omega / max(1e-6, 1 - alpha - beta)
    for i, e in enumerate(eps):
        s2[i + 1] = omega + alpha * e * e + beta * s2[i]
    return s2          # s2[i+1] = conditional variance for day i+1 given data through i


def garch_forecast(p: Panel, frame: pd.DataFrame, first_year: int, fit_years: int = 4) -> pd.Series:
    from arch import arch_model

    r = p.returns() * 100
    out = pd.Series(np.nan, index=frame.index)
    dates = frame.index.get_level_values(0)
    years = sorted({d.year for d in dates if d.year >= first_year})
    for tk in r.columns:
        s = r[tk].dropna()
        if len(s) < 300:
            continue
        for Y in years:
            fit = s[(s.index >= pd.Timestamp(f"{Y - fit_years}-01-01")) & (s.index < pd.Timestamp(f"{Y}-01-01"))]
            if len(fit) < 250:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    res = arch_model(fit, mean="Constant", vol="GARCH", p=1, q=1).fit(disp="off")
                except Exception:
                    continue
            mu, om, a, b = (res.params.get(k, np.nan) for k in ("mu", "omega", "alpha[1]", "beta[1]"))
            if not np.isfinite([mu, om, a, b]).all() or a + b >= 0.9999:
                continue
            upto = s[s.index < pd.Timestamp(f"{Y + 1}-01-01")]
            eps = (upto - mu).to_numpy()
            s2 = _garch_path(eps, om, a, b)
            lr = om / (1 - a - b)
            idx_of = {d: i for i, d in enumerate(upto.index)}
            for (d, t) in frame.index[(dates.year == Y) & (frame.index.get_level_values(1) == tk)]:
                i = idx_of.get(d)
                if i is None:
                    continue
                nxt = s2[i + 1]
                h = np.arange(H)
                var_sum = (lr + (a + b) ** h * (nxt - lr)).sum()      # sum of 1..H step variances, in %^2
                out[(d, t)] = np.sqrt(var_sum / H) / 100 * ANN
    return out


class VolLSTM(nn.Module):
    def __init__(self, hidden: int = 32, n_static: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(2, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, seq, static):
        _, (h, _) = self.lstm(seq)
        return self.head(torch.cat([h[-1], static], -1)).squeeze(-1)


def _static(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.log(frame["rv21"]), np.log(frame["rv63"].clip(lower=1e-4)),
                            frame["vix"], frame["mkt_vol"]]).astype(np.float32)


def _periods(dates: pd.DatetimeIndex, first_year: int, refit: str, window_years: int):
    """(train, validation, test) masks per refit period. 'yearly': test year Y, train [Y-w, Y-1), validate Y-1.
    'monthly': test month M, train the window up to 13 months before M, validate the 12 months before M (minus the
    last month, whose 21-day targets end inside M)."""
    if refit == "yearly":
        for Y in sorted({d.year for d in dates if d.year >= first_year}):
            yield ((dates >= pd.Timestamp(f"{Y - window_years}-01-01")) & (dates < pd.Timestamp(f"{Y - 1}-01-01")),
                   (dates >= pd.Timestamp(f"{Y - 1}-01-01")) & (dates < pd.Timestamp(f"{Y}-01-01")), dates.year == Y, Y)
        return
    months = pd.period_range(pd.Timestamp(f"{first_year}-01-01"), dates.max(), freq="M").to_timestamp()
    for M in months:
        v_end = M - pd.DateOffset(months=1)                     # targets of rows dated before this are known by M
        v0 = M - pd.DateOffset(months=13)
        yield ((dates >= M - pd.DateOffset(years=window_years)) & (dates < v0), (dates >= v0) & (dates < v_end),
               (dates >= M) & (dates < M + pd.DateOffset(months=1)), f"{M:%Y-%m}")


def lstm_forecast(frame: pd.DataFrame, seq: np.ndarray, first_year: int, window_years: int = 8,
                  epochs: int = 30, seed: int = 0, verbose: bool = True, refit: str = "yearly") -> pd.Series:
    torch.manual_seed(seed)
    dates = frame.index.get_level_values(0)
    y_all = np.log(frame["rv_next"].to_numpy()).astype(np.float32)
    labelled = np.isfinite(y_all)
    st_all = _static(frame)
    out = pd.Series(np.nan, index=frame.index)
    for tr, va, te, Y in _periods(dates, first_year, refit, window_years):
        tr, va = tr & labelled, va & labelled
        if tr.sum() < 1000 or va.sum() < 100 or not te.any():
            continue
        # Train rows end before validation starts; their 21-day targets end before the validation period.
        m = VolLSTM()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
        S, X, Yt = (torch.from_numpy(a) for a in (seq[tr].astype(np.float32), st_all[tr], y_all[tr]))
        Sv, Xv, Yv = (torch.from_numpy(a) for a in (seq[va].astype(np.float32), st_all[va], y_all[va]))
        best, state, bad = np.inf, None, 0
        for _ in range(epochs):
            m.train()
            perm = torch.randperm(len(Yt))
            for s in range(0, len(Yt), 512):
                i = perm[s:s + 512]
                loss = ((m(S[i], X[i]) - Yt[i]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                v = ((m(Sv, Xv) - Yv) ** 2).mean().item()
            if v < best - 1e-5:
                best, state, bad = v, {k: t.clone() for k, t in m.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 4:
                    break
        m.load_state_dict(state)
        m.eval()
        with torch.no_grad():
            out[te] = np.exp(m(torch.from_numpy(seq[te].astype(np.float32)), torch.from_numpy(st_all[te])).numpy())
        if verbose:
            print(f"  [vol-lstm] {Y}: val RMSE(log vol) {np.sqrt(best):.3f}", flush=True)
    return out


def vol_metrics(pred: pd.Series, realised: pd.Series) -> dict:
    d = pd.concat([pred.rename("f"), realised.rename("r")], axis=1).dropna()
    d = d[(d.f > 0) & (d.r > 0)]
    lf, lr = np.log(d.f), np.log(d.r)
    qlike = (d.r ** 2 / d.f ** 2 - np.log(d.r ** 2 / d.f ** 2) - 1).mean()
    return {"RMSE log-vol": float(np.sqrt(((lf - lr) ** 2).mean())),
            "MAE vol": float((d.f - d.r).abs().mean()),
            "QLIKE": float(qlike),
            "Corr": float(np.corrcoef(lf, lr)[0, 1]),
            "Bias (f/r)": float((d.f / d.r).median()),
            "N": int(len(d))}


def save_vol(frame: pd.DataFrame) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(PRED_DIR / "vol_forecasts.parquet")


def load_vol() -> pd.DataFrame | None:
    path = PRED_DIR / "vol_forecasts.parquet"
    return pd.read_parquet(path) if path.exists() else None


def update_live(p: Panel, window_years: int = 8, seed: int = 0) -> pd.DataFrame:
    """Today's next-month volatility forecast for every stock (run daily by the pipeline).

    The LSTM is refit once a month (cached in data/models/vol_lstm_YYYY-MM.pt) on the window up to the last month
    whose 21-day outcomes are known, then applied to today's row. Results are merged into vol_forecasts.parquet
    (replacing any earlier rows for the same dates), with hist and EWMA alongside."""
    from ..config import ROOT
    frame, seq = build_vol_data(p, live=True)
    dates = frame.index.get_level_values(0)
    today = dates.max()
    M = today.to_period("M").to_timestamp()
    cache = ROOT / "data" / "models" / f"vol_lstm_{M:%Y-%m}.pt"
    st_all = _static(frame)
    te = (dates >= M) & (dates <= today)
    m = VolLSTM()
    if cache.exists():
        m.load_state_dict(torch.load(cache))
    else:
        torch.manual_seed(seed)
        y_all = np.log(frame["rv_next"].to_numpy()).astype(np.float32)
        v_end, v0 = M - pd.DateOffset(months=1), M - pd.DateOffset(months=13)
        lab = np.isfinite(y_all)
        tr = (dates >= M - pd.DateOffset(years=window_years)) & (dates < v0) & lab
        va = (dates >= v0) & (dates < v_end) & lab
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
        S, X, Yt = (torch.from_numpy(a) for a in (seq[tr].astype(np.float32), st_all[tr], y_all[tr]))
        Sv, Xv, Yv = (torch.from_numpy(a) for a in (seq[va].astype(np.float32), st_all[va], y_all[va]))
        best, state, bad = np.inf, None, 0
        for _ in range(30):
            m.train()
            perm = torch.randperm(len(Yt))
            for s in range(0, len(Yt), 512):
                i = perm[s:s + 512]
                loss = ((m(S[i], X[i]) - Yt[i]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                v = ((m(Sv, Xv) - Yv) ** 2).mean().item()
            if v < best - 1e-5:
                best, state, bad = v, {k: t.clone() for k, t in m.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 4:
                    break
        m.load_state_dict(state)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, cache)
    m.eval()
    with torch.no_grad():
        pred = np.exp(m(torch.from_numpy(seq[te].astype(np.float32)), torch.from_numpy(st_all[te])).numpy())
    new = frame[te].copy()
    new["hist"] = new["rv21"]
    new["ewma"] = ewma_forecast(p, new).to_numpy()
    new["lstm"] = pred
    old = load_vol()
    if old is not None:
        old = old[~old.index.get_level_values(0).isin(new.index.get_level_values(0).unique())]
        new = pd.concat([old, new.reindex(columns=old.columns)]).sort_index()
    save_vol(new)
    return new.xs(today, level=0)
