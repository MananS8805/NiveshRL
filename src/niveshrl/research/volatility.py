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


def build_vol_data(p: Panel) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows (date, ticker): static features + target. Plus a sequence array (n, L, 2)."""
    r = p.returns()
    me = p.month_ends()
    pos = r.index.get_indexer(me)
    bench_vol = np.log(p.bench).diff().rolling(20).std() * ANN
    rows, seqs = [], []
    R = r.to_numpy()
    for i, t in enumerate(me):
        k = pos[i]
        if k < 70 or k + H >= len(r):
            continue
        past = R[k - L + 1:k + 1]                  # (L, N)
        fut = R[k + 1:k + 1 + H]
        ok = (np.isfinite(past).sum(0) >= L - 3) & (np.isfinite(fut).sum(0) >= H - 3)
        if ok.sum() < 20:
            continue
        cols = np.flatnonzero(ok)
        past_c = np.nan_to_num(past[:, cols])
        fut_c = fut[:, cols]
        rv_next = np.nanstd(fut_c, axis=0, ddof=0) * ANN
        rv21 = past_c[-21:].std(0) * ANN
        rv63 = np.nanstd(R[k - 62:k + 1, cols], axis=0) * ANN
        df = pd.DataFrame({
            "rv21": rv21, "rv63": rv63, "rv_next": rv_next,
            "vix": float(p.vix.iloc[k]) / 100 if np.isfinite(p.vix.iloc[k]) else 0.18,
            "mkt_vol": float(bench_vol.iloc[k]) if np.isfinite(bench_vol.iloc[k]) else 0.18,
        }, index=pd.MultiIndex.from_product([[t], r.columns[cols]], names=["date", "ticker"]))
        keep = ((df["rv_next"] > 0) & (df["rv21"] > 0)).to_numpy()   # drop frozen/zero-vol rows
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


def lstm_forecast(frame: pd.DataFrame, seq: np.ndarray, first_year: int, window_years: int = 8,
                  epochs: int = 30, seed: int = 0, verbose: bool = True) -> pd.Series:
    torch.manual_seed(seed)
    dates = frame.index.get_level_values(0)
    y_all = np.log(frame["rv_next"].to_numpy()).astype(np.float32)
    st_all = _static(frame)
    out = pd.Series(np.nan, index=frame.index)
    for Y in sorted({d.year for d in dates if d.year >= first_year}):
        tr = (dates >= pd.Timestamp(f"{Y - window_years}-01-01")) & (dates < pd.Timestamp(f"{Y - 1}-01-01"))
        va = (dates >= pd.Timestamp(f"{Y - 1}-01-01")) & (dates < pd.Timestamp(f"{Y}-01-01"))
        te = dates.year == Y
        # Train rows dated before Y-1; their 21-day targets end before the validation year starts.
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
