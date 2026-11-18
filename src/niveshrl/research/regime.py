"""Market regime detection: a 2-D embedding (PCA in the app; the original autoencoder for research) + k-means,
walk-forward.

Every week (Friday close) we describe the market with 8 features: NIFTY
20-day return, 20- and 60-day volatility, India VIX level and 20-day
change, breadth (the share of stocks above their 200-day average),
cross-sectional dispersion of 20-day stock returns, and NIFTY's drawdown
from its 1-year high.

Volatility, VIX and dispersion enter *relative to their own trailing
1-year median* (log ratio). With raw levels, which drift over the years,
the walk-forward model treated whole calendar years as one regime (all of
2021-2023 came out "Neutral"). Relative levels describe the week's state
instead.

For each year Y, a small autoencoder (8 -> 16 -> 2 -> 16 -> 8) and a
3-cluster k-means on its 2-D embedding are fit on weeks *before* Y only
(expanding window). The fitted model then labels the weeks of Y. Clusters
are named from their training-set averages: the highest-volatility cluster
is **Stress**, and of the other two the one with the higher 20-day return
is **Bull** and the other **Neutral**. No label uses information from the
week it describes onward.

The embedding is also exported (``z1``, ``z2``) for the regime map in the
dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from ..config import ROOT
from .data import Panel

PRED_DIR = ROOT / "data" / "predictions"          # same folder as research/rankers.py (no torch import needed)

REGIMES = ["Bull", "Neutral", "Stress"]
REGIME_COLORS = {"Bull": "#2ecc71", "Neutral": "#f1c40f", "Stress": "#e74c3c"}
FEATS = ["nifty_ret_20", "nifty_vol_20", "nifty_vol_60", "vix", "vix_chg_20", "breadth", "dispersion", "nifty_dd"]


def market_features(p: Panel) -> pd.DataFrame:
    lb = np.log(p.bench.ffill())
    br = lb.diff()
    c = p.close
    ma200 = c.rolling(200, min_periods=150).mean()
    above = (c > ma200).where(ma200.notna())
    r20 = c / c.shift(20) - 1
    rel = lambda x: np.log(x / x.rolling(252, min_periods=120).median())  # noqa: E731
    vol20 = br.rolling(20).std() * np.sqrt(252)
    f = pd.DataFrame({
        "nifty_ret_20": lb.diff(20),
        "nifty_vol_20": rel(vol20),
        "nifty_vol_60": rel(br.rolling(60).std() * np.sqrt(252)),
        "vix": rel(p.vix),
        "vix_chg_20": np.log(p.vix).diff(20),
        "breadth": above.mean(axis=1),
        "dispersion": rel(r20.std(axis=1)),
        "nifty_dd": p.bench / p.bench.rolling(252, min_periods=60).max() - 1,
    })
    f["vol_level"] = vol20          # raw level: kept for naming clusters, not a model input
    return f.resample("W-FRI").last().dropna()


def AE(n: int = len(FEATS), z: int = 2):
    """The research autoencoder (torch is imported only here; the app uses ``encoder="pca"``)."""
    import torch.nn as nn

    class _AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(n, 16), nn.Tanh(), nn.Linear(16, z))
            self.dec = nn.Sequential(nn.Linear(z, 16), nn.Tanh(), nn.Linear(16, n))

        def forward(self, x):
            zz = self.enc(x)
            return self.dec(zz), zz

    return _AE()


def _fit(X: np.ndarray, seed: int = 0, epochs: int = 600):
    import torch
    torch.manual_seed(seed)
    m = AE()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    x = torch.from_numpy(X.astype(np.float32))
    for _ in range(epochs):
        rec, _ = m(x)
        loss = ((rec - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m.eval()


class NumpyAE:
    """The same autoencoder (8 -> 16 -> 2 -> 16 -> 8, tanh, full-batch Adam lr 1e-2, 600 epochs, PyTorch's default
    uniform init) in plain NumPy, so the app needs no torch. Its labels match the torch version about as well as the
    torch version matches itself across seeds (report/results/regime_pca_check.md)."""

    def __init__(self, n: int = len(FEATS), z: int = 2, h: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        shapes = [(n, h), (h, z), (z, h), (h, n)]
        self.W = [rng.uniform(-1 / np.sqrt(a), 1 / np.sqrt(a), (a, b)) for a, b in shapes]
        self.b = [rng.uniform(-1 / np.sqrt(a), 1 / np.sqrt(a), b) for a, b in shapes]

    def encode(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(X @ self.W[0] + self.b[0]) @ self.W[1] + self.b[1]

    def fit(self, X: np.ndarray, epochs: int = 600, lr: float = 1e-2, b1: float = 0.9, b2: float = 0.999,
            eps: float = 1e-8) -> "NumpyAE":
        params = self.W + self.b
        m = [np.zeros_like(q) for q in params]
        v = [np.zeros_like(q) for q in params]
        N = X.size
        for t in range(1, epochs + 1):
            a1 = np.tanh(X @ self.W[0] + self.b[0])
            zz = a1 @ self.W[1] + self.b[1]
            a3 = np.tanh(zz @ self.W[2] + self.b[2])
            out = a3 @ self.W[3] + self.b[3]
            g_out = 2 * (out - X) / N                         # d mean-squared-error
            gW4, gb4 = a3.T @ g_out, g_out.sum(0)
            g3 = (g_out @ self.W[3].T) * (1 - a3 ** 2)
            gW3, gb3 = zz.T @ g3, g3.sum(0)
            gz = g3 @ self.W[2].T
            gW2, gb2 = a1.T @ gz, gz.sum(0)
            g1 = (gz @ self.W[1].T) * (1 - a1 ** 2)
            gW1, gb1 = X.T @ g1, g1.sum(0)
            grads = [gW1, gW2, gW3, gW4, gb1, gb2, gb3, gb4]
            for i, (q, g) in enumerate(zip(params, grads)):
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * g * g
                q -= lr * (m[i] / (1 - b1 ** t)) / (np.sqrt(v[i] / (1 - b2 ** t)) + eps)
        return self


def _embed(train_z: np.ndarray, test_z: np.ndarray, encoder: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """2-D embedding of the standardised features, fit on the training weeks only."""
    train_z, test_z = np.asarray(train_z, dtype=float), np.asarray(test_z, dtype=float)
    if encoder == "ae_numpy":
        ae = NumpyAE(seed=seed).fit(train_z)
        return ae.encode(train_z), ae.encode(test_z)
    if encoder == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(2, random_state=seed).fit(train_z)
        return pca.transform(train_z), pca.transform(test_z)
    import torch
    ae = _fit(train_z, seed)
    with torch.no_grad():
        return (ae.enc(torch.from_numpy(train_z.astype(np.float32))).numpy(),
                ae.enc(torch.from_numpy(test_z.astype(np.float32))).numpy())


def detect_regimes(p: Panel, first_year: int = 2010, seed: int = 0, window_years: int | None = None,
                   encoder: str = "ae") -> pd.DataFrame:
    """``window_years=None``: expanding window (all history before Y); an int: only the last ``window_years``.
    ``encoder``: "ae" (the original torch autoencoder), "ae_numpy" (the same network in NumPy, no torch) or "pca"
    (rejected: different labels, report/results/regime_pca_check.md)."""
    from sklearn.cluster import KMeans

    f = market_features(p)
    out = []
    for Y in sorted({d.year for d in f.index if d.year >= first_year}):
        lo = pd.Timestamp(f"{Y - window_years}-01-01") if window_years else f.index[0]
        train = f.loc[(f.index >= lo) & (f.index < pd.Timestamp(f"{Y}-01-01")), FEATS]
        test = f.loc[f.index.year == Y, FEATS]
        if len(train) < 104 or test.empty:
            continue
        mu, sd = train.mean(), train.std().replace(0, 1)
        z_tr, z_te = _embed(((train - mu) / sd).to_numpy(), ((test - mu) / sd).to_numpy(), encoder, seed)
        km = KMeans(3, n_init=10, random_state=seed).fit(z_tr)
        stats = train.assign(c=km.labels_, vol_level=f["vol_level"].reindex(train.index))             .groupby("c")[["vol_level", "nifty_ret_20"]].mean()
        stress = stats["vol_level"].idxmax()
        rest = stats.drop(index=stress).sort_values("nifty_ret_20")
        names = {stress: "Stress", rest.index[-1]: "Bull", rest.index[0]: "Neutral"}
        lab = [names[c] for c in km.predict(z_te)]
        out.append(test.assign(regime=lab, z1=z_te[:, 0], z2=z_te[:, 1],
                               vol_level=f["vol_level"].reindex(test.index)))
    return pd.concat(out)


def daily_regimes(weekly: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Label known at each trading day: the latest completed week's regime."""
    return weekly["regime"].reindex(index, method="ffill")


def regime_stats(weekly: pd.DataFrame, p: Panel) -> pd.DataFrame:
    """Out-of-sample check: NIFTY's *next* 4-week return and volatility, by regime."""
    b = p.bench.resample("W-FRI").last()
    lb = np.log(b)
    fwd_ret = (lb.shift(-4) - lb).reindex(weekly.index)
    daily = np.log(p.bench).diff()
    fwd_vol = daily.rolling(20).std().shift(-20).resample("W-FRI").last().reindex(weekly.index) * np.sqrt(252)
    d = weekly.assign(fwd_ret=fwd_ret, fwd_vol=fwd_vol)
    g = d.groupby("regime")
    return pd.DataFrame({"weeks": g.size(), "share": g.size() / len(d),
                         "next 4w NIFTY return": g["fwd_ret"].mean(),
                         "next-month NIFTY vol": g["fwd_vol"].mean(),
                         "hit rate (4w > 0)": g["fwd_ret"].apply(lambda s: (s > 0).mean())}).reindex(REGIMES)


def save_regimes(weekly: pd.DataFrame) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(PRED_DIR / "regimes.parquet")


def load_regimes() -> pd.DataFrame | None:
    path = PRED_DIR / "regimes.parquet"
    return pd.read_parquet(path) if path.exists() else None
