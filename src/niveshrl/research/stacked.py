"""Rolling (monthly) refits of the next-day models and a 4-model stacked next-day predictor.

**Why.** ``nextday.walk_forward`` refits once a year on years ``[Y-5, Y-2]`` and calibrates on ``Y-1``, so every
prediction made in year Y comes from a model that has seen nothing after December of ``Y-2``: up to 21 months stale.
Here every model is refit at the start of each month on the last ``train_years`` years up to ``val_months`` before
that month, and calibrated on those last ``val_months`` months. Whether that actually predicts better is measured in
``compare`` (out of sample), not assumed.

**Base models** (each refit monthly, each predicting for the month after its training data ends):

1. ``lgbm``   LightGBM classifier on the tabular features, isotonic-calibrated: P(beats tomorrow's median).
2. ``seq``    the transformer sequence net over the last 20 days + tabular features, isotonic-calibrated.
3. ``logreg`` L2 logistic regression on the same features (a stable linear view).
4. ``range``  LightGBM regressor of tomorrow's high-low range (from ``range_model``): how big the move may be.

**Stacking.** A meta model (logistic regression) combines the four base models' cross-sectional *ranks* (plus the
interaction of direction with expected range). It is trained only on the base models' past *out-of-sample*
predictions from the previous ``meta_months`` months, never on predictions for days the base models trained on, so
the stack cannot leak. Output per stock per day: ``p_up`` (stacked P(beats tomorrow's median)), ``exp_range`` and a
``pattern`` label (direction × size): e.g. "Up · volatile", "Flat · quiet". ``pattern_changes`` lists the stocks whose
pattern changed since the previous close.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import ROOT
from . import nextday as nd

MODEL_DIR = ROOT / "data" / "models" / "nextday_rolling"
BASE_PATH = ROOT / "data" / "predictions" / "nextday_rolling.parquet"      # base OOS predictions, grows daily
STACK_PATH = ROOT / "data" / "predictions" / "stacked.parquet"             # stacked OOS predictions
BASES = ["lgbm", "seq", "logreg", "range"]
UP, DOWN = 0.53, 0.47                    # stacked P(up) thresholds for the direction part of the pattern


# --------------------------------------------------------------------------- base models, refit monthly
@dataclass
class Bundle:
    month: pd.Timestamp
    train_end: pd.Timestamp
    mu: np.ndarray
    sd: np.ndarray
    lgbm: object = None
    lgbm_cal: object = None
    seq: object = None
    seq_cal: object = None
    logreg: object = None
    range_model: object = None
    seq_mu: np.ndarray | None = None          # the sequence net keeps the scaling of the month it was trained in
    seq_sd: np.ndarray | None = None
    seq_month: pd.Timestamp | None = None
    n_train: int = 0
    fit_seconds: float = 0.0


def _windows(dates: pd.DatetimeIndex, month: pd.Timestamp, train_years: int, val_months: int):
    month = pd.Timestamp(month).to_period("M").to_timestamp()
    v0 = month - pd.DateOffset(months=val_months)
    t0 = month - pd.DateOffset(years=train_years)
    tr = np.flatnonzero((dates >= t0) & (dates < v0))[:-1]            # purge the day whose label lies in validation
    va = np.flatnonzero((dates >= v0) & (dates < month))[:-1]         # purge the day whose label lies in the test month
    te = np.flatnonzero((dates >= month) & (dates < month + pd.DateOffset(months=1)))
    return tr, va, te


def fit_month(dd: nd.DayData, month, models=("lgbm", "seq", "logreg"), train_years: int = 4, val_months: int = 3,
              train_subsample: float = 0.5, seed: int = 0, range_xy: tuple | None = None,
              prev: "Bundle | None" = None, seq_subsample: float = 0.15) -> Bundle:
    """Fit the base models for ``month`` using only data that ended before that month.

    The sequence net is slow on a CPU (~30 min on half the rows), so it trains on ``seq_subsample`` of the rows and,
    when ``prev`` is given, is reused from that earlier bundle instead of refit (the caller refits it quarterly)."""
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    t0 = time.time()
    month = pd.Timestamp(month).to_period("M").to_timestamp()
    tr_t, va_t, _ = _windows(dd.dates, month, train_years, val_months)
    rng = np.random.default_rng(seed)
    t_tr, n_tr = nd._rows(dd, tr_t)
    if train_subsample < 1:
        keep = rng.random(len(t_tr)) < train_subsample
        t_tr, n_tr = t_tr[keep], n_tr[keep]
    t_va, n_va = nd._rows(dd, va_t)
    Xtr, mu, sd = nd._tab(dd, t_tr, n_tr)
    Xva, _, _ = nd._tab(dd, t_va, n_va, mu, sd)
    ytr, yva = dd.y[t_tr, n_tr], dd.y[t_va, n_va]
    b = Bundle(month, dd.dates[va_t[-1]] if len(va_t) else month, mu, sd, n_train=len(ytr))
    if "lgbm" in models:
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                               random_state=seed, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
        b.lgbm, b.lgbm_cal = m, IsotonicRegression(out_of_bounds="clip").fit(m.predict_proba(Xva)[:, 1], yva)
    if "seq" in models and prev is not None and prev.seq is not None:
        b.seq, b.seq_cal, b.seq_mu, b.seq_sd, b.seq_month = prev.seq, prev.seq_cal, prev.seq_mu, prev.seq_sd, prev.seq_month
    elif "seq" in models:
        ks = rng.random(len(t_tr)) < seq_subsample / max(train_subsample, 1e-9)
        st, sn = t_tr[ks], n_tr[ks]
        net = nd.SeqNet(len(nd.SEQ_FEATS), Xtr.shape[1], seed=seed)
        net.fit(nd._seq(dd, st, sn), Xtr[ks], ytr[ks].astype(np.float32), nd._seq(dd, t_va, n_va), Xva,
                yva.astype(np.float32))
        b.seq = net
        b.seq_cal = IsotonicRegression(out_of_bounds="clip").fit(net.predict(nd._seq(dd, t_va, n_va), Xva), yva)
        b.seq_mu, b.seq_sd, b.seq_month = mu, sd, month
    if "logreg" in models:
        b.logreg = LogisticRegression(C=0.1, max_iter=300).fit(Xtr, ytr)
    if range_xy is not None:
        from .range_model import FEATS as RF
        X, y = range_xy
        d = X.index.get_level_values(0)
        tr = (d >= month - pd.DateOffset(years=train_years)) & (d < month) & y.notna().to_numpy()
        if tr.sum() > 10_000:
            rm = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                                   subsample=0.7, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
            rm.fit(X[RF][tr], y[tr])
            b.range_model = rm
    b.fit_seconds = time.time() - t0
    return b


def predict_bases(b: Bundle, dd: nd.DayData, t_idx: np.ndarray, range_X: pd.DataFrame | None = None) -> pd.DataFrame:
    """Base-model predictions for the days ``t_idx`` (labels may be unknown yet)."""
    t, n = nd._rows(dd, t_idx, need_label=False)
    if not len(t):
        return pd.DataFrame()
    X, _, _ = nd._tab(dd, t, n, b.mu, b.sd)
    out = pd.DataFrame({"date": dd.dates[t], "ticker": np.array(dd.tickers)[n], "y": dd.y[t, n],
                        "ret_next": dd.ret_next[t, n], "refit": b.month})
    if b.lgbm is not None:
        out["lgbm"] = b.lgbm_cal.predict(b.lgbm.predict_proba(X)[:, 1])
    if b.seq is not None:
        Xs = X if b.seq_mu is None or b.seq_mu is b.mu else nd._tab(dd, t, n, b.seq_mu, b.seq_sd)[0]
        out["seq"] = b.seq_cal.predict(b.seq.predict(nd._seq(dd, t, n), Xs))
    if b.logreg is not None:
        out["logreg"] = b.logreg.predict_proba(X)[:, 1]
    out = out.set_index(["date", "ticker"])
    if b.range_model is not None and range_X is not None:
        from .range_model import FEATS as RF
        rx = range_X.reindex(out.index)
        ok = rx[RF].notna().all(axis=1).to_numpy()
        pr = np.full(len(out), np.nan)
        if ok.any():
            pr[ok] = b.range_model.predict(rx.loc[ok, RF])
        out["range"] = pr
    parts = [c for c in ("lgbm", "seq") if c in out]
    if parts:
        out["ensemble"] = out[parts].mean(axis=1)
    return out


def rolling(dd: nd.DayData, start: str, end: str | None = None, models=("lgbm", "seq", "logreg"),
            train_years: int = 4, val_months: int = 3, range_xy: tuple | None = None, verbose: bool = True,
            save_dir=None, seq_every: int = 3) -> pd.DataFrame:
    """Out-of-sample base predictions for every day from ``start``, each month predicted by models refit that month
    (the sequence net every ``seq_every`` months, in January, April, July and October for 3)."""
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end or dd.dates[-1]), freq="M").to_timestamp()
    out = []
    prev = None
    for m in months:
        _, _, te = _windows(dd.dates, m, train_years, val_months)
        if not len(te):
            continue
        reuse = prev if prev is not None and (m.month - 1) % seq_every != 0 else None
        b = fit_month(dd, m, models, train_years, val_months, range_xy=range_xy, prev=reuse)
        prev = b
        out.append(predict_bases(b, dd, te, range_xy[0] if range_xy else None))
        if save_dir is not None:
            save_bundle(b, save_dir)
        if verbose:
            print(f"  [rolling {m:%Y-%m}] train {b.n_train:,} rows to {b.train_end:%Y-%m-%d} · {b.fit_seconds:.0f}s",
                  flush=True)
    return pd.concat(out).sort_index()


def save_bundle(b: Bundle, d=None) -> None:
    d = d or MODEL_DIR
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{b.month:%Y-%m}.pkl", "wb") as f:
        pickle.dump(b, f)


def load_bundle(month, d=None) -> Bundle | None:
    p = (d or MODEL_DIR) / f"{pd.Timestamp(month):%Y-%m}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- stacking
def meta_features(base: pd.DataFrame) -> pd.DataFrame:
    """Per-day cross-sectional ranks (0-1) of each base prediction, so the meta model sees stationary inputs."""
    cols = [c for c in BASES if c in base]
    r = base[cols].groupby(level=0).rank(pct=True)
    r.columns = [f"r_{c}" for c in cols]
    if "r_range" in r and "r_lgbm" in r:
        r["dir_x_range"] = (r["r_lgbm"] - 0.5) * r["r_range"]
    return r


def fit_meta(base: pd.DataFrame, upto: pd.Timestamp, meta_months: int = 12):
    """Logistic meta model on base OOS predictions from the ``meta_months`` months before ``upto``."""
    from sklearn.linear_model import LogisticRegression
    d = base.index.get_level_values(0)
    sel = base[(d < upto) & (d >= upto - pd.DateOffset(months=meta_months))].dropna(subset=["y"])
    F = meta_features(sel).dropna()
    if len(F) < 5_000:
        return None
    y = sel.loc[F.index, "y"].to_numpy()
    m = LogisticRegression(C=1.0, max_iter=500).fit(F.to_numpy(), y)
    m.feature_names_ = list(F.columns)
    return m


def stack(base: pd.DataFrame, start: str, meta_months: int = 12, verbose: bool = True) -> pd.DataFrame:
    """Stacked out-of-sample P(up) for every day from ``start``: the meta model for month M is trained on base
    predictions from the 12 months before M (which were themselves out of sample)."""
    months = pd.period_range(pd.Timestamp(start), base.index.get_level_values(0).max(), freq="M").to_timestamp()
    F_all = meta_features(base)
    d = base.index.get_level_values(0)
    out = []
    for m in months:
        meta = fit_meta(base, m, meta_months)
        sel = (d >= m) & (d < m + pd.DateOffset(months=1))
        if meta is None or not sel.any():
            continue
        F = F_all[sel].reindex(columns=meta.feature_names_)
        ok = F.notna().all(axis=1).to_numpy()
        p = np.full(int(sel.sum()), np.nan)
        p[ok] = meta.predict_proba(F[ok].to_numpy())[:, 1]
        chunk = base[sel].assign(stacked=p)
        out.append(chunk)
        if verbose:
            coef = dict(zip(meta.feature_names_, np.round(meta.coef_[0], 2)))
            print(f"  [stack {m:%Y-%m}] weights {coef}", flush=True)
    res = pd.concat(out).sort_index()
    return add_patterns(res)


def add_patterns(df: pd.DataFrame, col: str = "stacked") -> pd.DataFrame:
    df = df.copy()
    direction = np.where(df[col] >= UP, "Up", np.where(df[col] <= DOWN, "Down", "Flat"))
    if "range" in df:
        rr = df["range"].groupby(level=0).rank(pct=True)
        size = np.where(rr >= 0.7, "volatile", np.where(rr <= 0.3, "quiet", "normal"))
    else:
        size = np.full(len(df), "normal")
    df["pattern"] = [f"{a} · {b}" for a, b in zip(direction, size)]
    df.loc[df[col].isna(), "pattern"] = np.nan
    return df


def pattern_changes(today: pd.DataFrame, yesterday: pd.DataFrame) -> pd.DataFrame:
    """Stocks whose stacked pattern changed between two closes, strongest moves in P(up) first."""
    j = today[["pattern", "stacked"]].join(yesterday[["pattern", "stacked"]], rsuffix="_prev", how="inner")
    ch = j[j["pattern"] != j["pattern_prev"]].copy()
    ch["Δ P(up)"] = ch["stacked"] - ch["stacked_prev"]
    return ch.reindex(ch["Δ P(up)"].abs().sort_values(ascending=False).index)


# --------------------------------------------------------------------------- measurement
def paired_ic(pred: pd.DataFrame, a: str, b: str) -> dict:
    """Daily rank-IC of model a minus model b on the same stock-days, with a t-statistic."""
    d = pred.dropna(subset=[a, b, "ret_next"])
    g = d.groupby(level=0)
    ica = g.apply(lambda x: x[a].rank().corr(x["ret_next"].rank()) if len(x) > 20 else np.nan)
    icb = g.apply(lambda x: x[b].rank().corr(x["ret_next"].rank()) if len(x) > 20 else np.nan)
    diff = (ica - icb).dropna()
    return {"IC a": float(ica.mean()), "IC b": float(icb.mean()), "diff": float(diff.mean()),
            "t": float(diff.mean() / diff.std() * np.sqrt(len(diff))) if diff.std() > 0 else np.nan,
            "days": int(len(diff)), "a better on % of days": float((diff > 0).mean())}


def notable(changes: pd.DataFrame) -> pd.Series:
    """A change worth attention: direction flipped Up↔Down, or size jumped quiet↔volatile (not just a nudge across
    the Flat or normal band)."""
    if changes.empty:
        return pd.Series(dtype=bool)
    d0, s0 = changes["pattern_prev"].str.split(" · ").str[0], changes["pattern_prev"].str.split(" · ").str[1]
    d1, s1 = changes["pattern"].str.split(" · ").str[0], changes["pattern"].str.split(" · ").str[1]
    flip = ((d0 == "Up") & (d1 == "Down")) | ((d0 == "Down") & (d1 == "Up"))
    jump = ((s0 == "quiet") & (s1 == "volatile")) | ((s0 == "volatile") & (s1 == "quiet"))
    return flip | jump


# --------------------------------------------------------------------------- the daily step (run at market close)
def _bundle_for(dd, m, models, range_xy, prev, seq_every: int = 3) -> Bundle:
    b = load_bundle(m)
    if b is None:
        reuse = prev if prev is not None and (m.month - 1) % seq_every != 0 else None
        b = fit_month(dd, m, models, range_xy=range_xy, prev=reuse)
        save_bundle(b)
    return b


def daily_update(p, dd: nd.DayData | None = None, models=("lgbm", "seq", "logreg"), meta_months: int = 12) -> dict:
    """Bring the rolling base predictions and the stack up to the panel's last day.

    Refits a month's base models only when that month's bundle is missing (so the pipeline trains once a month, the
    sequence net once a quarter), fills in labels of earlier days that are now known, fits the month's meta model once,
    and returns today's base + stacked predictions and the pattern changes since the previous close."""
    from . import range_model as RM
    dd = dd or nd.build(p)
    X, y, _ = RM.build(p)
    day = dd.dates[-1]
    base = pd.read_parquet(BASE_PATH) if BASE_PATH.exists() else pd.DataFrame()
    last = base.index.get_level_values(0).max() if len(base) else dd.dates[-1] - pd.DateOffset(months=1)
    todo = np.flatnonzero(dd.dates > last)
    new = []
    prev = None
    for m in pd.period_range(dd.dates[todo[0]], day, freq="M").to_timestamp() if len(todo) else []:
        idx = todo[(dd.dates[todo] >= m) & (dd.dates[todo] < m + pd.DateOffset(months=1))]
        if prev is None:
            prev = load_bundle(m - pd.DateOffset(months=1))
        b = _bundle_for(dd, m, models, (X, y), prev)
        prev = b
        if len(idx):
            new.append(predict_bases(b, dd, idx, X))
    if new:
        base = pd.concat([base, *new]) if len(base) else pd.concat(new)
        base = base[~base.index.duplicated(keep="last")].sort_index()
    # labels of recent days become known a day later
    miss = base["y"].isna()
    if miss.any():
        pos = {d: i for i, d in enumerate(dd.dates)}
        col = {t: j for j, t in enumerate(dd.tickers)}
        ii = [pos.get(d, -1) for d in base.index[miss].get_level_values(0)]
        jj = [col.get(t, -1) for t in base.index[miss].get_level_values(1)]
        ok = np.array([(i >= 0 and j >= 0) for i, j in zip(ii, jj)])
        yv, rv = np.full(int(miss.sum()), np.nan), np.full(int(miss.sum()), np.nan)
        ia, ja = np.array(ii)[ok], np.array(jj)[ok]
        yv[ok], rv[ok] = dd.y[ia, ja], dd.ret_next[ia, ja]
        base.loc[miss, "y"], base.loc[miss, "ret_next"] = yv, rv
    BASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(BASE_PATH)
    # meta model for this month (cached), applied to the days that do not have a stacked value yet
    M = day.to_period("M").to_timestamp()
    mp = MODEL_DIR / f"meta_{M:%Y-%m}.pkl"
    if mp.exists():
        with open(mp, "rb") as f:
            meta = pickle.load(f)
    else:
        meta = fit_meta(base, M, meta_months)
        if meta is not None:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            with open(mp, "wb") as f:
                pickle.dump(meta, f)
    stk = pd.read_parquet(STACK_PATH) if STACK_PATH.exists() else pd.DataFrame()
    if meta is None:
        return {"day": day, "base": base.xs(day, level=0) if day in base.index.get_level_values(0) else pd.DataFrame(),
                "stacked": pd.DataFrame(), "changes": pd.DataFrame(), "note": "not enough base history for the stack"}
    d = base.index.get_level_values(0)
    sel = d >= M
    F = meta_features(base[sel]).reindex(columns=meta.feature_names_)
    ok = F.notna().all(axis=1).to_numpy()
    p_up = np.full(int(sel.sum()), np.nan)
    p_up[ok] = meta.predict_proba(F[ok].to_numpy())[:, 1]
    cur = add_patterns(base[sel].assign(stacked=p_up))
    if len(stk):
        stk = stk[stk.index.get_level_values(0) < M]
        # older months keep their stored values; refresh their labels from base
        stk[["y", "ret_next"]] = base[["y", "ret_next"]].reindex(stk.index).to_numpy()
    stk = pd.concat([stk, cur]).sort_index() if len(stk) else cur
    stk.to_parquet(STACK_PATH)
    days = sorted(stk.index.get_level_values(0).unique())
    today = stk.xs(day, level=0)
    changes = pattern_changes(today, stk.xs(days[-2], level=0)) if len(days) > 1 else pd.DataFrame()
    if len(changes):
        changes["notable"] = notable(changes)
    return {"day": day, "base": base.xs(day, level=0), "stacked": today, "changes": changes,
            "weights": dict(zip(meta.feature_names_, np.round(meta.coef_[0], 3)))}


def model_health(stk: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Rolling ``window``-day mean rank IC of each base model and the stack (drift monitor)."""
    cols = [c for c in ("lgbm", "seq", "logreg", "stacked") if c in stk]
    d = stk.dropna(subset=["ret_next"])
    g = d.groupby(level=0)
    ic = pd.DataFrame({c: g.apply(lambda x, c=c: x[c].rank().corr(x["ret_next"].rank()) if x[c].notna().sum() > 20
                                  else np.nan) for c in cols})
    return ic.rolling(window, min_periods=max(10, window // 3)).mean()
