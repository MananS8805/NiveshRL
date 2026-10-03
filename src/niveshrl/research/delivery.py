"""NSE delivery % (from the daily bhavcopy) and a volume phase: Accumulation / Neutral / Distribution.

Delivery % = the share of a day's traded quantity that was actually delivered (taken into demat), from NSE's
``sec_bhavdata_full_DDMMYYYY.csv``. High delivery on up days and low on down days is read as genuine buying.
Files are cached in data/bhav/; a failed download (network, holiday) is never cached as 'no data'.

Volume phase score (0–100, starts at 50), from daily bars:
- up-day vs down-day volume over 50 days: ±25 (25 × log2 of the ratio, capped)
- Chaikin Money Flow over 20 days: ±15
- accumulation minus distribution days (up/down closes on above-average volume) over 25 days: ±10
- OBV change over 20 days relative to the volume traded: ±10
- delivery % on up days minus down days over the last 25 sessions: ±10 (left out when no bhavcopy data)
≥ 60 Accumulation, ≤ 40 Distribution, otherwise Neutral. Estimated from price and volume: it cannot see real orders.
"""
from __future__ import annotations

import io
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from ..config import ROOT
from .data import Panel

CACHE = ROOT / "data" / "bhav"
URL = "https://archives.nseindia.com/products/content/sec_bhavdata_full_{d:%d%m%Y}.csv"


def fetch(day: pd.Timestamp) -> pd.DataFrame | None:
    """One day's bhavcopy (EQ series): symbol -> delivery %, quantity, close. None if NSE has no file (holiday)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{day:%Y%m%d}.csv"
    if path.exists():
        raw = path.read_bytes()
    else:
        try:
            req = urllib.request.Request(URL.format(d=day), headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None                      # no file that day (holiday): fine, nothing cached
            raise
        path.write_bytes(raw)                    # only successful downloads are cached
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip() for c in df.columns]
    df = df[df["SERIES"].str.strip() == "EQ"]
    out = pd.DataFrame({"deliv_pct": pd.to_numeric(df["DELIV_PER"], errors="coerce") / 100,
                        "qty": pd.to_numeric(df["TTL_TRD_QNTY"], errors="coerce"),
                        "close": pd.to_numeric(df["CLOSE_PRICE"], errors="coerce"),
                        "prev_close": pd.to_numeric(df["PREV_CLOSE"], errors="coerce")})
    out.index = df["SYMBOL"].str.strip() + ".NS"
    return out


def recent(days: list[pd.Timestamp]) -> pd.DataFrame:
    """Delivery % per (date x ticker) for the given trading days (missing days skipped)."""
    cols = {}
    for d in days:
        f = fetch(pd.Timestamp(d))
        if f is not None:
            cols[pd.Timestamp(d)] = f["deliv_pct"]
    return pd.DataFrame(cols).T.sort_index() if cols else pd.DataFrame()


def volume_phase(p: Panel, deliv: pd.DataFrame | None = None, day: pd.Timestamp | None = None) -> pd.DataFrame:
    c, h, l, v = p.close, p.high, p.low, p.volume
    d = day or c.index[-1]
    pos = c.index.get_loc(d)
    sl = slice(max(0, pos - 60), pos + 1)
    cc, hh, ll, vv = c.iloc[sl], h.iloc[sl], l.iloc[sl], v.iloc[sl]
    chg = cc.diff()
    up_v = vv.where(chg > 0).iloc[-50:].sum()
    dn_v = vv.where(chg < 0).iloc[-50:].sum()
    ratio = (up_v / dn_v.replace(0, np.nan)).clip(0.1, 10)
    s_ud = (25 * np.log2(ratio)).clip(-25, 25)
    mfm = ((cc - ll) - (hh - cc)) / (hh - ll).replace(0, np.nan)
    cmf = (mfm * vv).iloc[-20:].sum() / vv.iloc[-20:].sum().replace(0, np.nan)
    s_cmf = (15 * cmf / 0.2).clip(-15, 15)
    avgv = vv.rolling(20, min_periods=10).mean()
    big = vv > avgv
    acc = ((chg > 0) & big).iloc[-25:].sum()
    dist = ((chg < 0) & big).iloc[-25:].sum()
    s_ad = (2 * (acc - dist)).clip(-10, 10)
    obv = (np.sign(chg).fillna(0) * vv).cumsum()
    obv_chg = (obv.iloc[-1] - obv.iloc[-21]) / vv.iloc[-20:].sum().replace(0, np.nan)
    s_obv = (10 * obv_chg / 0.3).clip(-10, 10)
    score = 50 + s_ud.fillna(0) + s_cmf.fillna(0) + s_ad.fillna(0) + s_obv.fillna(0)
    out = pd.DataFrame({"up_down_vol": ratio, "cmf20": cmf, "acc_minus_dist": acc - dist, "obv_chg": obv_chg})
    if deliv is not None and len(deliv):
        dl = deliv.reindex(columns=c.columns).iloc[-25:]
        ch = c.reindex(dl.index).diff()
        upd = dl.where(ch > 0).mean()
        dnd = dl.where(ch < 0).mean()
        diff = upd - dnd
        s_del = (10 * diff / 0.10).clip(-10, 10)
        score = score + s_del.fillna(0)
        out["deliv_pct"] = dl.iloc[-1]
        out["deliv_avg20"] = dl.iloc[-20:].mean()
        out["deliv_up_minus_down"] = diff
    out["vol_phase_score"] = score.clip(0, 100)
    out["vol_phase"] = np.where(out["vol_phase_score"] >= 60, "Accumulation",
                                np.where(out["vol_phase_score"] <= 40, "Distribution", "Neutral"))
    return out


def measure_phase_edge(p: Panel, step: int = 20, horizon: int = 20, start: str = "2014-02-01") -> pd.DataFrame:
    """Forward 20-day excess return (vs the median member) by volume phase, without delivery (no long bhavcopy history),
    on non-overlapping dates."""
    c = p.close
    fwd = c.shift(-horizon) / c - 1
    days = [d for d in c.index[::step] if d >= pd.Timestamp(start) and c.index.get_loc(d) > 70]
    rows = []
    for d in days:
        ph = volume_phase(p, None, d)
        f = fwd.loc[d]
        el = f.notna() & c.loc[d].notna()
        if p.member is not None:
            el &= p.member.loc[d].reindex(f.index).fillna(False)
        ex = f[el] - f[el].mean()                    # vs the average member, not the median (skewed returns)
        for k in ("Accumulation", "Neutral", "Distribution"):
            idx = ph.index[(ph["vol_phase"] == k)].intersection(ex.index)
            if len(idx):
                rows.append({"date": d, "phase": k, "excess": float(ex[idx].mean()), "n": len(idx)})
    df = pd.DataFrame(rows)
    g = df.groupby("phase")
    return pd.DataFrame({"days": g.size(), "stocks per day": g["n"].mean(), "avg 20d excess": g["excess"].mean(),
                         "t-stat": g["excess"].mean() / g["excess"].std() * np.sqrt(g.size())})
