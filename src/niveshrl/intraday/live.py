"""Live paper trading during NSE hours (09:15–15:30 IST, weekdays). Paper only: no orders are ever placed.

Every 5 minutes (a few seconds after each 5-minute bar closes) it downloads today's 5-minute bars for the liquid
universe and NIFTY from Yahoo, keeps only *finished* bars, and re-runs the day so far with every decision frozen the
first time it was made (so a decision never changes after the fact). Fills are the next bar's open, exactly like the
replay. It writes data/intraday/live.json (status, open positions, today's trades and skips) for the app. After 15:20
it settles the day: costs, tax, learning (bandit + scorer) and the day record. A day with no bars by 09:35 is treated
as a market holiday.

Yahoo's intraday data can lag a few minutes; signals are only ever taken on bars that had finished.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import DIR, config
from .agent import Agent
from .universe import load as load_universe

IST = timezone(timedelta(hours=5, minutes=30))


def now_ist() -> datetime:
    return datetime.now(IST)


def _write_status(obj: dict) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    p = DIR / "live.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)


def today_bars(tickers: list[str], batch: int = 100) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out = {}
    allt = list(tickers) + ["^NSEI"]
    for i in range(0, len(allt), batch):
        chunk = allt[i:i + batch]
        raw = yf.download(chunk, period="1d", interval="5m", auto_adjust=False, progress=False, threads=True,
                          group_by="ticker")
        for t in chunk:
            try:
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            d = d.dropna(subset=["Close"])
            if d.empty:
                continue
            idx = d.index.tz_convert("Asia/Kolkata").tz_localize(None)
            out[t] = pd.DataFrame({"open": d["Open"].to_numpy(), "high": d["High"].to_numpy(), "low": d["Low"].to_numpy(),
                                   "close": d["Close"].to_numpy(), "volume": d["Volume"].to_numpy()}, index=idx)
    return out


def finished(bars: dict[str, pd.DataFrame], at: datetime) -> dict[str, pd.DataFrame]:
    """Drop the still-forming bar: a 5-minute bar starting at T is finished only after T + 5 minutes."""
    cutoff = pd.Timestamp(at.replace(tzinfo=None)) - pd.Timedelta(minutes=5)
    return {t: b[b.index <= cutoff] for t, b in bars.items()}


def warm_start() -> bool:
    """First live day: start from what the replay learned (bandit buckets, ML scorer, outcome history), never from its
    pool or trades. Logged in the learning log so it is visible."""
    import shutil
    rep = DIR / "replay"
    if (DIR / "bandit.json").exists() or not (rep / "bandit.json").exists():
        return False
    DIR.mkdir(parents=True, exist_ok=True)
    for name in ("bandit.json", "scorer.pkl", "shadow.parquet"):
        if (rep / name).exists():
            shutil.copy2(rep / name, DIR / name)
    b = json.loads((DIR / "bandit.json").read_text(encoding="utf-8"))
    b.setdefault("log", []).append({"day": now_ist().date().isoformat(), "bucket": "(all)", "from": "empty", "to": "warm start",
                                    "n": sum(v[0] for k, v in b.get("stats", {}).items() if k.count("|") == 0),
                                    "why": "copied what the 52-day replay learned; the live pool and trades start fresh",
                                    "when": now_ist().isoformat(timespec="seconds")})
    (DIR / "bandit.json").write_text(json.dumps(b), encoding="utf-8")
    return True


def context_from_universe(u: pd.DataFrame) -> dict[str, dict]:
    cols = ["prev_high", "prev_low", "prev_close", "adr", "atr", "avg_volume", "nr7"]
    return {t: {c: float(r[c]) for c in cols} for t, r in u.iterrows()}


def run_live(stop_flag=None, sleep=time.sleep) -> dict:
    """Run one trading day. ``stop_flag()`` -> True stops early (the app's Stop button)."""
    cfg = config()
    warm_start()
    agent = Agent(DIR, cfg)
    t0 = now_ist()
    day = t0.date().isoformat()
    if t0.weekday() >= 5:
        _write_status({"phase": "weekend", "day": day, "at": t0.isoformat(timespec="seconds"),
                       "message": "NSE is closed on weekends."})
        return {"phase": "weekend"}
    _write_status({"phase": "starting", "day": day, "at": t0.isoformat(timespec="seconds"),
                   "message": "Loading the liquid NSE universe…"})
    u = load_universe(max_age_days=1)        # rebuilt each morning: yesterday's high/low/close, ADR, NR7, volumes
    ctx = context_from_universe(u)
    tickers = u.index.tolist()
    decisions: dict = {}
    dec_path = DIR / f"decisions_{day}.json"
    if dec_path.exists():
        decisions = json.loads(dec_path.read_text(encoding="utf-8"))
    last_bar = None
    while True:
        if stop_flag and stop_flag():
            _write_status({"phase": "stopped", "day": day, "at": now_ist().isoformat(timespec="seconds"),
                           "message": "Stopped from the app; open paper positions stay as they were."})
            return {"phase": "stopped"}
        n = now_ist()
        hm = n.hour * 60 + n.minute
        if hm < 9 * 60 + 20:
            _write_status({"phase": "waiting", "day": day, "at": n.isoformat(timespec="seconds"),
                           "message": "Waiting for the market (first decisions after 09:20)."})
            sleep(30)
            continue
        if hm >= 15 * 60 + 21:
            break
        if last_bar is not None and (hm // 5) == last_bar:
            sleep(15)
            continue
        if n.second < 20 and hm % 5 == 0:                   # let the bar close and Yahoo catch up
            sleep(20 - n.second)
            continue
        last_bar = hm // 5
        bars = finished(today_bars(tickers), now_ist())
        nifty = bars.pop("^NSEI", None)
        bars = {t: b for t, b in bars.items() if len(b) >= 3 and pd.Timestamp(b.index[0]).date() == n.date()}
        if not bars and hm >= 9 * 60 + 35:
            _write_status({"phase": "holiday", "day": day, "at": n.isoformat(timespec="seconds"),
                           "message": "No bars for today by 09:35: treated as a market holiday."})
            return {"phase": "holiday"}
        clock = nifty.index if nifty is not None and len(nifty) else None
        if clock is not None:
            bars = {t: b.reindex(clock).ffill().assign(volume=lambda x: x["volume"].fillna(0)) for t, b in bars.items()
                    if len(b) and b.index[0] <= clock[min(2, len(clock) - 1)]}
        from .sim import Account
        sigs = agent.signals_for_day(bars, ctx, nifty)
        acct = Account(cfg, float(agent.state["pool"]))
        trades, log = acct.run_day(sigs, bars, agent.frozen_decide(decisions), close_at_end=False)
        dec_path.write_text(json.dumps(decisions), encoding="utf-8")
        day_net = sum(t.net for t in trades) + sum(p["unrealized"] for p in acct.open_positions)
        _write_status({"phase": "trading", "day": day, "at": now_ist().isoformat(timespec="seconds"),
                       "bars": len(clock) if clock is not None else 0, "followed": len({s.ticker for s in sigs}),
                       "pool": agent.state["pool"], "day_pnl": round(day_net, 2), "signals": len(sigs),
                       "positions": acct.open_positions, "trades": [t.to_dict() for t in trades],
                       "log": log[-200:], "leverage": cfg["leverage"],
                       "message": f"Following {len({s.ticker for s in sigs})} in-play stocks; "
                                  f"{len(acct.open_positions)} open, {len(trades)} closed today."})
        sleep(5)
    # after 15:20: settle and learn on the full day
    bars = finished(today_bars(tickers), now_ist())
    nifty = bars.pop("^NSEI", None)
    bars = {t: b for t, b in bars.items() if len(b) >= 3}
    if nifty is not None and len(nifty):
        bars = {t: b.reindex(nifty.index).ffill().assign(volume=lambda x: x["volume"].fillna(0)) for t, b in bars.items()
                if b.index[0] <= nifty.index[min(2, len(nifty) - 1)]}
    res, trades, log = agent.trade_day(day, bars, ctx, nifty, learn=True, decisions=decisions)
    dec_path.write_text(json.dumps(decisions), encoding="utf-8")
    _write_status({"phase": "closed", "day": day, "at": now_ist().isoformat(timespec="seconds"), "result": res.to_dict(),
                   "trades": [t.to_dict() for t in trades], "log": log[-200:], "pool": agent.state["pool"],
                   "message": f"Day settled: net ₹{res.net:+,.0f} after ₹{res.costs:,.0f} costs; learned "
                              f"{len(res.learned)} change(s)."})
    return {"phase": "closed", "result": res.to_dict()}
