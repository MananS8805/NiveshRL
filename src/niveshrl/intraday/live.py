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


def _v2_status(obj: dict) -> None:
    d = DIR / "v2"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "live.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)


_DHAN: dict = {}


def _dhan():
    """The Dhan data client when configured and its Data API plan is active (checked once a day), else None."""
    day = now_ist().date()
    if _DHAN.get("day") != day:
        _DHAN.clear()
        _DHAN["day"] = day
        try:
            from ..dhan import Dhan
            d = Dhan.available()
            st = d.status() if d else {"ok": False, "why": "not configured"}
            _DHAN["client"] = d if st["ok"] else None
            _DHAN["why"] = st.get("why", "")
        except Exception as e:                                   # noqa: BLE001
            _DHAN["client"], _DHAN["why"] = None, str(e)[:160]
    return _DHAN.get("client")


def data_source() -> str:
    return "Dhan" if _DHAN.get("client") else f"Yahoo ({_DHAN.get('why') or 'Dhan not configured'})"


def today_bars(tickers: list[str], batch: int = 100) -> dict[str, pd.DataFrame]:
    """Today's 5-minute bars: from Dhan (exchange data) when configured, else Yahoo; NIFTY always from Yahoo, and any
    stock Dhan does not return falls back to Yahoo."""
    d = _dhan()
    if d is not None:
        try:
            out = d.today_bars([t for t in tickers if t != "^NSEI"])
        except Exception:                                        # noqa: BLE001
            out = {}
        rest = [t for t in tickers if t not in out]
        return out | _yahoo_bars(rest, batch)
    return _yahoo_bars(tickers, batch)


def _yahoo_bars(tickers: list[str], batch: int = 100) -> dict[str, pd.DataFrame]:
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


def _cp_status(obj: dict) -> None:
    from .copilot import save_json
    save_json("live.json", obj)


class _CopilotDay:
    """The copilot runs beside v1/v2 in the same loop; any copilot error is reported and never stops the others."""

    def __init__(self, u, day: str):
        self.ok, self.day = False, day
        try:
            from .copilot import CDIR, config as cp_config
            from .copilot.engine import Copilot, liquid
            self.cp = Copilot(cp_config())
            self.tickers = set(liquid(u, self.cp.cfg))
            self.path = CDIR / f"decisions_{day}.json"
            self.frozen = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
            self.ok = True
        except Exception as e:                                  # noqa: BLE001
            _cp_status({"phase": "error", "day": day, "message": f"copilot unavailable: {e}"})

    def brief(self) -> None:
        if not self.ok:
            return
        try:
            from .copilot import load_json
            from .copilot.brief import morning_brief
            if not load_json(f"briefs/{self.day}.json"):
                morning_brief(self.day, sorted(self.tickers), self.cp.cfg)
        except Exception as e:                                  # noqa: BLE001
            _cp_status({"phase": "error", "day": self.day, "message": f"morning brief failed: {e}"})

    def step(self, bars, ctx, nifty) -> None:
        if not self.ok:
            return
        try:
            from .copilot.brief import opening_update
            b = {t: x for t, x in bars.items() if t in self.tickers}
            if nifty is not None and len(nifty) >= 3:
                opening_update(self.day, b, ctx)
            cards, trades, log, acct = self.cp.run(b, ctx, nifty, self.frozen, self.day, log=True)
            self.path.write_text(json.dumps(self.frozen), encoding="utf-8")
            top = self.cp.top_cards(cards)
            _cp_status({"phase": "trading", "day": self.day, "at": now_ist().isoformat(timespec="seconds"),
                        "pool": self.cp.state["pool"], "cards": cards.drop(columns=["key"], errors="ignore").to_dict("records") if len(cards) else [],
                        "top": top.drop(columns=["key"], errors="ignore").to_dict("records") if len(top) else [],
                        "positions": acct.open_positions, "trades": [t.to_dict() for t in trades], "log": log[-200:],
                        "day_pnl": round(sum(t.net for t in trades) + sum(p["unrealized"] for p in acct.open_positions), 2),
                        "rules": {k: v for k, v in self.cp.rules.items() if k != "log"},
                        "model": {"skilled": self.cp.model.skilled, **self.cp.model.skill},
                        "message": f"{len(cards)} setups so far, {len(trades)} closed, {len(acct.open_positions)} open."})
        except Exception as e:                                  # noqa: BLE001
            _cp_status({"phase": "error", "day": self.day, "message": f"copilot error: {e}"})

    def settle(self, bars, ctx, nifty) -> None:
        if not self.ok:
            return
        try:
            b = {t: x for t, x in bars.items() if t in self.tickers}
            rec = self.cp.settle(self.day, b, ctx, nifty, self.frozen)
            self.path.write_text(json.dumps(self.frozen), encoding="utf-8")
            _cp_status({"phase": "closed", "day": self.day, "at": now_ist().isoformat(timespec="seconds"),
                        "pool": self.cp.state["pool"], "result": {k: v for k, v in rec.items() if k != "review"},
                        "review": rec["review"], "message": f"Day reviewed: net ₹{rec['net']:+,.0f}; "
                                                           f"{rec['changes']} rule change(s)."})
        except Exception as e:                                  # noqa: BLE001
            _cp_status({"phase": "error", "day": self.day, "message": f"copilot settle error: {e}"})


class _AlgoDay:
    """The 50-trade paper book (Algo trader) in the same loop; its errors are reported and never stop the others."""

    def __init__(self, u, day: str):
        self.ok, self.day, self.book = False, day, None
        try:
            from .copilot import config as cp_config
            cfg = cp_config()
            if not (cfg.get("algo") or {}).get("enabled"):
                return
            from .algo.book import AlgoBook
            self.book = AlgoBook(cfg, u, day)
            self.ok = True
        except Exception as e:                                  # noqa: BLE001
            self._err(f"algo book unavailable: {e}")

    def _err(self, msg: str) -> None:
        try:
            from .algo.book import _save
            _save("live.json", {"phase": "error", "day": self.day, "at": now_ist().isoformat(timespec="seconds"), "message": msg})
        except Exception:                                       # noqa: BLE001
            pass

    def step(self, bars, ctx) -> None:
        if self.ok:
            try:
                self.book.step(bars, ctx, now_ist().isoformat(timespec="seconds"))
            except Exception as e:                              # noqa: BLE001
                self._err(f"algo book error: {e}")

    def settle(self, bars, ctx) -> None:
        if self.ok:
            try:
                self.book.settle(bars, ctx, now_ist().isoformat(timespec="seconds"))
            except Exception as e:                              # noqa: BLE001
                self._err(f"algo book settle error: {e}")


def late_note(t0: datetime) -> str | None:
    """A run that starts after the first decisions (09:20) replays the earlier bars from finished data in one batch:
    still bar by bar and without look-ahead, but nobody could have acted on those decisions live. Say so."""
    hm = t0.hour * 60 + t0.minute
    if hm <= 9 * 60 + 20:
        return None
    when = t0.strftime("%H:%M")
    if hm >= 15 * 60 + 21:
        return (f"Started at {when}, after the close (was the laptop asleep?): the whole day was replayed from finished "
                "bars. No decision was live and none was logged before its outcome.")
    return (f"Started late, at {when}: decisions before then were replayed from finished bars and were logged after "
            "their outcomes could be seen; only later decisions were live.")


def _mark_late(path, note: str) -> None:
    """Prepend the late-start note to a saved review and correct its forward-log claim."""
    try:
        r = json.loads(path.read_text(encoding="utf-8"))
        r["summary"] = [note] + [x for x in r.get("summary", []) if x != note]
        r["learnings"] = [x.replace("Every decision was recorded before its outcome.",
                                    "Today's early decisions were logged after the fact (late start).")
                          for x in r.get("learnings", [])]
        r["late_start"] = note
        path.write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    except Exception:                                            # noqa: BLE001
        pass


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
    cp = _CopilotDay(u, day)
    cp.brief()
    algo = _AlgoDay(u, day)
    decisions: dict = {}
    dec_path = DIR / f"decisions_{day}.json"
    v2, v2_frozen, v2_path = None, {}, DIR / "v2" / f"decisions_{day}.json"
    if cfg.get("engine", "shadow") in ("shadow", "v2"):          # the deep-learning engine runs beside v1, never instead
        try:
            from .engine_v2 import EngineV2
            v2 = EngineV2(cfg)
            if v2_path.exists():
                v2_frozen = json.loads(v2_path.read_text(encoding="utf-8"))
        except Exception as e:                                   # noqa: BLE001 - v2 must never stop v1
            v2 = None
            _v2_status({"phase": "error", "day": day, "message": f"v2 unavailable: {e}"})
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
                       "log": log[-200:], "leverage": cfg["leverage"], "source": data_source(),
                       "message": f"Following {len({s.ticker for s in sigs})} in-play stocks; "
                                  f"{len(acct.open_positions)} open, {len(trades)} closed today."})
        if v2 is not None:
            try:
                cand, t2, lg2, a2 = v2.run(bars, ctx, nifty, v2_frozen, close_at_end=False)
                v2_path.write_text(json.dumps(v2_frozen), encoding="utf-8")
                _v2_status({"phase": "trading", "day": day, "at": now_ist().isoformat(timespec="seconds"),
                            "pool": v2.state["pool"], "candidates": int(len(cand)), "positions": a2.open_positions,
                            "trades": [t.to_dict() for t in t2], "log": lg2[-200:],
                            "day_pnl": round(sum(t.net for t in t2) + sum(p["unrealized"] for p in a2.open_positions), 2),
                            "model": v2.meta, "policy": v2.policy})
            except Exception as e:                               # noqa: BLE001
                _v2_status({"phase": "error", "day": day, "message": f"v2 error: {e}"})
        cp.step(bars, ctx, nifty)
        algo.step(bars, ctx)
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
    try:                                                         # grow the 5-minute history beyond Yahoo's 60 days
        from .dl.dataset import archive_day
        long = pd.concat([b.assign(ticker=t).rename_axis("ts").reset_index() for t, b in bars.items()]
                         + ([nifty.assign(ticker="^NSEI").rename_axis("ts").reset_index()] if nifty is not None else []))
        archive_day(long[["ticker", "ts", "open", "high", "low", "close", "volume"]])
    except Exception:                                            # noqa: BLE001
        pass
    if v2 is not None:
        try:
            rec = v2.settle(day, bars, ctx, nifty, v2_frozen)
            from .engine_v2 import refit_if_due
            note = refit_if_due(cfg)
            _v2_status({"phase": "closed", "day": day, "at": now_ist().isoformat(timespec="seconds"), "result": rec,
                        "pool": v2.state["pool"], "message": f"v2 settled: net ₹{rec['net']:+,.0f}; {note}"})
        except Exception as e:                                   # noqa: BLE001
            _v2_status({"phase": "error", "day": day, "message": f"v2 settle error: {e}"})
    cp.settle(bars, ctx, nifty)
    algo.settle(bars, ctx)
    note = late_note(t0)
    if note:
        from .copilot import CDIR
        from .algo.book import ADIR
        for f in (CDIR / "reviews" / f"{day}.json", ADIR / "reviews" / f"{day}.json"):
            if f.exists():
                _mark_late(f, note)
    _write_status({"phase": "closed", "day": day, "at": now_ist().isoformat(timespec="seconds"), "result": res.to_dict(),
                   "trades": [t.to_dict() for t in trades], "log": log[-200:], "pool": agent.state["pool"],
                   "message": f"Day settled: net ₹{res.net:+,.0f} after ₹{res.costs:,.0f} costs; learned "
                              f"{len(res.learned)} change(s)."})
    return {"phase": "closed", "result": res.to_dict()}
