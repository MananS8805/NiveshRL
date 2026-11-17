"""The 50-trade paper book (Algo trader). Paper only: it never places an order.

Each morning it freezes the day's universe (top ``top_liquid`` by trailing turnover) and the rule it will trade: the
tournament leader (best net ₹/day over the last ``window_days`` days among rules averaging at least ``min_trades``
trades a day), or a pinned ``rule``. Every 5 minutes it recomputes that rule's signals on finished bars, fills a new
signal at the next bar's open (decisions are frozen once made), holds for the rule's period or to the 15:15
square-off, and charges real MIS costs on the actual quantity. Risk limits beat the quota: at most ``max_open``
positions, each worth pool × leverage ÷ max_open, and no new entry after the day loses ``daily_loss_stop`` of the pool.
After the close every rule in the grid is scored on the day in shadow (the tournament), and a review is written."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import DIR
from ..costs import round_trip
from . import research as R
from . import tournament as T

ADIR = DIR / "algo"
TOURN = ADIR / "tournament.parquet"


def _load(name: str, default):
    p = ADIR / name
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default
    except (OSError, ValueError):
        return default


def _save(name: str, obj) -> None:
    p = ADIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, default=str, indent=1), encoding="utf-8")
    tmp.replace(p)


def load_json(name: str, default=None):
    return _load(name, default)


def simulate(m: dict, sig: np.ndarray, hold: int, cfg: dict, pool: float, frozen: dict, close_at_end: bool):
    """Run the book over the bars so far. ``frozen['entries']`` keeps every decision once made (key 'bar|ticker');
    bars up to ``frozen['last_bar']`` are never re-decided. Returns (closed trades, open positions, log)."""
    a, c = cfg["algo"], cfg["costs"]
    o, cl = m["open"], m["close"]
    nb = o.shape[0]
    tick = m["tickers"]
    notional = pool * float(a["leverage"]) / int(a["max_open"])
    stop_rs = pool * float(a["daily_loss_stop"])
    entries: dict = frozen.setdefault("entries", {})
    last = int(frozen.get("last_bar", -1))
    closed, opened, log = [], [], []
    busy_until = {}
    realized = 0.0
    halted = False
    for i in range(R.OPEN_BAR - 1, min(R.LAST_ENTRY, nb - 2) + 1):
        e = i + 1
        # positions whose exit bar has passed are closed before new entries on this bar
        live_pos = [p for p in opened if p["exit_bar"] > e]
        for p in [p for p in opened if p["exit_bar"] <= e]:
            closed.append(p)
            realized += p["net"]
        opened = live_pos
        unreal = sum(p["side"] * (cl[e, p["j"]] - p["entry"]) * p["qty"] for p in opened)
        if not halted and realized + unreal <= -stop_rs:
            halted = True
            log.append({"bar": int(e), "event": "loss stop", "detail": f"day at ₹{realized + unreal:+,.0f}: no new entries"})
        cand = np.nonzero(sig[i])[0] if i > last else []
        n_open = len(opened)
        for j in cand:                                       # new decisions on bars never seen before
            k = f"{i}|{tick[j]}"
            if k in entries:
                continue
            if halted:
                entries[k] = {"side": int(sig[i, j]), "taken": False, "why": "daily loss stop"}
            elif busy_until.get(j, -1) >= e:
                entries[k] = {"side": int(sig[i, j]), "taken": False, "why": "already holding this stock"}
            elif n_open >= int(a["max_open"]):
                entries[k] = {"side": int(sig[i, j]), "taken": False, "why": f"{a['max_open']} positions open"}
            else:
                entries[k] = {"side": int(sig[i, j]), "taken": True, "why": "rule signal"}
                n_open += 1
        for k, d in entries.items():                         # replay the frozen decisions of this bar
            b, t = k.split("|", 1)
            if int(b) != i or not d["taken"] or t not in tick:
                continue
            j = tick.index(t)
            entry = float(o[e, j])
            qty = int(notional // entry) if entry > 0 else 0
            if qty <= 0:
                continue
            x = min(e + hold, R.SQUARE)
            pos = {"ticker": t, "j": j, "side": d["side"], "signal_bar": i, "entry_bar": e, "exit_bar": x,
                   "entry": entry, "qty": qty}
            if x < nb or close_at_end:
                xx = min(x, nb - 1)
                px = float(o[xx, j]) if x < nb else float(cl[-1, j])
                side = "long" if d["side"] > 0 else "short"
                cost = round_trip(entry, px, qty, side, c).total
                gross = d["side"] * (px - entry) * qty
                pos |= {"exit": px, "gross": gross, "costs": cost, "net": gross - cost, "exit_bar": xx if x >= nb else x}
            opened.append(pos)
            busy_until[j] = x
    for p in opened:
        if "net" in p and (p["exit_bar"] < nb or close_at_end):
            closed.append(p)
    still = [p for p in opened if "net" not in p]
    for p in still:
        p["last"] = float(cl[-1, p["j"]])
        p["unrealized"] = p["side"] * (p["last"] - p["entry"]) * p["qty"]
    frozen["last_bar"] = max(last, min(R.LAST_ENTRY, nb - 2))
    return closed, still, log


class AlgoBook:
    def __init__(self, cfg: dict, u: pd.DataFrame | None, day: str):
        self.cfg, self.day = cfg, day
        self.a = cfg["algo"]
        self.state = _load("state.json", {"pool": float(cfg["pool"]), "start_pool": float(cfg["pool"]), "days": []})
        self.path = ADIR / f"decisions_{day}.json"
        self.frozen = _load(self.path.name, {})
        if not self.frozen.get("rule"):
            self.frozen["rule"], self.frozen["why"] = self.pick_rule(day)
        if u is not None and not self.frozen.get("tickers"):
            from ..copilot.engine import liquid
            self.frozen["tickers"] = liquid(u, cfg | {"universe": cfg["universe"] | {"top_liquid": self.a["top_liquid"]}})
        self.fam, self.p = R.rule_by_name(self.frozen["rule"])

    def pick_rule(self, day: str) -> tuple[str, str]:
        if self.a.get("rule"):
            return self.a["rule"], "pinned in configs/copilot.yaml"
        tab = pd.read_parquet(TOURN) if TOURN.exists() else pd.DataFrame()
        name = T.leader(tab, day, self.a["window_days"], self.a["min_trades"], int(self.a["max_open"])) if len(tab) else None
        if name:
            return name, f"best net ₹/day over the last {self.a['window_days']} days among rules with ≥ {self.a['min_trades']} trades/day"
        return "xs_mom look=3 every=3 k=2 hold=12", "no tournament history yet: default rule"

    def _norms(self, ctx: dict, tickers: list[str]) -> tuple[np.ndarray, np.ndarray]:
        bv = np.array([float((ctx.get(t) or {}).get("avg_volume", np.nan)) / 75.0 for t in tickers])
        adr = np.array([float((ctx.get(t) or {}).get("adr", np.nan)) for t in tickers])
        return bv, adr

    def run(self, bars: dict, ctx: dict, close_at_end: bool = False):
        tick = [t for t in self.frozen.get("tickers", []) if t in bars]
        m = R.matrices_from_dict(bars, tick)
        if not m["tickers"] or m["open"].shape[0] < R.OPEN_BAR + 1:
            return [], [], [], m
        bv, adr = self._norms(ctx, m["tickers"])
        sig = R.signals(m, self.fam, self.p | {"_bar_vol": bv, "_adr": adr})
        closed, still, log = simulate(m, sig, self.p["hold"], self.cfg, float(self.state["pool"]), self.frozen, close_at_end)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.frozen), encoding="utf-8")
        return closed, still, log, m

    @staticmethod
    def _rows(trs: list[dict]) -> list[dict]:
        keep = ("ticker", "side", "signal_bar", "entry_bar", "exit_bar", "entry", "exit", "last", "qty", "gross", "costs",
                "net", "unrealized")
        return [{k: (("long" if v > 0 else "short") if k == "side" else v) for k, v in t.items() if k in keep} for t in trs]

    def step(self, bars: dict, ctx: dict, now: str) -> dict:
        closed, still, log, m = self.run(bars, ctx)
        net = sum(t["net"] for t in closed)
        unreal = sum(p["unrealized"] for p in still)
        status = {"phase": "trading", "day": self.day, "at": now, "rule": self.frozen["rule"], "why": self.frozen.get("why"),
                  "pool": self.state["pool"], "universe": len(m["tickers"]), "trades": self._rows(closed),
                  "positions": self._rows(still), "log": log, "day_pnl": round(net + unreal, 2),
                  "costs": round(sum(t["costs"] for t in closed), 2), "signals": len(self.frozen.get("entries", {})),
                  "message": f"{len(closed)} closed, {len(still)} open, {len(self.frozen.get('entries', {}))} signals; "
                             f"rule {self.frozen['rule']}."}
        _save("live.json", status)
        return status

    def settle(self, bars: dict, ctx: dict, now: str) -> dict:
        closed, _, log, m = self.run(bars, ctx, close_at_end=True)
        net = sum(t["net"] for t in closed)
        costs = sum(t["costs"] for t in closed)
        gross = sum(t["gross"] for t in closed)
        notional_sum = sum(t["entry"] * t["qty"] for t in closed)
        pool0 = float(self.state["pool"])
        self.state["pool"] = pool0 + net
        # the tournament: every rule on today's bars, in shadow
        shadow = pd.DataFrame()
        if m["tickers"] and m["open"].shape[0] > R.SQUARE:
            bv, adr = self._norms(ctx, m["tickers"])
            names = R.rule_names()
            rows = [(self.day, names[gi], side, r, i) for gi, i, j, side, r in R.day_trades(m, bv, adr)]
            tr = pd.DataFrame(rows, columns=["day", "name", "side", "gross", "bar"])
            if len(tr):
                n = T.notional(self.cfg)
                shadow = T.daily_table(tr, T.cost_needed(self.cfg), n).assign(source="live")
                old = pd.read_parquet(TOURN) if TOURN.exists() else pd.DataFrame()
                if len(old):
                    old = old[~((old["day"] == self.day) & (old.get("source", "replay") == "live"))]
                ADIR.mkdir(parents=True, exist_ok=True)
                pd.concat([old, shadow], ignore_index=True).to_parquet(TOURN)
        rec = {"day": self.day, "rule": self.frozen["rule"], "pool_start": pool0, "pool_end": self.state["pool"],
               "trades": len(closed), "net": round(net, 2), "gross": round(gross, 2), "costs": round(costs, 2),
               "gross_bps": round(gross / notional_sum * 1e4, 2) if notional_sum else None,
               "cost_bps": round(costs / notional_sum * 1e4, 2) if notional_sum else None,
               "skipped": sum(1 for d in self.frozen.get("entries", {}).values() if not d["taken"]),
               "halted": any(x["event"] == "loss stop" for x in log)}
        self.state["days"] = [d for d in self.state["days"] if d["day"] != self.day] + [rec]
        _save("state.json", self.state)
        tp = ADIR / "trades.csv"
        new = pd.DataFrame(self._rows(closed)).assign(day=self.day)
        old = pd.read_csv(tp) if tp.exists() else pd.DataFrame()
        if len(old):
            old = old[old["day"] != self.day]
        if len(new) or len(old):
            pd.concat([old, new], ignore_index=True).to_csv(tp, index=False)
        review = self.review(rec, shadow)
        _save(f"reviews/{self.day}.json", review)
        _save("live.json", {"phase": "closed", "day": self.day, "at": now, "pool": self.state["pool"], "result": rec,
                            "review": review, "trades": self._rows(closed),
                            "message": f"Day settled: {len(closed)} trades, net ₹{net:+,.0f} after ₹{costs:,.0f} costs."})
        return rec | {"review": review}

    def review(self, rec: dict, shadow: pd.DataFrame) -> dict:
        s = [f"{rec['trades']} trades with rule {rec['rule']}, net ₹{rec['net']:+,.0f} "
             f"(gross ₹{rec['gross']:+,.0f}, costs ₹{rec['costs']:,.0f}); pool ₹{rec['pool_end']:,.0f}."]
        if rec.get("gross_bps") is not None:
            s.append(f"Per trade it earned {rec['gross_bps']:+.1f} bps gross and paid {rec['cost_bps']:.1f} bps in costs: "
                     + ("the edge covered the costs." if rec["gross_bps"] > rec["cost_bps"] else "costs were bigger than the edge."))
        if rec["trades"] < self.a["min_trades"]:
            s.append(f"Below the {self.a['min_trades']}-trade target"
                     + (": the 2% daily loss stop halted new entries." if rec["halted"]
                        else f": the rule produced fewer signals ({rec['skipped']} more were skipped by the position limit)."))
        lessons = []
        if len(shadow):
            sh = shadow.sort_values("net", ascending=False)
            top = sh.iloc[0]
            lessons.append(f"Best rule today in shadow: {top['name']} ({int(top['trades'])} trades, {top['gross_bps']:+.1f} bps gross, "
                           f"₹{top['net']:+,.0f}).")
            pos = int((sh["net"] > 0).sum())
            lessons.append(f"{pos} of {len(sh)} rules made money after costs today.")
        days = pd.DataFrame(self.state["days"])
        if len(days):
            lessons.append(f"Forward record: {int(days['trades'].sum())} trades over {len(days)} days, net ₹{days['net'].sum():+,.0f}.")
        nxt, why = self.pick_rule(pd.Timestamp(self.day).date().isoformat() + "~")
        lessons.append(f"Tomorrow's rule: {nxt} ({why}).")
        return {"day": self.day, "summary": s, "learnings": lessons, "record": rec}
