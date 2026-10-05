"""The agent: one trading day end to end, shared by replay (history) and live (market hours).

Day flow: in-play selection → setups on those stocks (only from the bar they joined; never 'late' signals that
already happened) → ML probability + bandit decision → the paper Account trades within the guardrails → a random
control account trades the same signals on a coin flip → after the close: costs, speculative tax for the
financial year, learning (bandit on every signal's shadow outcome; scorer retrained on all past outcomes) →
everything saved under ``root`` (data/intraday/ for live, data/intraday/replay/ for history).
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .bandit import Bandit
from .costs import fy_of, speculative_tax
from .scorer import Scorer
from .setups import Signal, day_signals, nifty_trend
from .sim import Account, shadow

OPEN_MIN = 9 * 60 + 15


def in_play_matrix(bars: dict[str, pd.DataFrame], ctx: dict[str, dict], ucfg: dict) -> tuple[list[str], np.ndarray]:
    """Vectorised in-play test for every stock at every bar, using bars 0..k only at column k:
    (|gap| ≥ 1% or volume pace ≥ 1.5×) and range used ≥ 25% of ADR. Returns tickers and a (N × bars) score
    matrix (NaN = not in play; score = range used × capped volume pace)."""
    tick = [t for t in bars if ctx.get(t, {}).get("prev_close") and ctx[t].get("adr") == ctx[t].get("adr")
            and ctx[t].get("adr") and ctx[t].get("avg_volume")]
    nb = max((len(bars[t]) for t in tick), default=0)
    S = np.full((len(tick), nb), np.nan)
    el = (np.arange(nb) + 1) * 5 / 375
    for i, t in enumerate(tick):
        b, c = bars[t], ctx[t]
        h, l, v = (b[k].to_numpy(float) for k in ("high", "low", "volume"))
        n = len(h)
        used = (np.maximum.accumulate(h) - np.minimum.accumulate(l)) / (c["adr"] * c["prev_close"])
        pace = np.cumsum(np.nan_to_num(v)) / np.maximum(c["avg_volume"] * el[:n], 1.0)
        gap = abs(float(b["open"].iloc[0]) / c["prev_close"] - 1)
        ok = ((gap >= ucfg["gap_min"]) | (pace >= ucfg["volume_pace_min"])) & (used >= ucfg["range_used_min"])
        S[i, :n] = np.where(ok, used * np.minimum(pace, 5.0), np.nan)
    return tick, S


def in_play(bars: dict[str, pd.DataFrame], ctx: dict[str, dict], k: int, ucfg: dict) -> list[tuple[str, float]]:
    """Stocks in play at bar k (bars 0..k only), best first."""
    tick, S = in_play_matrix(bars, ctx, ucfg)
    if not tick or k >= S.shape[1]:
        return []
    col = S[:, k]
    idx = np.flatnonzero(np.isfinite(col))
    return [(tick[i], float(col[i])) for i in idx[np.argsort(-col[idx])]]


@dataclass
class DayResult:
    day: str
    pool_start: float
    pool_end: float
    gross: float
    costs: float
    net: float
    tax_accrued: float
    trades: int
    signals: int
    skipped: int
    control_net: float
    hit_target: bool
    learned: list

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class Agent:
    def __init__(self, root: Path, cfg: dict | None = None, seed: int = 0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or config()
        self.state = self._load_json("state.json", {"pool": float(self.cfg["pool"]), "control_pool": float(self.cfg["pool"]),
                                                     "start_pool": float(self.cfg["pool"]), "fy_net": {}, "days": []})
        self.bandit = Bandit.load(self.cfg, self.root / "bandit.json", seed)
        self.scorer = Scorer.load(self.root / "scorer.pkl")
        self.seed = seed

    # ------------------------------------------------------------------ persistence
    def _load_json(self, name, default):
        p = self.root / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                pass
        return default

    def _save_json(self, name, obj):
        p = self.root / name
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
        tmp.replace(p)

    def history(self) -> pd.DataFrame:
        p = self.root / "shadow.parquet"
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    # ------------------------------------------------------------------ one day
    def signals_for_day(self, bars: dict[str, pd.DataFrame], ctx: dict[str, dict], nifty: pd.DataFrame | None,
                        upto: int | None = None) -> list[Signal]:
        ucfg, g = self.cfg["universe"], self.cfg["guardrails"]
        trend = nifty_trend(nifty) if nifty is not None and len(nifty) else None
        nb = max((len(b) for b in bars.values()), default=0)
        last = nb - 1 if upto is None else min(upto, nb - 1)
        joined: dict[str, int] = {}
        tick, S = in_play_matrix(bars, ctx, ucfg)
        for k in range(2, min(last + 1, S.shape[1])):    # stocks join at the bar they qualify, top N, never padded
            if len(joined) >= ucfg["in_play_top"]:
                break
            col = S[:, k]
            for i in np.flatnonzero(np.isfinite(col))[np.argsort(-col[np.isfinite(col)])]:
                if tick[i] not in joined and len(joined) < ucfg["in_play_top"]:
                    joined[tick[i]] = k
        sigs = []
        for t, k0 in joined.items():
            for s in day_signals(t, bars[t], ctx.get(t, {}), self.cfg["setups"], g, upto=last, nifty_trend=trend):
                if s.bar >= k0:                           # only signals from when it was being followed
                    s.features["in_play_rank"] = float(list(joined).index(t))
                    sigs.append(s)
        return sigs

    def cost_r(self, sig: Signal) -> float:
        """Round-trip costs as a share of R for this signal (proportional charges + slippage, both sides)."""
        c = self.cfg["costs"]
        frac = (c["stt_sell"] + c["stamp_buy"] + 2 * (c["exchange"] + c["sebi"]) * (1 + c["gst"])
                + 2 * c["slippage"] + 2 * c["brokerage_pct"] * (1 + c["gst"]))
        dist = abs(sig.ref - sig.stop)
        return frac * sig.ref / dist if dist > 0 else float("inf")

    def decide(self, sig: Signal) -> tuple[str, float, str]:
        cr = self.cost_r(sig)
        sig.features["cost_r"] = float(cr)
        if cr > self.cfg["guardrails"].get("cost_r_max", 1e9):
            return "SKIP", float("nan"), f"costs would eat {cr:.2f}R (> {self.cfg['guardrails']['cost_r_max']:.2f}R): stop too tight"
        p = self.scorer.prob(sig.features)
        action, why = self.bandit.decide(sig.features, p, self.scorer.ready)
        return action, p, why

    def frozen_decide(self, cache: dict):
        """Decisions made once per signal and then frozen (live re-runs the day every 5 minutes)."""
        def f(sig):
            key = f"{sig.ticker}|{sig.ts}|{sig.setup}|{sig.side}"
            if key not in cache:
                cache[key] = list(self.decide(sig))
            a, p, w = cache[key]
            return a, float(p) if p is not None else float("nan"), w
        return f

    def trade_day(self, day: str, bars: dict[str, pd.DataFrame], ctx: dict[str, dict],
                  nifty: pd.DataFrame | None = None, learn: bool = True,
                  decisions: dict | None = None) -> tuple[DayResult, list, list]:
        sigs = self.signals_for_day(bars, ctx, nifty)
        pool0 = float(self.state["pool"])
        decide = self.frozen_decide(decisions) if decisions is not None else self.decide
        trades, log = Account(self.cfg, pool0).run_day(sigs, bars, decide)
        rng = np.random.default_rng(zlib.crc32(f"{day}|{self.seed}".encode()))   # reproducible coin flips
        coin = {(s.ticker, str(s.ts)): rng.random() < 0.5 for s in sigs}
        ctrl, _ = Account(self.cfg, float(self.state["control_pool"])).run_day(
            sigs, bars, lambda s: ("TAKE" if coin[(s.ticker, str(s.ts))] else "SKIP", float("nan"), "random control"))
        gross = sum(t.gross for t in trades)
        costs = sum(t.costs for t in trades)
        net = gross - costs
        fy = fy_of(pd.Timestamp(day))
        before = speculative_tax(self.state["fy_net"].get(fy, 0.0), self.cfg["tax"])
        self.state["fy_net"][fy] = self.state["fy_net"].get(fy, 0.0) + net
        tax_delta = speculative_tax(self.state["fy_net"][fy], self.cfg["tax"]) - before
        self.state["pool"] = pool0 + net
        self.state["control_pool"] = float(self.state["control_pool"]) + sum(t.net for t in ctrl)
        hit = net >= self.cfg["learning"]["reward_bonus_day_return"] * pool0
        learned = []
        if learn:
            sh = shadow(sigs, bars, self.cfg)
            if len(sh):
                sh["day"] = day
                taken = {(t.ticker, t.signal_ts) for t in trades}
                learned = self.bandit.update(sh, taken, self.cfg["learning"]["reward_bonus_r"] if hit else 0.0, day)
                hist = pd.concat([self.history(), sh], ignore_index=True)
                hist.to_parquet(self.root / "shadow.parquet")
                self.scorer = Scorer().fit(hist, self.seed, self.cfg["learning"].get("half_life_days"))
                if self.scorer.ready:
                    self.scorer.save(self.root / "scorer.pkl")
            self.bandit.save(self.root / "bandit.json")
        res = DayResult(day, pool0, float(self.state["pool"]), round(gross, 2), round(costs, 2), round(net, 2),
                        round(tax_delta, 2), len(trades), len(sigs), sum(1 for e in log if e.get("event") == "skip"),
                        round(sum(t.net for t in ctrl), 2), bool(hit), learned)
        self.state["days"] = [d for d in self.state["days"] if d["day"] != day] + [res.to_dict()]
        self._save_json("state.json", self.state)
        tpath = self.root / "trades.csv"
        new = pd.DataFrame([t.to_dict() | {"day": day} for t in trades])
        if len(new):
            new = new.drop(columns=["cost_detail"]).assign(**{f"cost_{k}": [t.cost_detail.get(k, 0) for t in trades]
                                                               for k in ("brokerage", "stt", "exchange", "sebi", "stamp", "gst")})
            old = pd.read_csv(tpath) if tpath.exists() else pd.DataFrame()
            if len(old):
                old = old[old["day"] != day]
            pd.concat([old, new], ignore_index=True).to_csv(tpath, index=False)
        self._save_json(f"log_{day}.json", log)
        return res, trades, log

    # ------------------------------------------------------------------ reporting
    def summary(self) -> dict:
        days = pd.DataFrame(self.state["days"])
        if days.empty:
            return {"days": 0}
        tr = pd.read_csv(self.root / "trades.csv") if (self.root / "trades.csv").exists() else pd.DataFrame()
        tax = sum(speculative_tax(v, self.cfg["tax"]) for v in self.state["fy_net"].values())
        start = float(self.state["start_pool"])
        return {"days": len(days), "pool": float(self.state["pool"]), "start": start,
                "net": float(days["net"].sum()), "gross": float(days["gross"].sum()), "costs": float(days["costs"].sum()),
                "tax_due": tax, "after_tax": float(self.state["pool"]) - tax,
                "control_pool": float(self.state["control_pool"]), "trades": int(len(tr)),
                "win_rate": float((tr["net"] > 0).mean()) if len(tr) else float("nan"),
                "avg_r": float(tr["r"].mean()) if len(tr) else float("nan"),
                "days_hit_target": int(days["hit_target"].sum()), "best_day": float(days["net"].max()),
                "worst_day": float(days["net"].min())}
