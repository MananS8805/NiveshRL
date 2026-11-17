"""Copilot engine: signals -> cards (features, cost in R, calibrated probability) -> chronological, frozen decisions
under the tuned rules and fixed guardrails -> the paper account (sim.Account: next-bar-open fills, stops, targets,
square-off, NSE costs). After the close: every card gets its full-size outcome, the day review is written, the
rules are tuned on the evidence window and the model is refit weekly.

A card is the unit of evidence: every signal becomes a card whether or not it is traded, so the copilot learns from
the trades it skipped as much as from the ones it took, and the forward scorecard cannot be cherry-picked.
"""
from __future__ import annotations

import pandas as pd

from .. import sim
from ..costs import round_trip
from . import CDIR, config, load_json, rules as load_rules, save_json
from .model import CardModel
from .setups import all_signals

MODEL = CDIR / "model.pkl"
CARDS = CDIR / "cards.parquet"


def liquid(u: pd.DataFrame, cfg: dict) -> list[str]:
    uc = cfg["universe"]
    u = u[u["close"] >= uc["min_price"]].sort_values("turnover_cr", ascending=False)
    return u.index.tolist()[: uc["top_liquid"]] if "ticker" not in u else u["ticker"].tolist()[: uc["top_liquid"]]


def cost_r(ref: float, stop: float, side: str, cfg: dict) -> float:
    dist = abs(ref - stop)
    if dist <= 0:
        return float("nan")
    qty = max(1, int(cfg["pool"] * cfg["guardrails"]["risk_per_trade"] // dist))
    return round_trip(ref, ref, qty, side, cfg["costs"]).total / (dist * qty)


def cards_frame(signals: list[sim.Signal], cfg: dict, model: CardModel | None, day: str) -> pd.DataFrame:
    if not signals:
        return pd.DataFrame()
    rows = []
    for s in signals:
        rows.append(s.features | {"day": day, "ticker": s.ticker, "ts": str(s.ts), "bar": s.bar, "ref": s.ref,
                                  "stop": s.stop, "cost_r": cost_r(s.ref, s.stop, s.side, cfg)})
    df = pd.DataFrame(rows)
    df = df[df["cost_r"] <= cfg["guardrails"]["cost_r_max"]].reset_index(drop=True)
    df["p"] = model.gated_prob(df) if model is not None and model.ready and len(df) else float("nan")   # NaN = rules decide
    df["key"] = df["ticker"] + "|" + df["ts"] + "|" + df["setup"] + "|" + df["side"]
    return df


def outcome_cols(df: pd.DataFrame, bars: dict[str, pd.DataFrame], cfg: dict, slips=(1, 2)) -> pd.DataFrame:
    """Full-size R after costs for each card: 2R target exit and hold-to-square-off exit, at 1x and 2x slippage."""
    base = cfg["costs"]["slippage"]
    out = {f"r_{e}_{m}x": [] for e in ("2r", "eod") for m in slips}
    out["exit_bar"] = []
    out["reason"] = []
    for r in df.itertuples():
        sig = sim.Signal(r.ticker, pd.Timestamp(r.ts), int(r.bar), r.setup, r.side, float(r.ref), float(r.stop))
        b = bars.get(r.ticker)
        for e, tr in (("2r", cfg["guardrails"]["target_r"]), ("eod", 1000.0)):
            c2 = cfg | {"guardrails": cfg["guardrails"] | {"target_r": tr}}
            for m in slips:
                oc = sim.outcome(sig, b, c2, slip=base * m) if b is not None else None
                out[f"r_{e}_{m}x"].append(oc["r"] if oc else float("nan"))
                if e == "2r" and m == 1:
                    out["exit_bar"].append(oc["exit_bar"] if oc else -1)
                    out["reason"].append(oc["reason"] if oc else "")
    for k, v in out.items():
        df[k] = v
    df["r"] = df["r_2r_1x"]
    return df


def _minute(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m) - (9 * 60 + 15)


def passes(card, rules: dict) -> tuple[bool, str]:
    sr = rules["setups"].get(card["setup"], {})
    if not sr.get("enabled", True):
        return False, "setup benched by the learner"
    if card["minute"] > _minute(rules.get("last_entry", "14:30")):
        return False, f"after the {rules.get('last_entry')} entry cutoff"
    p = card.get("p", float("nan"))
    if p == p and p < sr.get("min_p", 0.45):
        return False, f"P(win) {p:.0%} below the {sr.get('min_p', 0.45):.0%} bar"
    return True, "passes the rules" + (f" (P(win) {p:.0%})" if p == p else " (model has no proven skill: rules decide)")


def choose_day(cards: pd.DataFrame, rules: dict, g: dict, rcol: str = "r", exit_col: str = "exit_bar") -> list[int]:
    """Backtest decisions in time order (same logic as live): at most max_trades_per_day, one trade per stock, no new
    trades once closed trades have lost the daily stop (in R: daily_loss_stop / risk_per_trade)."""
    taken, tickers = [], set()
    stop_r = -g["daily_loss_stop"] / g["risk_per_trade"]
    for i, c in cards.sort_values(["bar", "p"], ascending=[True, False]).iterrows():
        if len(taken) >= g["max_trades_per_day"] or c["ticker"] in tickers:
            continue
        closed = [cards.at[j, rcol] for j in taken if cards.at[j, exit_col] >= 0 and cards.at[j, exit_col] <= c["bar"]]
        if closed and sum(x for x in closed if x == x) <= stop_r:
            break
        ok, _ = passes(c, rules)
        if ok:
            taken.append(i)
            tickers.add(c["ticker"])
    return taken


HYPOTHESES = {   # registered 2026-11-16 from the replay, judged only on later live days (never on the data that suggested it)
    "rules only, hold to close": {"rules": "start", "exit": "eod",
                                  "why": "the only variant positive in both replay halves (+₹7,481 design, +₹7,150 holdout), "
                                         "noticed after seeing the holdout, so it is unproven"},
}


def forward_hypotheses(cfg: dict | None = None, since: str = "2026-11-17") -> list[dict]:
    """Daily net ₹ of each registered hypothesis on live cards from ``since`` on (same choice logic as live)."""
    cfg = cfg or config()
    if not CARDS.exists():
        return []
    c = pd.read_parquet(CARDS)
    c = c[(c.get("source", "replay") == "live") & (c["day"] >= since)] if "source" in c else c.iloc[0:0]
    unit = cfg["pool"] * cfg["guardrails"]["risk_per_trade"]
    start = {"setups": {s: dict(v) for s, v in cfg["rules"]["setups"].items()}, "last_entry": cfg["rules"]["last_entry"]}
    start = {"setups": {s: dict(v, min_p=0.0) for s, v in start["setups"].items()}, "last_entry": start["last_entry"]}
    out = []
    for name, h in HYPOTHESES.items():
        rcol = f"r_{h['exit']}_1x"
        days = []
        for d, cd in c.groupby("day"):
            cd = cd.reset_index(drop=True)
            idx = choose_day(cd, start, cfg["guardrails"], rcol=rcol)
            days.append({"day": d, "trades": len(idx), "net": float(cd.loc[idx, rcol].fillna(0).sum() * unit)})
        out.append({"name": name, "why": h["why"], "days": days})
    return out


class Copilot:
    """Live copilot (one trading day at a time). State in data/intraday/copilot/."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or config()
        self.g = self.cfg["guardrails"]
        self.state = load_json("state.json", {"pool": float(self.cfg["pool"]), "start_pool": float(self.cfg["pool"]),
                                              "days": []})
        self.rules = load_rules(self.cfg)
        self.model = CardModel.load(MODEL)

    def cards(self, bars, ctx, nifty, day: str) -> pd.DataFrame:
        sigs = all_signals(bars, ctx, self.g, nifty)
        return cards_frame(sigs, self.cfg, self.model, day)

    def run(self, bars, ctx, nifty, frozen: dict, day: str, close_at_end: bool = False, log: bool = False):
        cards = self.cards(bars, ctx, nifty, day)
        taken = 0
        rows = cards.sort_values(["bar", "p"], ascending=[True, False]).iterrows() if len(cards) else iter(())
        for _, c in rows:                                  # an hour with no setups yet is normal, not an error
            if c["key"] in frozen:
                taken += frozen[c["key"]]["take"]
                continue
            ok, why = passes(c, self.rules)
            if ok and taken >= self.g["max_trades_per_day"]:
                ok, why = False, f"already {self.g['max_trades_per_day']} trades today"
            frozen[c["key"]] = {"take": bool(ok), "why": why, "p": None if c["p"] != c["p"] else float(c["p"])}
            taken += bool(ok)
        if len(cards):
            cards["take"] = cards["key"].map(lambda k: frozen.get(k, {}).get("take", False))
            cards["why"] = cards["key"].map(lambda k: frozen.get(k, {}).get("why", ""))
            if log:                                          # write each decision before its outcome can be known
                from .forward_log import append, decisions_from
                append(decisions_from(cards, day))
        sigs = [sim.Signal(r.ticker, pd.Timestamp(r.ts), int(r.bar), r.setup, r.side, float(r.ref), float(r.stop))
                for r in cards.itertuples() if r.take] if len(cards) else []
        look = {(r.ticker, str(pd.Timestamp(r.ts))): r for r in cards.itertuples()} if len(cards) else {}

        def decide(s):
            r = look.get((s.ticker, str(s.ts)))
            p = float(r.p) if r is not None and r.p == r.p else float("nan")
            return "TAKE", p, (r.why if r is not None else "")
        acct = sim.Account(self.cfg, float(self.state["pool"]))
        trades, log = acct.run_day(sigs, bars, decide, close_at_end=close_at_end)
        return cards, trades, log, acct

    def top_cards(self, cards: pd.DataFrame, k: int = 3, recent_bars: int = 6) -> pd.DataFrame:
        """The best few cards right now (always shown, even below the bar, and labelled)."""
        if not len(cards):
            return cards
        c = cards.copy()
        c["recent"] = c["bar"] >= c["bar"].max() - recent_bars + 1
        c["rank_p"] = c["p"].fillna(0.5)
        # taken cards first, then the freshest, then the most probable: always k rows when the day has k cards
        return c.sort_values(["take", "recent", "rank_p"], ascending=False).head(k).drop(columns=["recent", "rank_p"])

    def settle(self, day: str, bars, ctx, nifty, frozen: dict) -> dict:
        from .brief import day_review
        from .tuner import tune
        cards, trades, log, _ = self.run(bars, ctx, nifty, frozen, day, close_at_end=True)
        if len(cards):
            cards = outcome_cols(cards, bars, self.cfg)
            cards["source"] = "live"
            from .forward_log import append, decisions_from, outcomes_from
            append(decisions_from(cards, day))                 # any decision not yet logged, then every outcome
            append(outcomes_from(cards, day))
        hist = pd.read_parquet(CARDS) if CARDS.exists() else pd.DataFrame()
        if len(hist):
            hist = hist[~((hist["day"] == day) & (hist.get("source", "live") == "live"))]
        allc = pd.concat([hist, cards], ignore_index=True) if len(cards) else hist
        if len(allc):
            CDIR.mkdir(parents=True, exist_ok=True)
            allc.to_parquet(CARDS)
        net = sum(t.net for t in trades)
        pool0 = float(self.state["pool"])
        self.state["pool"] = pool0 + net
        new_rules, changes = tune(self.rules, allc, day, self.cfg["tuning"])
        save_json("rules.json", new_rules)
        refit = ""
        if not self.model.ready or len(set(allc["day"]) - set(self.model.days)) >= self.cfg["model"]["refit_every_days"]:
            m = CardModel.fit(allc)
            if m.ready:
                m.save(MODEL)
                self.model = m
                refit = (f"model refit on {m.n:,} cards over {len(m.days)} days; held-out AUC {m.skill.get('auc', float('nan')):.2f}, "
                         + ("skilled: it filters trades" if m.skilled else "not skilled: the rules decide alone"))
        rec = {"day": day, "pool_start": pool0, "pool_end": self.state["pool"], "trades": len(trades),
               "net": round(net, 2), "gross": round(sum(t.gross for t in trades), 2),
               "costs": round(sum(t.costs for t in trades), 2), "cards": int(len(cards)),
               "avg_r": float(pd.Series([t.r for t in trades]).mean()) if trades else None,
               "cards_avg_r": float(cards["r"].mean()) if len(cards) else None, "changes": len(changes), "refit": refit}
        self.state["days"] = [d for d in self.state["days"] if d["day"] != day] + [rec]
        save_json("state.json", self.state)
        tp = CDIR / "trades.csv"
        new = pd.DataFrame([t.to_dict() | {"day": day} for t in trades]).drop(columns=["cost_detail"], errors="ignore")
        old = pd.read_csv(tp) if tp.exists() else pd.DataFrame()
        if len(old):
            old = old[old["day"] != day]
        if len(new) or len(old):
            pd.concat([old, new], ignore_index=True).to_csv(tp, index=False)
        from .forward_log import verify
        chain = verify()
        rec["forward_log"] = {k: chain[k] for k in ("ok", "entries", "head")}
        review = day_review(day, rec, cards, trades, changes, nifty, self.state, refit)
        save_json(f"reviews/{day}.json", review)
        return rec | {"review": review}
