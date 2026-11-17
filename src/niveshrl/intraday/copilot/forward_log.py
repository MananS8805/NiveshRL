"""Append-only, hash-chained forward log of the copilot's decisions and outcomes.

Every decision is appended the moment it is made (before its outcome can be known) and every outcome after the close.
Each entry carries ``prev`` (the previous entry's hash) and ``hash`` = sha256(prev + canonical JSON of the entry), so
editing, deleting or reordering any past entry breaks the chain from that point on and ``verify`` reports where.
This is what makes the forward record checkable: predictions provably existed before the results they are scored on.
Kept local (paper only); only aggregate scores and the latest hash are meant to be shared.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
GENESIS = "0" * 64


def _path():
    from . import CDIR                                     # read at call time (tests point CDIR at a temp folder)
    return CDIR / "forward_log.jsonl"


def _digest(prev: str, body: dict) -> str:
    return hashlib.sha256((prev + json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)).encode()).hexdigest()


def _tail() -> tuple[str, set]:
    """(last hash, keys already logged as decision/outcome)."""
    p = _path()
    last, seen = GENESIS, set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                last = e["hash"]
                seen.add((e["type"], e.get("key")))
    return last, seen


def append(entries: list[dict]) -> int:
    """Append new entries ({type, key, ...}); entries already logged with the same (type, key) are skipped."""
    if not entries:
        return 0
    prev, seen = _tail()
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "a", encoding="utf-8") as f:
        for e in entries:
            if (e["type"], e.get("key")) in seen:
                continue
            body = {**e, "logged_at": datetime.now(IST).isoformat(timespec="seconds"), "prev": prev}
            body["hash"] = _digest(prev, {k: v for k, v in body.items() if k != "hash"})
            f.write(json.dumps(body, sort_keys=True, default=str) + "\n")
            prev = body["hash"]
            seen.add((e["type"], e.get("key")))
            n += 1
    return n


def verify() -> dict:
    """Re-hash the whole chain. Returns {ok, entries, decisions, outcomes, head, broken_at}."""
    p = _path()
    out = {"ok": True, "entries": 0, "decisions": 0, "outcomes": 0, "head": GENESIS, "broken_at": None}
    if not p.exists():
        return out
    prev = GENESIS
    for i, line in enumerate(l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()):
        e = json.loads(line)
        body = {k: v for k, v in e.items() if k != "hash"}
        if e.get("prev") != prev or _digest(prev, body) != e.get("hash"):
            out.update(ok=False, broken_at=i)
            return out
        prev = e["hash"]
        out["entries"] += 1
        out["decisions" if e["type"] == "decision" else "outcomes"] += e["type"] in ("decision", "outcome")
    out["head"] = prev
    return out


def decisions_from(cards, day: str) -> list[dict]:
    if cards is None or not len(cards):
        return []
    return [{"type": "decision", "key": r.key, "day": day, "ticker": r.ticker, "setup": r.setup, "side": r.side,
             "signal_ts": r.ts, "ref": round(float(r.ref), 2), "stop": round(float(r.stop), 2),
             "p": None if r.p != r.p else round(float(r.p), 4), "take": bool(r.take)} for r in cards.itertuples()]


def outcomes_from(cards, day: str) -> list[dict]:
    if cards is None or not len(cards) or "r" not in cards:
        return []
    return [{"type": "outcome", "key": r.key, "day": day, "r_2r": None if r.r_2r_1x != r.r_2r_1x else round(float(r.r_2r_1x), 4),
             "r_eod": None if r.r_eod_1x != r.r_eod_1x else round(float(r.r_eod_1x), 4), "exit_reason": r.reason}
            for r in cards.itertuples()]
