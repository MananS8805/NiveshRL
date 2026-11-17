"""Build the copilot's 5-minute history: the stored replay bars (13 Jul - 1 Oct 2026) + archived live days + Yahoo's
latest 60 days, for the most liquid stocks and NIFTY. Writes data/intraday/copilot/hist5m.parquet.

    python scripts/copilot_data.py [--top 200]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from niveshrl.intraday import DIR  # noqa: E402
from niveshrl.intraday.copilot import CDIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--out", default="hist5m.parquet", help="file name in data/intraday/copilot/")
    a = ap.parse_args()
    u = pd.read_csv(DIR / "universe.csv")
    u = u[u["close"] >= 100].sort_values("turnover_cr", ascending=False)
    tickers = u["ticker"].head(a.top).tolist()
    keep = set(tickers) | {"^NSEI"}
    parts = []
    if (DIR / "bars5m.parquet").exists():
        b = pd.read_parquet(DIR / "bars5m.parquet")
        parts.append(b[b["ticker"].isin(keep)])
    for f in sorted((DIR / "archive").glob("*.parquet")):
        b = pd.read_parquet(f)
        parts.append(b[b["ticker"].isin(keep)][["ticker", "ts", "open", "high", "low", "close", "volume"]])
    from niveshrl.intraday import replay
    import niveshrl.intraday.replay as R
    R.BARS = CDIR / "_fresh.parquet"                         # don't overwrite the agent's replay bars
    CDIR.mkdir(parents=True, exist_ok=True)
    fresh = replay.download(tickers, days=60, progress=lambda f: print(f"download {f:.0%}", flush=True))
    parts.append(fresh)
    allb = pd.concat(parts, ignore_index=True)
    allb["ts"] = pd.to_datetime(allb["ts"])
    allb = allb.drop_duplicates(["ticker", "ts"], keep="last").sort_values(["ticker", "ts"])
    allb.to_parquet(CDIR / a.out)
    (CDIR / "_fresh.parquet").unlink(missing_ok=True)
    days = allb["ts"].dt.normalize().nunique()
    print(f"{a.out}: {len(allb):,} bars, {allb['ticker'].nunique()} tickers, {days} days "
          f"({allb['ts'].min().date()} to {allb['ts'].max().date()})")


if __name__ == "__main__":
    main()
