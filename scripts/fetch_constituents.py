"""Download archived NIFTY 200 constituent lists from the Internet Archive (web.archive.org).

    python scripts/fetch_constituents.py

Writes data/constituents/nifty200_<YYYYMMDD>.csv (one per distinct archived copy of NSE's official
file, including the older CNX 200 name) and prints the snapshot dates, gaps and former members.
Used by research/constituents.py to rebuild index membership as it was on each date.
"""
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from niveshrl.research import constituents as cs  # noqa: E402

URLS = ["www.niftyindices.com/IndexConstituent/ind_nifty200list.csv",
        "niftyindices.com/IndexConstituent/ind_nifty200list.csv",
        "archives.nseindia.com/content/indices/ind_nifty200list.csv",
        "www.nseindia.com/content/indices/ind_nifty200list.csv",
        "nseindia.com/content/indices/ind_cnx200list.csv",
        "www.nseindia.com/content/indices/ind_cnx200list.csv"]
UA = {"User-Agent": "Mozilla/5.0 (research; niveshrl)"}


def get(url: str, tries: int = 3) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                data = r.read()
                if data[:2] == b"\x1f\x8b":                        # some copies are stored gzipped
                    import gzip
                    data = gzip.decompress(data)
                return data
        except Exception as e:
            if k == tries - 1:
                raise
            print(f"  retry ({e})", flush=True)
            time.sleep(5 * (k + 1))
    return b""


def main() -> None:
    cs.DIR.mkdir(parents=True, exist_ok=True)
    found = {}
    for u in URLS:
        q = f"https://web.archive.org/cdx/search/cdx?url={u}&output=json&filter=statuscode:200&collapse=digest&fl=timestamp,original"
        try:
            rows = json.loads(get(q) or b"[]")[1:]
        except Exception as e:
            print(f"{u}: CDX failed ({e})")
            continue
        print(f"{u}: {len(rows)} archived copies")
        for ts, orig in rows:
            found.setdefault(ts[:8], orig if orig.startswith("http") else "http://" + orig)
    saved = 0
    for day, orig in sorted(found.items()):
        out = cs.DIR / f"nifty200_{day}.csv"
        if out.exists():
            continue
        try:
            raw = get(f"https://web.archive.org/web/{day}id_/{orig}")
            df = pd.read_csv(io.BytesIO(raw))
            df.columns = [c.strip() for c in df.columns]
            if not {"Symbol", "ISIN Code"} <= set(df.columns) or len(df) < 150:
                print(f"  {day}: not a constituent file, skipped")
                continue
            df.to_csv(out, index=False)
            saved += 1
            print(f"  {day}: {len(df)} members saved")
        except Exception as e:
            print(f"  {day}: failed ({e})")
    snaps = cs.snapshot_files()
    print(f"\n{len(snaps)} snapshots on disk ({saved} new):", ", ".join(d.strftime("%Y-%m-%d") for d, _ in snaps))
    gaps = [(b - a).days for (a, _), (b, _) in zip(snaps, snaps[1:])]
    if gaps:
        print(f"largest gap between snapshots: {max(gaps)} days")
    h = cs.history()
    cur = set(pd.read_csv(cs.CURRENT)["Symbol"].str.strip())
    tm = cs.ticker_map(h)
    former = sorted({tm[s] for s in h["symbol"] if tm[s][:-3] not in cur})
    print(f"former members (not in today's list after renames): {len(former)}")


if __name__ == "__main__":
    main()
