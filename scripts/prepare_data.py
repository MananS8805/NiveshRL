"""Download NSE data, clean it, and print a data-quality report.

    python scripts/prepare_data.py [--refresh]
"""
import argparse

from niveshrl.pipeline import load_all

ap = argparse.ArgumentParser()
ap.add_argument("--refresh", action="store_true", help="re-download even if cached")
args = ap.parse_args()

cfg, md, fs = load_all(refresh=args.refresh)
print(f"{len(md.tickers)} stocks, {len(md.dates)} trading days: {md.dates[0].date()} -> {md.dates[-1].date()}")
print(f"features: stock {fs.stock.shape}, market {fs.market.shape}, valid from {md.dates[fs.valid_from].date()}")
for split, (a, b) in cfg["splits"].items():
    print(f"  {split:8s} {a} -> {b}: {int(md.split_mask(a, b).sum())} days")
zero_vol = (md.volume == 0).mean()
print("share of zero-volume days (top 5):")
print(zero_vol.sort_values(ascending=False).head().to_string())
stale = md.jumps.attrs.get("stale_runs", {})
print(f"stale-price runs >= 10 days: {stale or 'none'}")
print(f"repaired one-day bad prints: {md.jumps.attrs.get('repaired') or 'none'}")
if len(md.jumps):
    print(f"\n{len(md.jumps)} single-day moves above {cfg['data']['jump_threshold']:.0%} (check vs NSE corporate actions):")
    print(md.jumps.to_string())
else:
    print("no suspicious single-day jumps")
