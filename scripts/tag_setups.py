"""Tag the point-in-time swing-rule replay with each trade's starting context (market and stock trend, RSI zone,
distance from the 20-day high) for the stock page's 'similar past setups'.

    python scripts/tag_setups.py
"""
from niveshrl.research.tradecheck import TAGGED, TAG_ORDER, build_tagged

if __name__ == "__main__":
    t = build_tagged()
    print(f"{len(t):,} closed trades tagged -> {TAGGED}")
    print("missing tags:", t[TAG_ORDER].isna().mean().round(3).to_dict())
