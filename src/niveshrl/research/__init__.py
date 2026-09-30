"""Research platform: cross-sectional deep-learning rankers, volatility and regime
models, and a strategy backtester over the NIFTY 200 universe.

Separate from the RL allocator (``niveshrl.env`` & co.), which works on a fixed
29-stock universe with weekly decisions. Everything here is monthly and
cross-sectional, and tolerates stocks entering and leaving the data.
"""
