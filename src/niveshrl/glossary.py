"""Financial, technical and model glossary shared by the desktop app (and later the Streamlit app).

Every metric, model output and chart the app shows has an ``Entry``: what it is, how *this app*
computes it, what ranges of values usually mean, how it helps, caveats and related terms. Model
entries also point at the app's own measured results (``results``), which the UI reads live from
``report/results/*.csv`` so the numbers never go stale.

Ranges are common conventions and this project's own measurements, not trading rules or advice.

``lookup(label)`` maps any label shown in the app (KPI tile, table header, chart title) to an entry;
``band(entry, value)`` finds the range a value falls in.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

INF = math.inf
CATEGORIES = ["Market", "Valuation", "Quality & growth", "Technicals", "News & analysts", "Model outputs",
              "Model quality", "Models", "Backtest & risk", "Volatility forecaster", "Regimes", "Charts", "Desk",
              "Stock deep-dive"]


@dataclass
class Entry:
    key: str
    title: str
    category: str
    what: str
    how: str = ""
    bands: list = field(default_factory=list)       # [(upper_bound, label, meaning), ...] ascending; value < upper
    unit: str = "num"                               # num | pct (stored as a fraction) | x
    use: str = ""
    caveat: str = ""
    related: list = field(default_factory=list)
    aliases: list = field(default_factory=list)
    results: tuple | None = None                    # (csv name, row) of this app's measured record
    reading: str = ""                               # charts: how to read it


G: dict[str, Entry] = {}


def E(key, title, category, what, **kw) -> Entry:
    e = Entry(key, title, category, what, **kw)
    G[key] = e
    return e


# =====================================================================================================
# Market
# =====================================================================================================
E("nifty", "NIFTY 50", "Market",
  "India's benchmark index: the 50 largest, most liquid NSE companies, weighted by free-float market cap.",
  how="Daily close from Yahoo (^NSEI); live value from the Yahoo stream during market hours (09:15–15:30 IST).",
  use="The yardstick for 'the market'. Every return in the app is easier to judge next to NIFTY's.",
  caveat="Price index: excludes dividends (~1.2%/yr). Heavily weighted to banks, IT and energy.",
  related=["ret_1d", "ret_ytd", "india_vix", "breadth"], aliases=["NIFTY", "NIFTY 50 (weekly)", "NIFTY 50 (buy & hold)"])
E("ret_1d", "1-day return", "Market",
  "Percentage change from the previous close to the latest price (or today's close).",
  how="Live tick's change vs previous close when streaming, otherwise last close vs the one before.",
  bands=[(-0.03, "big fall", "a large one-day drop; for NIFTY, under 3% of days are this bad"),
         (-0.01, "down day", "clearly negative"), (0.01, "normal", "typical daily noise for a stock or index"),
         (0.03, "up day", "clearly positive"), (INF, "big rise", "a large one-day jump; check for news")],
  unit="pct", use="Spot what moved today and whether it was unusual for that stock.",
  caveat="Single-day moves are mostly noise; look at volume and news before reading meaning into them.",
  related=["vol_ratio", "gap_pct", "atr_pct"], aliases=["1D", "1D %", "Chg", "1D (watchlist)"])
E("ret_1w", "1-week return", "Market", "Return over the last 5 trading days.", unit="pct", aliases=["1W", "1W %"],
  related=["ret_1m"])
E("ret_1m", "1-month return", "Market", "Return over the last 21 trading days (about one calendar month).",
  unit="pct", bands=[(-0.10, "sharp fall", "a large monthly decline"), (-0.03, "weak", "underperforming"),
                     (0.03, "flat", "little net change"), (0.10, "strong", "solid monthly gain"),
                     (INF, "very strong", "a big run; momentum but also short-term stretch")],
  use="Short-term trend context; the monthly rankers deliberately skip the latest month (it tends to reverse).",
  aliases=["1M", "1M %", "1M ret"], related=["ret_3m", "momentum_model"])
E("ret_3m", "3-month return", "Market", "Return over the last 63 trading days.", unit="pct", aliases=["3M %", "3M"])
E("ret_6m", "6-month return", "Market", "Return over the last 126 trading days.", unit="pct", aliases=["6M %", "6M"])
E("ret_1y", "1-year return", "Market", "Return over the last 252 trading days.", unit="pct", aliases=["1Y %", "1Y"])
E("ret_ytd", "Year-to-date return", "Market", "Change since the last close of the previous calendar year.",
  unit="pct", aliases=["YTD"], related=["nifty"])
E("gap_pct", "Gap %", "Market",
  "How far today's open was from yesterday's close, as a percentage.",
  how="Adjusted open ÷ previous adjusted close − 1.",
  bands=[(-0.02, "gap down", "opened well below yesterday; usually news or a weak global cue"),
         (-0.005, "small gap down", ""), (0.005, "no gap", "opened near yesterday's close"),
         (0.02, "small gap up", ""), (INF, "gap up", "opened well above yesterday; news or strong cues")],
  unit="pct", use="Large gaps show overnight news being priced in; many traders watch whether a gap 'fills'.",
  aliases=["Gap %"], related=["ret_1d"])
E("india_vix", "India VIX", "Market",
  "The market's expected NIFTY volatility over the next 30 days, in annualised %, implied from NIFTY option prices. "
  "Often called the 'fear gauge'.",
  how="Yahoo ^INDIAVIX. The tile turns red when VIX is above its own 1-year median.",
  bands=[(12, "calm", "options price small moves; complacency is possible"),
         (16, "normal", "typical Indian market conditions"),
         (22, "nervous", "traders are paying up for protection; expect bigger daily swings"),
         (30, "fearful", "stress conditions; daily NIFTY moves of 1.5–2% are common"),
         (INF, "panic", "crisis levels (e.g. March 2020 above 80)")],
  use="Size positions and stops to the volatility regime; rising VIX with falling NIFTY = stress building.",
  caveat="VIX measures expected size of moves, not direction. High VIX periods have often preceded rebounds.",
  related=["regime", "vol_60d", "atr_pct"], aliases=["VIX", "INDIA VIX", "vix"])
E("vix_chg_20", "VIX change (20 days)", "Regimes", "Change in India VIX over the last 20 trading days.",
  aliases=["vix_chg_20"], related=["india_vix", "regime"])
E("breadth", "Breadth (above 200-day average)", "Market",
  "Share of NIFTY 200 stocks trading above their own 200-day moving average: how broad the market's strength is.",
  how="Count of stocks with last price > SMA200, divided by stocks with enough history.",
  bands=[(0.25, "very weak", "most stocks are in long-term downtrends; washed-out markets often bottom here"),
         (0.45, "weak", "more stocks below their long-term trend than above"),
         (0.60, "mixed", "no broad trend"),
         (0.80, "healthy", "most stocks in uptrends"),
         (INF, "very strong / stretched", "almost everything is rising; often late in a rally")],
  unit="pct", use="An index can rise on a few giants while most stocks fall; breadth tells you which it is.",
  related=["adv_dec", "regime", "vs_sma", "chart_breadth"], aliases=["Breadth >200DMA", "breadth", "BREADTH >200DMA", "% above 200-day"])
E("global_markets", "Global markets & macro", "Market",
  "World indices (US, Europe, Asia), Brent crude, gold, silver, the rupee (USD/INR), the dollar index and the US 10-year "
  "Treasury yield, with their changes over 1 day to 1 year.",
  how="Yahoo daily closes, refreshed every 30 minutes. Each market shows its own last close ('As of'): US markets close "
      "after Indian markets, so their '1D' is usually the previous evening's move.",
  reading="Indian stocks often open in the direction of the US close and Asian mornings. Rising crude and a weaker rupee "
          "hurt India (a large oil importer; foreign investors lose on the currency); a rising US 10-year yield and a "
          "stronger dollar tend to pull foreign money out of emerging markets.",
  use="Context for the next open and for why the whole market moves on a day without Indian news.",
  caveat="Relationships shift over time; see the correlation column for how tight each one actually was last year.",
  related=["global_corr", "nifty", "india_vix"],
  aliases=["S&P 500", "Nasdaq", "Dow Jones", "FTSE 100", "DAX", "Nikkei 225", "Hang Seng", "Shanghai", "Brent crude",
           "Gold", "Silver", "USD/INR", "Dollar index", "US 10Y yield", "Market", "As of", "Global && macro"])
E("global_corr", "Correlation with NIFTY (weekly, 1 year)", "Market",
  "How closely each market's weekly returns moved with NIFTY's over the last 52 weeks, from −1 (opposite) to +1.",
  how="Friday-to-Friday returns, so markets in different time zones line up.",
  bands=[(-0.3, "moves against NIFTY", "e.g. crude, the dollar and USD/INR: up when Indian stocks fall"),
         (0.3, "little relation", ""), (INF, "moves with NIFTY", "global risk appetite drives both")],
  use="Tells you which overnight moves have actually mattered for Indian stocks recently.",
  related=["global_markets", "corr_nifty"], aliases=["Corr. with NIFTY (weekly, 1y)"])
E("chart_breadth", "Breadth history", "Charts",
  "The share of NIFTY 200 stocks above their 50-day and 200-day averages, every day for 3 years.",
  reading="Breadth turning up from very low levels (under ~20% above the 50-day) has often marked washed-out markets; "
          "NIFTY making new highs while breadth falls (a divergence) means fewer stocks are carrying the index.",
  caveat="Today's members only (stocks that left the index are missing), so older readings are slightly flattering.",
  related=["breadth", "ad_line", "highs_lows"],
  aliases=["% above 50-day", "Breadth 1 month ago", "Share of stocks above their 50- and 200-day averages (%)"])
E("ad_line", "Advance-decline line", "Charts",
  "The running total of (stocks up − stocks down) each day.",
  reading="A rising line = most stocks are rising, a healthy, broad advance. An index rising while the A/D line falls "
          "is a narrow rally led by a few large stocks.", related=["adv_dec", "chart_breadth"],
  aliases=["A/D line", "A/D line, 1 month", "Advance-decline line (cumulative advancers − decliners)"])
E("highs_lows", "New highs minus new lows", "Charts",
  "Each day, how many stocks closed above their previous 52-week high minus how many closed below their 52-week low "
  "(shown as a 5-day average next to NIFTY's change).",
  reading="Expanding new highs confirm an uptrend; a market where lows outnumber highs is under broad pressure even if "
          "the index holds up.", related=["new_52w", "chart_breadth"],
  aliases=["New highs − lows", "Highs − lows", "New 52-week highs minus lows (5-day average) and NIFTY % change"])
E("rrg", "Relative rotation graph (sector rotation)", "Market",
  "Every NSE industry plotted by its relative trend against NIFTY (RS-Ratio, across) and whether that relative trend is "
  "improving (RS-Momentum, up). Four quadrants: Leading (strong and strengthening), Weakening (strong but fading), "
  "Lagging (weak and weakening) and Improving (weak but recovering).",
  how="Weekly. RS = equal-weight industry index ÷ NIFTY; RS-Ratio = 100 × RS ÷ its 10-week average; RS-Momentum = 100 × "
      "RS-Ratio ÷ its value 4 weeks earlier. Tails show the last 5 weeks.",
  reading="Sectors tend to rotate clockwise. A sector moving from Improving into Leading is gaining leadership; one "
          "sliding from Leading into Weakening is losing it. Long tails = fast rotation.",
  use="Shows where money is rotating, to favour stocks in leading or improving sectors and question buys in lagging ones.",
  caveat="An open approximation of the JdK RRG (the commercial formula is proprietary). Relative strength, not absolute: "
         "a 'Leading' sector can still fall if the whole market falls. Not tested for edge in this app.",
  related=["rs_line", "chart_sectors"],
  aliases=["RS-Ratio", "RS-Momentum", "Quadrant", "4 weeks ago", "13-week return vs NIFTY", "Sector rotation",
           "Sector rotation (relative rotation graph)"])
E("option_chain", "Option chain and positioning", "Market",
  "Every call and put strike for an expiry with its open interest (OI, contracts still open), today's change in OI, "
  "volume, implied volatility and last price, from NSE. Summaries: put-call ratio, max pain, the strikes with the most "
  "call and put OI, and at-the-money implied volatility.",
  how="NSE's public option-chain API (optional, off by default). Max pain = the expiry price at which option buyers' "
      "total payoff is smallest. Put-call ratio = total put OI ÷ total call OI.",
  reading="Large call OI above the price is often read as resistance (writers expect it not to be crossed), large put OI "
          "below as support. A PCR well above 1 means many puts are open: read either as hedging/fear or, contrarian, "
          "as support. Prices are said to gravitate towards max pain near expiry; the evidence for that is weak.",
  use="Shows where traders have positioned and how much movement options are pricing in (ATM IV).",
  caveat="Positioning is not prediction: OI mixes hedges, spreads and naked bets, and writers are often wrong in trends. "
         "NSE's data can be delayed and the endpoint can stop working. Not tested for edge in this app.",
  related=["implied_vol", "open_interest", "vol_cone"],
  aliases=["Put-call ratio (OI)", "Max pain", "Highest call OI", "Highest put OI", "ATM implied vol", "Option chain",
           "Call LTP", "Put LTP", "Call volume", "Put volume", "Index options positioning (nearest expiry)",
           "F&&O && shareholding", "Flows && F&&O"])
E("implied_vol", "Implied volatility (IV)", "Market",
  "The annualised volatility that makes an option's market price fair under the Black-Scholes model: how much "
  "movement option buyers are paying for.",
  bands=[(15, "calm", "for an index; single stocks are usually higher"), (30, "normal for a stock", ""),
         (50, "elevated", "results or news expected"), (INF, "very high", "")],
  use="Compare with the stock's realised or forecast volatility (Risk tab): IV far above it means options are "
      "expensive, often before results. ATM IV ÷ √12 ≈ the 1-month move options price in.",
  related=["option_chain", "vol_cone", "india_vix"], aliases=["Call IV", "Put IV"])
E("open_interest", "Open interest (OI)", "Market",
  "The number of option (or futures) contracts open at the end of the day. ΔOI = the change today: rising OI = new "
  "positions, falling OI = positions being closed.",
  reading="Price up with rising call writing (call OI up) can cap a move; put OI building under the price as it rises "
          "is often read as support being added.",
  related=["option_chain"], aliases=["Call OI", "Put OI", "Call ΔOI", "Put ΔOI"])
E("shareholding", "Shareholding pattern", "Quality & growth",
  "The % of shares held by the promoter and promoter group versus the public, quarter by quarter, from the company's "
  "filings to NSE.",
  reading="A rising promoter stake (buying from the market) is usually read as confidence; a steady fall can mean "
          "selling or dilution. Large one-quarter changes deserve a look at the filing.",
  caveat="Quarterly and filed weeks later. NSE's pledge data endpoint currently returns nothing, so pledges are not shown.",
  related=["promoters", "institutions"],
  aliases=["Promoter %", "Public %", "Employee trusts %", "Promoter change (pp)", "Filed",
           "Shareholding by quarter (NSE filings)"])
E("large_deals", "Bulk and block deals", "Market",
  "Large trades disclosed by NSE for the latest day: bulk deals (one client trading more than 0.5% of a company's "
  "shares in a day) and block deals (single trades of ₹10 crore or more in the special block window).",
  use="Shows who is buying or selling in size: funds, promoters or known investors. Repeated buying by the same client "
      "can be informative.", caveat="One day only (NSE's snapshot); a deal can be one side of a pre-arranged transfer.",
  related=["shareholding", "fii_dii"],
  aliases=["Client", "Side", "Quantity", "Price ₹", "Value ₹ Cr", "Symbol",
           "Bulk and block deals (latest day)", "Bulk and block deals in this stock (latest day)"])
E("fii_dii", "FII / DII flows", "Market",
  "Net buying (+) or selling (−) in the cash market by foreign portfolio investors (FII/FPI) and domestic institutions "
  "(mutual funds, insurers; DII), in ₹ crore, as published by NSE each evening.",
  reading="Sustained FII selling has weighed on Indian stocks in many sell-offs, often absorbed by DII buying (SIP "
          "inflows). One day means little; look at the run of days.",
  caveat="Provisional figures. NSE serves only the latest day, so the history here starts when you turned the NSE "
         "switch on.", related=["global_markets", "large_deals"],
  aliases=["FII net", "DII net", "FII net, days seen", "DII net, days seen", "FII buy", "FII sell", "DII buy",
           "DII sell", "FII / DII cash-market flows (₹ crore)"])
E("adv_dec", "Advancers / decliners", "Market",
  "How many NIFTY 200 stocks rose vs fell today.",
  use="A quick read of participation: 150/44 is a broad rally, 51/143 broad selling, even if NIFTY moved little.",
  related=["breadth", "ret_1d"], aliases=["Adv / Dec", "ADV / DEC"])
E("dispersion", "Dispersion", "Regimes",
  "How differently stocks moved from each other (cross-sectional spread of returns).",
  use="High dispersion = stock-picking matters more; low dispersion = everything moves with the index.",
  aliases=["dispersion"], related=["regime"])
E("nifty_dd", "NIFTY drawdown", "Regimes", "How far NIFTY is below its previous peak, as a fraction (−0.135 = 13.5% below).",
  unit="pct", bands=[(-0.20, "bear market", "more than 20% off the high"), (-0.10, "correction", "10–20% off the high"),
                     (-0.03, "pullback", "a normal dip"), (INF, "near highs", "within 3% of the peak")],
  aliases=["nifty_dd"], related=["max_drawdown", "regime"])
E("nifty_ret_20", "NIFTY 20-day return", "Regimes", "NIFTY's return over the last 20 trading days (one regime input).",
  unit="pct", aliases=["nifty_ret_20"], related=["regime"])
E("nifty_vol", "NIFTY realised volatility", "Regimes",
  "Annualised standard deviation of NIFTY's daily returns over 20 or 60 days, relative to its 1-year median in the "
  "regime features (0 = normal, positive = more volatile than usual).",
  aliases=["nifty_vol_20", "nifty_vol_60"], related=["india_vix", "regime"])
E("feed", "Live feed status", "Desk",
  "Whether live prices are streaming from Yahoo Finance.",
  how="LIVE = a tick arrived in the last 90 s; STALE = none since then; MARKET CLOSED = outside 09:15–15:30 IST "
      "(prices shown are the last close); OFFLINE = connection problem (the app reconnects automatically).",
  caveat="Yahoo's stream can be delayed and does not cover every stock at every moment.",
  aliases=["Feed", "FEED", "Live"], related=["ticks"])
E("ticks", "Ticks per second", "Desk", "How many live price updates per second the app is receiving.",
  aliases=["ticks/s"], related=["feed"])

# =====================================================================================================
# Valuation
# =====================================================================================================
E("market_cap", "Market capitalisation", "Valuation",
  "The stock market's total value of a company: share price × shares outstanding.",
  how="From Yahoo. Shown in ₹ crore (Cr; 1 Cr = 10 million) or lakh crore (L Cr; 1 L Cr = 1 trillion).",
  bands=[(5_000, "small cap", "under ₹5,000 Cr: higher growth potential, more volatile, less liquid"),
         (20_000, "mid cap", "₹5,000–20,000 Cr"),
         (INF, "large cap", "over ₹20,000 Cr: SEBI's large caps are the top 100 companies by size")],
  use="Size drives liquidity and risk: large caps move less and are easier to trade in size.",
  caveat="Bands here are rough; SEBI defines caps by rank (top 100 = large, 101–250 = mid), not fixed amounts.",
  aliases=["Market cap", "Market cap ₹Cr", "marketCap"], related=["ev_ebitda"])
E("pe", "P/E ratio (price to earnings)", "Valuation",
  "Share price ÷ earnings per share: how many rupees investors pay for ₹1 of annual profit. TTM = trailing 12 months; "
  "forward = analysts' estimate of next year's earnings.",
  how="From Yahoo (trailingPE / forwardPE). The Explain panel also shows the median P/E of the stock's NSE sector.",
  bands=[(0, "negative", "the company is loss-making; P/E is not meaningful"),
         (12, "cheap", "low expectations: value, a cyclical at peak earnings, or a business in trouble"),
         (25, "moderate", "roughly in line with the broad Indian market (NIFTY's long-run P/E is ~20–24)"),
         (45, "expensive", "investors expect strong growth"),
         (INF, "very expensive", "priced for a lot of future growth; disappointments get punished")],
  use="Compare within a sector: banks normally trade far cheaper than FMCG or IT. A forward P/E below the trailing "
      "P/E means analysts expect earnings to grow.",
  caveat="One-off gains or losses distort earnings; cyclicals look cheapest exactly when profits are peaking.",
  related=["pb", "eps", "peg", "roe"], aliases=["P/E (ttm)", "P/E (fwd)", "trailingPE", "forwardPE", "P/E"])
E("pb", "P/B ratio (price to book)", "Valuation",
  "Share price ÷ book value per share (net assets per share).",
  bands=[(1, "below book", "market values the company below its accounting net worth: distress or deep value"),
         (3, "moderate", "typical for banks and asset-heavy businesses"),
         (8, "high", "asset-light, high-return businesses (IT, FMCG) usually trade here"),
         (INF, "very high", "the value is in brands, IP or growth, not in assets")],
  use="Most useful for banks and financials, where book value is close to real value. P/B ÷ P/E = ROE.",
  related=["pe", "roe"], aliases=["P/B", "priceToBook"])
E("ev_ebitda", "EV / EBITDA", "Valuation",
  "Enterprise value (market cap + debt − cash) ÷ earnings before interest, tax, depreciation and amortisation.",
  bands=[(8, "cheap", ""), (15, "moderate", ""), (25, "expensive", ""), (INF, "very expensive", "")],
  use="Compares companies with different debt levels fairly, which P/E cannot. Less useful for banks.",
  related=["pe", "debt_equity"], aliases=["EV/EBITDA", "enterpriseToEbitda"])
E("eps", "EPS (earnings per share)", "Valuation",
  "Net profit attributable to shareholders ÷ shares outstanding, over the trailing 12 months, in ₹.",
  use="Growing EPS over years is what ultimately drives share prices; compare it with the price via P/E.",
  related=["pe", "earnings_growth"], aliases=["EPS (ttm)", "EPS (ttm) ₹", "trailingEps", "Diluted EPS (₹)"])
E("peg", "PEG ratio", "Valuation", "P/E ÷ expected earnings growth (in %). Around 1 is 'fair' by convention.",
  bands=[(1, "growth looks cheap", ""), (2, "fair to full", ""), (INF, "expensive for its growth", "")],
  related=["pe", "earnings_growth"], aliases=["PEG"])
E("dividend_yield", "Dividend yield", "Valuation",
  "Annual dividends per share ÷ share price.",
  bands=[(0.005, "minimal", "the company reinvests most profit"), (0.02, "modest", ""),
         (0.04, "good income", "above the market average (~1.2–1.5% for NIFTY)"),
         (INF, "high", "check it is sustainable: a falling price inflates the yield")],
  unit="pct", use="Income investing; PSUs and mature companies tend to pay more.",
  caveat="Dividends are taxed at your slab rate in India.", related=["payout"], aliases=["Dividend yield", "dividendYield"])
E("payout", "Payout ratio", "Valuation", "Share of profit paid out as dividends.",
  bands=[(0.3, "low", "most profit reinvested"), (0.6, "balanced", ""),
         (0.9, "high", "little room to grow the dividend"), (INF, "above 90%", "paying more than it earns: may not last")],
  unit="pct", related=["dividend_yield"], aliases=["Payout ratio", "payoutRatio"])
E("week52", "52-week position", "Valuation",
  "Where today's price sits in its 52-week range: 0% = at the 1-year low, 100% = at the 1-year high.",
  bands=[(0.2, "near the low", "weak stock or potential value"), (0.8, "middle", ""),
         (INF, "near the high", "strong trend; momentum investors like this zone")],
  unit="pct", related=["from_52w_high", "from_52w_low"], aliases=["52w position"])

# =====================================================================================================
# Quality & growth
# =====================================================================================================
E("roe", "ROE (return on equity)", "Quality & growth",
  "Net profit ÷ shareholders' equity: how much profit the company makes on the owners' money.",
  how="Yahoo's figure where available (only ~1 in 6 NSE stocks). Otherwise derived as (P/B) ÷ (P/E) = EPS ÷ book "
      "value per share, only when both are positive; this matches Yahoo's own figure closely where both exist "
      "(correlation 0.99 in this app's data).",
  bands=[(0, "negative", "loss-making"), (0.10, "weak", "below the cost of equity in India (~12–14%)"),
         (0.15, "adequate", ""), (0.25, "good", "a quality business"),
         (INF, "excellent", "very high returns on capital: strong brand, moat or asset-light model")],
  unit="pct", use="Sustained high ROE with low debt is a classic sign of a quality company.",
  caveat="High debt can inflate ROE; check debt/equity alongside it. Banks run on leverage, so compare banks with banks.",
  related=["roa", "debt_equity", "pb", "pe"], aliases=["ROE", "returnOnEquity"])
E("roa", "ROA (return on assets)", "Quality & growth", "Net profit ÷ total assets.",
  bands=[(0.01, "low", "normal for banks (~1–2%)"), (0.05, "moderate", ""), (INF, "high", "")],
  unit="pct", related=["roe"], aliases=["ROA", "returnOnAssets"])
E("operating_margin", "Operating margin", "Quality & growth", "Operating profit ÷ revenue, before interest and tax.",
  bands=[(0.05, "thin", "commodity-like business, little pricing power"), (0.15, "moderate", ""),
         (0.25, "strong", ""), (INF, "very strong", "pricing power (software, consumer brands)")],
  unit="pct", related=["net_margin"], aliases=["Operating margin", "operatingMargins"])
E("net_margin", "Net (profit) margin", "Quality & growth", "Net profit ÷ revenue.",
  bands=[(0, "loss", ""), (0.05, "thin", ""), (0.15, "healthy", ""), (INF, "high", "")],
  unit="pct", related=["operating_margin", "roe"], aliases=["Net margin", "profitMargins"])
E("debt_equity", "Debt / equity", "Quality & growth",
  "Total debt ÷ shareholders' equity (shown as a multiple: 0.5x = debt is half of equity).",
  bands=[(0.3, "low debt", "conservative balance sheet"), (1.0, "moderate", ""),
         (2.0, "high", "interest costs bite in downturns"), (INF, "very high", "risky outside banks/NBFCs")],
  use="Lower is safer, especially when rates rise.",
  caveat="Banks and NBFCs borrow by design; this ratio is not comparable for them.",
  related=["roe", "ev_ebitda"], aliases=["Debt / equity", "Debt / equity (x)", "debtToEquity"])
E("revenue_growth", "Revenue growth", "Quality & growth",
  "Year-on-year change in quarterly revenue (latest quarter vs the same quarter last year).",
  bands=[(0, "shrinking", ""), (0.08, "slow", "around inflation"), (0.20, "healthy", ""), (INF, "fast", "")],
  unit="pct", related=["earnings_growth"], aliases=["Revenue growth", "Revenue growth (yoy)", "revenueGrowth"])
E("earnings_growth", "Earnings growth", "Quality & growth",
  "Year-on-year change in quarterly earnings.",
  bands=[(0, "falling profits", ""), (0.10, "slow", ""), (0.25, "healthy", ""), (INF, "fast", "")],
  unit="pct", caveat="Quarterly profits swing with one-offs; look at several quarters.",
  related=["revenue_growth", "eps"], aliases=["Earnings growth", "Earnings growth (yoy)", "earningsGrowth"])
E("promoters", "Promoter / insider holding", "Quality & growth",
  "Share of the company owned by its promoters (founders, parent company) and insiders.",
  bands=[(0.3, "low", "widely held or professionally run"), (0.6, "typical", ""),
         (INF, "high", "promoter-controlled; check for pledged shares")],
  unit="pct", use="Rising promoter holding is often read as confidence; falling as a warning.",
  related=["institutions"], aliases=["Promoters", "Promoters / insiders", "heldPercentInsiders"])
E("institutions", "Institutional holding", "Quality & growth",
  "Share owned by institutions: mutual funds, insurers, foreign portfolio investors.",
  unit="pct", related=["promoters"], aliases=["Institutions", "heldPercentInstitutions"])

# =====================================================================================================
# Technicals
# =====================================================================================================
E("rsi", "RSI (14-day)", "Technicals",
  "Relative Strength Index: momentum on a 0–100 scale comparing the size of recent up-moves with down-moves.",
  how="Wilder's RSI over 14 trading days on adjusted closes.",
  bands=[(30, "oversold", "heavy recent selling; bounces are common but not guaranteed"),
         (45, "weak", "weak momentum, approaching oversold"),
         (55, "neutral", "no strong momentum either way"),
         (70, "healthy uptrend", "the monitor list counts 55–70 as a bullish setup"),
         (INF, "overbought", "stretched after a strong run; pullbacks become more likely")],
  use="Timing entries within a trend and spotting exhaustion. 'Oversold quality' in the screener combines RSI<35 "
      "with profitability.",
  caveat="In strong trends RSI can stay overbought or oversold for weeks.",
  related=["macd", "supertrend", "from_52w_high", "bb_pctb"], aliases=["RSI 14", "RSI", "rsi14", "RSI(14)"])
E("macd", "MACD histogram", "Technicals",
  "Moving Average Convergence Divergence: the 12-day EMA minus the 26-day EMA (the MACD line), minus its own 9-day "
  "average (the signal line). The histogram is that difference.",
  bands=[(0, "bearish momentum", "MACD below its signal: momentum is fading or negative"),
         (INF, "bullish momentum", "MACD above its signal: momentum is building")],
  use="Crossings of zero mark momentum turning; widening bars mean the move is accelerating.",
  caveat="Lags price; gives many false signals in sideways markets.", related=["rsi", "sma"],
  aliases=["MACD hist", "MACD"])
E("adx", "ADX (14-day)", "Technicals",
  "Average Directional Index: how strong a trend is, regardless of direction (0–100).",
  bands=[(20, "no trend", "range-bound; trend-following signals fail more often"),
         (25, "weak trend", ""), (40, "strong trend", "trend-following works best here"),
         (INF, "very strong trend", "often late in a move")],
  use="Decides which tools to trust: trend tools when ADX is high, range tools (RSI, Bollinger) when it is low.",
  related=["supertrend", "vs_sma"], aliases=["ADX 14", "ADX", "adx14"])
E("supertrend", "Supertrend (10, 3)", "Technicals",
  "A trailing trend line placed 3 × ATR(10) away from price. +1 = price above the line (uptrend), −1 = below (downtrend).",
  how="Computed in the C++ core; matches the pandas reference exactly.",
  bands=[(0, "downtrend", "price is below the Supertrend line"), (INF, "uptrend", "price is above the line")],
  use="A simple trend filter and trailing stop level.", caveat="Whipsaws in choppy markets.",
  related=["adx", "atr_pct"], aliases=["Supertrend (+1 up)", "Supertrend"])
E("sma", "Moving averages (SMA / EMA)", "Technicals",
  "The average closing price over the last N days (simple, SMA) or weighted to recent days (exponential, EMA). "
  "The 20-day tracks the short-term trend, 50-day the medium term, 200-day the long term.",
  use="Price above a rising 200-day average is the most widely used definition of a long-term uptrend.",
  related=["vs_sma", "golden_cross", "breadth"], aliases=["SMA50", "SMA200", "SMA50 / SMA200", "EMA"])
E("vs_sma", "Distance from moving average", "Technicals",
  "How far the price is above (+) or below (−) its 20-, 50- or 200-day simple moving average, in %.",
  bands=[(-0.15, "far below", "deep downtrend or oversold"), (-0.02, "below", "under the trend line"),
         (0.02, "at the average", ""), (0.15, "above", "in an uptrend"),
         (INF, "far above", "extended; mean reversion risk")],
  unit="pct", related=["sma", "golden_cross"], aliases=["vs SMA20 %", "vs SMA50 %", "vs SMA200 %"])
E("golden_cross", "Golden cross / death cross", "Technicals",
  "Golden cross: the 50-day average crosses above the 200-day (bullish). Death cross: crosses below (bearish). "
  "The flag is 1 if it happened in the last 5 days.",
  bands=[(0.5, "no", ""), (INF, "yes", "the cross happened within the last 5 trading days")],
  caveat="Well known and slow; by the time it happens much of the move is often done.",
  related=["sma"], aliases=["Golden cross (5d)", "Death cross (5d)"])
E("breakout", "20-day breakout", "Technicals",
  "1 if today's close is above the highest close of the previous 20 days (a Donchian breakout).",
  bands=[(0.5, "no", ""), (INF, "yes", "a fresh short-term high")],
  use="A classic momentum entry signal; part of the monitor list's 'setup' score.",
  related=["from_52w_high", "vol_ratio"], aliases=["20d breakout"])
E("from_52w_high", "Distance from 52-week high", "Technicals",
  "How far the price is below its highest close of the last year (0% = at the high).",
  bands=[(-0.30, "far below", "beaten down: value or broken trend"), (-0.10, "below", ""),
         (-0.03, "near the high", ""), (INF, "at the high", "momentum investors watch new highs")],
  unit="pct", related=["new_52w", "week52"], aliases=["From 52w high %"])
E("from_52w_low", "Distance from 52-week low", "Technicals",
  "How far the price is above its lowest close of the last year (0% = at the low).", unit="pct",
  related=["from_52w_high"], aliases=["From 52w low %"])
E("new_52w", "New 52-week high / low", "Technicals", "1 if today set a new 1-year closing high (or low).",
  bands=[(0.5, "no", ""), (INF, "yes", "")], related=["from_52w_high"], aliases=["New 52w high", "New 52w low"])
E("vwap", "VWAP (volume-weighted average price)", "Technicals",
  "The average price paid today, weighting each bar's typical price (high + low + close) ÷ 3 by its volume. Resets at "
  "every session's open; drawn on intraday timeframes only.",
  reading="Above VWAP = today's buyers are, on average, in profit (intraday strength); below = sellers in control. "
          "Institutions benchmark executions against it, so price often reacts at the line.",
  use="The intraday agent's VWAP-reclaim/rejection setups use it. For a swing entry, buying well above VWAP means "
      "paying up versus everyone else today.", caveat="Meaningless across days or on daily bars.",
  related=["candle", "agent_trade"], aliases=["VWAP (intraday)", "VWAP"])
E("pivots_cpr", "Pivot points and CPR", "Technicals",
  "Floor-trader levels from the previous period's high (H), low (L) and close (C): pivot P = (H+L+C)/3, R1 = 2P−L, "
  "S1 = 2P−H, R2 = P+(H−L), S2 = P−(H−L). The central pivot range (CPR) spans BC = (H+L)/2 and TC = 2P−BC.",
  how="Previous day for intraday bars, previous week for daily bars, previous month for weekly bars.",
  reading="A narrow CPR often precedes a trending day, a wide one a range day. Price holding above P/TC = bullish "
          "bias; R1/S1 and R2/S2 are common profit-taking or reversal zones.",
  caveat="Widely watched, which is why they sometimes work; not tested for edge in this app.",
  related=["candle", "vwap"], aliases=["Pivots + CPR", "Pivots / CPR", "CPR"])
E("fibonacci", "Fibonacci retracements", "Technicals",
  "Horizontal levels at 23.6%, 38.2%, 50%, 61.8% and 78.6% of the swing between the highest high and lowest low on "
  "screen, measured back from the end of the swing.",
  how="Recomputed whenever you zoom or pan: the swing is always the visible window's.",
  reading="In an up-move, pullbacks that hold the 38.2-61.8% zone are often read as healthy; a break of 78.6% suggests "
          "the move has failed.", caveat="The ratios have no proven predictive power; treat them as a ruler, not a "
          "signal.", related=["candle", "pivots_cpr"], aliases=["Fibonacci (visible swing)", "Fibonacci"])
E("bb_pctb", "Bollinger %B", "Technicals",
  "Where the price sits inside its Bollinger Bands (20-day average ± 2 standard deviations): 0 = lower band, "
  "1 = upper band.",
  bands=[(0, "below lower band", "unusually weak vs recent range"), (0.2, "near lower band", ""),
         (0.8, "inside the bands", ""), (1, "near upper band", ""),
         (INF, "above upper band", "unusually strong vs recent range")],
  related=["bb_width", "rsi"], aliases=["Bollinger %B"])
E("bb_width", "Bollinger width", "Technicals",
  "Width of the Bollinger Bands as a % of price: how volatile the last 20 days were.",
  unit="pct", use="Very narrow bands (a 'squeeze') often come before a big move.", related=["bb_pctb", "atr_pct"],
  aliases=["Bollinger width"])
E("atr_pct", "ATR %", "Technicals",
  "Average True Range over 14 days as a % of price: the typical daily trading range.",
  bands=[(0.015, "quiet", "small daily moves"), (0.03, "normal", ""), (0.05, "volatile", ""),
         (INF, "very volatile", "wide stops needed")],
  unit="pct", use="Set stop-losses and position sizes in multiples of ATR rather than fixed percentages.",
  related=["vol_60d", "supertrend"], aliases=["ATR %"])
E("vol_60d", "Volatility (60 days)", "Technicals",
  "Annualised standard deviation of daily returns over the last 60 trading days.",
  bands=[(0.20, "low", "steady large caps"), (0.35, "moderate", "typical NIFTY 200 stock"),
         (0.55, "high", ""), (INF, "very high", "")],
  unit="pct", use="Risk sizing: a stock with 40% volatility swings about twice as much as one with 20%.",
  related=["atr_pct", "vol_lstm", "beta"], aliases=["Volatility 60d"])
E("beta", "Beta (1 year)", "Technicals",
  "How much a stock tends to move when NIFTY moves 1%. Beta 1.3 → typically 1.3%.",
  how="Rolling 252-day covariance with NIFTY ÷ NIFTY's variance (the stock page's Beta tile is Yahoo's figure).",
  bands=[(0.7, "defensive", "moves less than the market"), (1.2, "market-like", ""),
         (INF, "aggressive", "amplifies market moves both ways")],
  related=["vol_60d", "alpha"], aliases=["Beta 1Y", "Beta"])
E("max_dd_1y", "Max drawdown (1 year)", "Technicals",
  "The worst peak-to-trough fall in the stock's price over the last year.",
  unit="pct", bands=[(-0.40, "severe", ""), (-0.20, "large", ""), (-0.10, "moderate", ""), (INF, "mild", "")],
  related=["max_drawdown"], aliases=["Max DD 1Y %"])
E("vol_ratio", "Volume / 20-day average", "Technicals",
  "Today's traded volume ÷ the average of the last 20 days.",
  bands=[(0.7, "quiet", "below-normal interest"), (1.5, "normal", ""),
         (3, "unusual", "1.5× or more: the monitor list flags this as unusual activity"),
         (INF, "very unusual", "news, results or a block deal is likely")],
  use="Price moves on heavy volume are taken more seriously than moves on light volume.",
  related=["breakout", "monitor_score"], aliases=["Volume / 20d avg"])
E("volume", "Volume", "Technicals", "Number of shares traded today (or in the latest session); 'avg' is the "
  "recent daily average.", use="Compare with the average: the ratio is what matters (see Volume / 20-day average).",
  related=["vol_ratio", "turnover_cr"], aliases=["Volume"])
E("turnover_cr", "Average turnover", "Technicals", "Average daily traded value over 20 days, in ₹ crore.",
  use="Liquidity: higher turnover means you can buy or sell without moving the price much.",
  aliases=["Avg turnover ₹Cr"])
E("signal_text", "Technical signal (text)", "Technicals",
  "A plain-language reading of the indicator's value using the conventional ranges on this page.",
  aliases=["Signal"])
E("candle", "Price (candlestick) chart", "Charts",
  "Each candle is one bar of the chosen timeframe (1 minute … 1 week): the body spans open → close (green = closed "
  "higher, red = lower), the thin wick spans the bar's high and low. Overlays and panes are chosen in Indicators: "
  "SMA 50/200 (dashed blue/purple), EMA 9/21, VWAP, Bollinger bands, Supertrend, pivots + CPR, Fibonacci; volume "
  "(green/red by candle), RSI(14) and MACD panes. Dashed horizontal lines show the swing trade plan (entry, stop, T1, "
  "T2).",
  how="1D/1W: the adjusted daily panel (weekly bars resampled from it). 1m (last 7 days), 5m/15m (60 days) and 1h "
      "(2 years): Yahoo intraday bars in IST, cached for a minute or more; they can be delayed or have gaps.",
  reading="Rising candles above a rising 200-day line = long-term uptrend. Price crossing the 50-day line with high "
          "volume = a trend change worth noticing. RSI near 70 = stretched, near 30 = washed out. Compare overlays "
          "NIFTY or the sector rebased to the first visible bar: the stock above the line = beating it in this "
          "window. Log scale makes equal percentage moves equal heights (use it for long histories). Use the range "
          "buttons, the mouse wheel to zoom, drag to pan, double-click to reset; the readout shows values under the "
          "cursor.",
  use="See trend, momentum and volume at a glance before acting on any number in the tables, and check a trade "
      "plan's stop and targets against recent swings.",
  caveat="Indicators describe the past; none of them has a measured edge in this app on its own (see Backtest lab).",
  related=["sma", "rsi", "vol_ratio", "vwap", "pivots_cpr", "fibonacci", "supertrend", "macd", "trade_plan"],
  aliases=["chart_price", "daily (adjusted)", "weekly (adjusted)", "hourly", "5-minute", "1-minute", "15-minute"])
E("finbert", "FinBERT sentiment score", "News & analysts",
  "A deep-learning score of how positive or negative a headline is, from −1 (very negative) to +1 (very positive).",
  how="FinBERT (ProsusAI/finbert), a BERT language model fine-tuned on financial text, gives each headline "
      "probabilities of positive / neutral / negative; score = P(positive) − P(negative). A stock's score is the "
      "average of its last-48-hour headlines from Google News, weighted towards recent ones (12-hour half-life).",
  bands=[(-0.5, "very negative", "clearly bad news (losses, probes, downgrades)"),
         (-0.2, "negative", ""), (0.2, "neutral", "mostly factual or mixed headlines"),
         (0.5, "positive", ""), (INF, "very positive", "clearly good news (beats, orders, upgrades)")],
  use="Explains price moves and flags stocks in the news; feeds 15% of the monitor-list score.",
  caveat="Headline tone, not fundamental truth. Its predictive value is still being measured as news history "
         "accumulates, so it is an overlay, not part of the backtests.",
  related=["sentiment_adj", "buzz", "n_news"],
  aliases=["News sentiment", "FinBERT sentiment", "NEWS SENTIMENT"])
E("sentiment_adj", "Sentiment (confidence-adjusted)", "News & analysts",
  "The FinBERT sentiment shrunk towards zero when there are few headlines: sentiment × n ÷ (n + 3).",
  how="One glowing headline gives +0.25 at most; ten consistent headlines keep most of their score.",
  bands=[(-0.2, "negative", ""), (0.2, "neutral", ""), (INF, "positive", "")],
  use="Fairer comparison across stocks with very different news coverage. The screener's 'Positive news buzz' "
      "preset uses ≥ 0.2 with at least 3 headlines.",
  related=["finbert", "n_news"], aliases=["Sentiment (adj.)", "Sentiment"])
E("n_news", "Headlines (48 h)", "News & analysts",
  "Number of relevant Google News headlines about the stock in the last 48 hours (de-duplicated, junk filtered, "
  "whole-word name match).",
  related=["buzz", "finbert"], aliases=["Headlines (48h)", "Headlines", "Bullish", "Bearish"])
E("buzz", "News buzz", "News & analysts",
  "Today's headline count ÷ the stock's average count over previous daily runs.",
  bands=[(0.7, "quiet", ""), (1.5, "normal", ""), (INF, "buzzing", "much more coverage than usual")],
  caveat="Needs several days of history to be meaningful; it builds up as the daily pipeline runs.",
  related=["n_news"], aliases=["News buzz"])
E("news_label", "Headline label", "News & analysts", "FinBERT's most likely class for the headline: positive, "
  "neutral or negative.", aliases=["Label"], related=["finbert"])
E("analyst_score", "Analyst score (0–100)", "News & analysts",
  "A single score summarising what brokerage analysts think, built by this app from Yahoo Finance data.",
  how="40% consensus rating (1 strong buy … 5 sell, rescaled) + 30% upside to the mean price target (−20%…+40% "
      "mapped to 0–100) + 20% change in the share of buy ratings over 3 months + 10% coverage (25+ analysts = full). "
      "Missing parts are skipped and weights renormalised; no score without analyst coverage.",
  bands=[(30, "negative", "analysts are cautious or cutting"), (50, "lukewarm", ""), (70, "positive", ""),
         (INF, "strongly positive", "broad buy ratings, upside and upgrades")],
  use="A quick read of the professional consensus.",
  caveat="Third-party opinion, often slow and optimistic on average. Shown for context, not endorsed.",
  related=["target_upside", "n_analysts"], aliases=["Analyst score", "Analyst", "Analyst score (0-100)", "ANALYST SCORE"])
E("target_upside", "Target upside", "News & analysts", "Analysts' mean price target ÷ current price − 1.",
  unit="pct", bands=[(0, "below price", "analysts see downside"), (0.15, "modest", ""), (INF, "large", "")],
  caveat="Targets lag prices and cluster around round numbers.", related=["analyst_score"], aliases=["Target upside"])
E("n_analysts", "Number of analysts", "News & analysts", "How many analysts cover the stock on Yahoo.",
  aliases=["# analysts"], related=["analyst_score"])
E("next_results", "Next results date", "News & analysts",
  "The next scheduled quarterly results date (Yahoo calendar). 'Results in' = days until then.",
  use="Prices often move sharply on results day; the watchlist alerts you within 7 days.",
  related=["alerts"], aliases=["Next results", "Results in", "Days to results", "Results date", "Results this week"])

# =====================================================================================================
# Model outputs
# =====================================================================================================
E("p_up", "Next-day P(up)", "Model outputs",
  "The calibrated probability that the stock beats tomorrow's median NIFTY 200 return (not simply 'goes up').",
  how="Ensemble of LightGBM and SeqNet, calibrated with isotonic regression on a validation year, trained "
      "walk-forward on 2015 → today.",
  bands=[(0.47, "lean weak", "the model expects it to lag the median tomorrow"),
         (0.53, "no real view", "close to a coin flip, which is most stocks most days"),
         (INF, "lean strong", "the model expects it to beat the median tomorrow")],
  use="Ranks what to watch tomorrow; drives 60% of the monitor-list score.",
  caveat="The edge is small: next-day AUC 0.526. Over 2,900 days the top-10 picks beat the median 54% of the time but "
         "lost money after ~0.25% round-trip costs. Use it to decide what to look at, not what to buy.",
  related=["nextday_model", "monitor_score", "auc", "calibration"],
  aliases=["Next-day P(up)", "P(up)", "P(up) 1D", "prob_up", "NEXT-DAY P(UP)"], results=("nextday_summary.csv", "ensemble"))
E("calibration", "Calibration", "Model quality",
  "A model is calibrated when its probabilities match reality: of all stocks given P(up) = 55%, about 55% should "
  "actually beat the median.",
  how="Next-day probabilities are calibrated with isotonic regression fit on a held-out validation year.",
  related=["p_up"], aliases=["calibrated"])
E("monitor_score", "Monitor score", "Model outputs",
  "How strongly a stock deserves attention tomorrow, from −1 (watch for weakness) to +1 (watch for strength).",
  how="0.60 × model (P(up) rescaled) + 0.15 × news (adjusted sentiment) + 0.15 × activity (volume vs normal) + "
      "0.10 × setup (share of 5 bullish setups: 20-day breakout, new 52-week high, golden cross, Supertrend up, "
      "RSI 55–70). Missing parts are skipped.",
  bands=[(-0.2, "watch for weakness", ""), (0.2, "low priority", ""), (INF, "watch for strength", "")],
  use="A short, explained list of where to point your attention at the open; the 'Why' column says what triggered it.",
  caveat="A watch list, not buy/sell signals.", related=["p_up", "finbert", "vol_ratio", "breakout"],
  aliases=["Score", "Watch for strength", "Watch for weakness", "Top stocks to monitor tomorrow", "Why"])
E("ranker_pct", "Ranker percentile", "Model outputs",
  "Where a model ranks the stock among all eligible NIFTY 200 stocks for next month: 100% = its top pick, "
  "50% = middle, 0% = bottom.",
  how="Each monthly ranker scores every stock at month-end; the column is that score's percentile rank.",
  bands=[(0.2, "bottom fifth", "the model expects it to lag next month"), (0.8, "middle", "no strong view"),
         (INF, "top fifth", "the model expects it to beat the median next month")],
  use="Monthly stock selection: the backtests buy the top 20 (or top decile) by this rank.",
  caveat="Monthly-horizon research output; edges are small and survivorship-biased. Not a recommendation.",
  related=["consensus", "ffnn", "lstm", "transformer"], aliases=["Monthly ranker %ile", "monthly_pct"])
E("consensus", "Consensus", "Model outputs",
  "The average percentile across all monthly ranking models; the Rankers page sorts by it.",
  bands=[(0.2, "models agree it's weak", ""), (0.8, "mixed / middle", ""), (INF, "models agree it's strong", "")],
  use="Combining models is usually more reliable than trusting any single one.",
  related=["ranker_pct", "agreement"], aliases=["Consensus", "Top consensus", "Bottom consensus"])
E("agreement", "Agreement", "Model outputs",
  "How much the models agree about the stock: 100% = identical percentiles, lower = they disagree.",
  how="1 − 2 × the standard deviation of the models' percentiles.",
  bands=[(0.5, "models disagree", "treat the consensus with caution"), (0.8, "partial agreement", ""),
         (INF, "strong agreement", "the consensus is more trustworthy")],
  related=["consensus"], aliases=["Agreement"])
E("rank", "Rank", "Model outputs", "Position by consensus percentile: 1 = the models' favourite for next month.",
  related=["consensus"], aliases=["Rank"])
E("asof", "Prediction date", "Model outputs",
  "The month-end date of the features the monthly rankers used; they predict the following month.",
  aliases=["Predictions as of", "Predicting", "Models"], related=["ranker_pct"])

# =====================================================================================================
# Model quality
# =====================================================================================================
E("auc", "AUC (area under the ROC curve)", "Model quality",
  "The chance that the model ranks a random winner above a random loser. 0.50 = coin flip, 1.00 = perfect.",
  bands=[(0.50, "worse than chance", ""), (0.51, "no skill", ""),
         (0.53, "small real edge", "typical of genuine stock-return models; this app's models are here"),
         (0.56, "strong for finance", "rare out of sample"),
         (INF, "suspicious", "check for look-ahead bias before believing it")],
  use="A threshold-free measure of ranking skill.",
  caveat="Small AUCs can still be valuable over thousands of predictions, or worthless after costs.",
  related=["ic", "accuracy"], aliases=["AUC"])
E("accuracy", "Accuracy", "Model quality",
  "Share of predictions on the correct side (beat / didn't beat the median).",
  unit="pct", bands=[(0.5, "no skill", ""), (0.54, "small edge", "51–54% is a good, honest result for next-day "
                                                                    "direction"), (INF, "unusually high", "")],
  caveat="Says nothing about how much the right calls earned.", related=["auc"], aliases=["Accuracy"])
E("ic", "IC (information coefficient)", "Model quality",
  "The Spearman rank correlation between the model's scores and the stocks' actual next-period returns, averaged over "
  "periods.",
  bands=[(0, "negative", "rankings point the wrong way"), (0.02, "weak", ""),
         (0.05, "useful", "practitioners treat IC ≈ 0.03–0.05 as a real signal"),
         (INF, "strong", "rare out of sample")],
  use="The standard measure of a ranking signal's quality.", related=["ic_tstat", "ic_hit", "auc"],
  aliases=["IC mean", "IC", "Mean monthly IC by year"])
E("ic_tstat", "t-statistic", "Model quality",
  "Average ÷ (its standard error): how many standard errors the result is from zero.",
  bands=[(2, "could be luck", "below ~2, a positive average may be chance"),
         (3, "significant", "unlikely to be luck"), (INF, "highly significant", "")],
  caveat="Significance is not size: a tiny edge can be very significant over thousands of days and still lose after "
         "costs.", related=["ic"], aliases=["IC t-stat", "Spread t-stat", "IC t", "t-stat"])
E("ic_hit", "IC hit rate", "Model quality", "Share of months in which the IC was positive.", unit="pct",
  bands=[(0.5, "inconsistent", ""), (0.6, "mostly positive", ""), (INF, "very consistent", "")],
  related=["ic"], aliases=["IC hit rate"])
E("decile", "Decile returns and spread", "Model quality",
  "Stocks are split into 10 equal groups (deciles) by model score each month. Top decile = the 10% the model likes "
  "most. Spread = top-decile minus bottom-decile average return.",
  bands=[(0, "negative spread", "the model's favourites did worse"), (0.005, "small", ""),
         (INF, "meaningful", "above ~0.5%/month before costs")],
  unit="pct", use="Shows whether ranking translates into returns.",
  caveat="Before costs and not directly investable (the bottom decile would need shorting).",
  related=["ic"], aliases=["Top decile / mo", "Bottom decile / mo", "Spread / mo"])
E("top10_hit", "Top-10 hit rate", "Model quality",
  "Of the 10 stocks the next-day model liked most each day, the share that beat the median the next day.",
  unit="pct", bands=[(0.5, "no edge", ""), (0.53, "small edge", ""), (INF, "useful edge", "")],
  related=["top10_net"], aliases=["Top10 hit rate"])
E("top10_net", "Top-10 excess return (gross and net of costs)", "Model quality",
  "Average next-day return of the top-10 picks minus the median, per day; 'net' subtracts ~0.25% round-trip costs.",
  unit="pct", bands=[(0, "loses money", "after costs, trading it daily would lose"), (INF, "makes money", "")],
  use="The honest test: an edge that disappears after costs is for watching, not trading.",
  related=["top10_hit", "cost_drag"], aliases=["Top10 excess / day", "Top10 net of costs / day"])
E("days_n", "Days / months / N", "Model quality", "How many out-of-sample periods or observations the result is based on. "
  "More periods = more trustworthy averages.", aliases=["Days", "N", "Months", "weeks", "n"])

# =====================================================================================================
# Models
# =====================================================================================================
_RANK_INPUTS = ("Inputs (each month, for every stock, z-scored across stocks): 12 cumulative monthly returns "
                "(t−13 → t−2, skipping the latest month), 20 cumulative daily returns (last 20 days) and a January "
                "dummy: 33 features (Takeuchi & Lee, 2013). Output: the probability the stock beats next month's "
                "cross-sectional median. Validation: rolling walk-forward: each year is predicted by a model trained "
                "only on the previous 8 years.")
E("ffnn", "FFNN ranker (Takeuchi & Lee)", "Models",
  "A feed-forward neural network, 33 → 40 → 4 → 50 → 1, with a 4-unit 'bottleneck' that forces it to compress past "
  "returns into a few momentum-like factors.",
  how=_RANK_INPUTS,
  use="Gives a monthly ranking of all NIFTY 200 stocks (Rankers page, Backtest lab signal, stock page 'NiveshRL models' "
      "tab). Point-in-time 2014-2026: IC 0.043 (t 3.8), top-20 strategy 16.3%/yr after costs vs 16.7% equal weight.",
  caveat="On this universe none of the deep rankers beat plain 12-1 momentum after costs; all results carry "
         "survivorship bias (today's index members applied to the past).",
  related=["lstm", "transformer", "momentum_model", "ranker_pct", "ic"], aliases=["FFNN", "FFNN ranker"],
  results=("rankers_summary.csv", "ffnn"))
E("lstm", "LSTM ranker", "Models",
  "A Long Short-Term Memory recurrent network that reads the last 13 monthly returns as a sequence, plus the daily and "
  "January context.",
  how=_RANK_INPUTS,
  use="A sequence-model view of momentum; one of the votes in the Consensus column.",
  caveat="The weakest deep ranker here (IC t-stat 3.3) and the most turnover-heavy as a strategy. A deeper model is not "
         "automatically better.",
  related=["ffnn", "transformer"], aliases=["LSTM", "LSTM ranker"], results=("rankers_summary.csv", "lstm"))
E("transformer", "Transformer ranker", "Models",
  "A 2-layer Transformer (the attention architecture behind modern language models) over the same return sequence.",
  how=_RANK_INPUTS,
  use="A consistent monthly ranking signal: on point-in-time members 2014-2026 IC 0.043 (t 3.7), beating 97% of "
      "random portfolios with the same rules. But as a top-20 monthly strategy it earned 16.1%/yr after costs vs 16.7% "
      "for equal weight (34.4% on today's members: survivorship bias). Quarterly rebalancing cuts costs ~1.6%/yr.",
  caveat="Trades a lot monthly; slow to train (≈75 min walk-forward on CPU).",
  related=["ffnn", "lstm", "consensus"], aliases=["TRANSFORMER", "Transformer ranker"],
  results=("rankers_summary.csv", "transformer"))
E("logreg", "Logistic regression (baseline)", "Models",
  "A linear model on the same 33 inputs. The simplest learned baseline.",
  use="If a deep model can't beat this, the depth isn't helping. Here it is the weakest ranker (IC t-stat 1.8).",
  related=["ffnn"], aliases=["LOGREG", "Logistic regression", "logreg"], results=("rankers_summary.csv", "logreg"))
E("momentum_model", "Momentum 12-1 (baseline)", "Models",
  "No learning: rank stocks by their return from 12 months ago to 1 month ago (skipping the latest month).",
  use="The bar every model must clear, and still the best monthly strategy on honest data: 19.3%/yr, Sharpe 0.64 for "
      "the top 20 on point-in-time members 2014-2026 (39.8% on today's members: survivorship bias flatters it most).",
  related=["ret_1m", "transformer"], aliases=["MOMENTUM", "Momentum 12-1", "Momentum top-10", "momentum"],
  results=("rankers_summary.csv", "momentum"))
E("nextday_model", "Next-day model (LightGBM + SeqNet ensemble)", "Models",
  "Predicts whether each stock beats tomorrow's cross-sectional median, from 41 causal features: 31 per stock (returns "
  "1–20 days, gap, intraday range, RSI, MACD, ATR, ADX, %B, distance to averages and 52-week high, volume z-score, "
  "relative strength…, ranked across stocks each day) and 10 market features (VIX level and change, breadth, regime, "
  "day of week…).",
  how="LightGBM (gradient-boosted trees) plus SeqNet (a 1-D CNN + Transformer over 60-day feature windows), each "
      "isotonic-calibrated, averaged. Walk-forward by year on a 5-year rolling window with a purged gap.",
  use="Feeds P(up) and the 'Watch tomorrow' list on Today. Measured 2015 → Oct 2026 over 2,900 days.",
  caveat="Real but small edge; negative after realistic daily trading costs.",
  related=["p_up", "lightgbm", "seqnet", "monitor_score"], aliases=["Next-day model", "Ensemble (LightGBM + SeqNet)"],
  results=("nextday_summary.csv", "ensemble"))
E("lightgbm", "LightGBM", "Models",
  "Gradient-boosted decision trees: many small trees, each correcting the previous ones' errors. A strong standard "
  "for tabular data.", related=["nextday_model"], aliases=["LightGBM", "lgbm"], results=("nextday_summary.csv", "lgbm"))
E("seqnet", "SeqNet (CNN + Transformer)", "Models",
  "The deep-learning half of the next-day ensemble: a 1-D convolution over the last 60 days of each stock's features "
  "followed by a Transformer encoder.",
  caveat="Slightly weaker than LightGBM alone here, but adds diversity to the ensemble.",
  related=["nextday_model", "transformer"], aliases=["SeqNet (CNN + Transformer)", "seq"],
  results=("nextday_summary.csv", "seq"))
E("reversal", "Short-term reversal (baseline)", "Models",
  "No learning: bet that yesterday's losers beat yesterday's winners tomorrow.",
  use="The next-day baseline; the learned models beat it clearly (AUC 0.526 vs 0.511).",
  related=["nextday_model"], aliases=["Short-term reversal", "reversal"], results=("nextday_summary.csv", "reversal"))
E("vol_lstm", "Volatility LSTM", "Volatility forecaster",
  "Forecasts each stock's realised volatility over the next 21 trading days from its last 60 daily returns plus "
  "realised-vol, VIX and market-volatility context. One model pooled across all stocks, retrained yearly on a "
  "rolling 8-year window.",
  use="The most accurate forecaster here (lowest RMSE and QLIKE). Use it to size positions and set stop distances: "
      "a stock forecast at 40% vol deserves about half the position of one at 20%.",
  related=["garch", "ewma", "rmse_logvol", "qlike"], aliases=["LSTM (deep learning)", "Volatility LSTM"],
  results=("vol_summary.csv", "lstm"))
E("garch", "GARCH(1,1)", "Volatility forecaster",
  "The classic econometric volatility model: tomorrow's variance = constant + a share of today's squared return + a "
  "share of today's variance. Refit yearly per stock.",
  use="The textbook benchmark. Here it over-forecasts (bias 1.18) and trails both EWMA and the LSTM.",
  related=["vol_lstm", "ewma"], aliases=["GARCH(1,1)", "garch"], results=("vol_summary.csv", "garch"))
E("ewma", "EWMA (λ = 0.94)", "Volatility forecaster",
  "Exponentially weighted volatility (RiskMetrics): recent days count more, decaying by 6% per day.",
  use="A strong simple baseline; the LSTM beats it on RMSE but not on correlation.",
  related=["vol_lstm"], aliases=["EWMA λ=0.94", "ewma"], results=("vol_summary.csv", "ewma"))
E("hist_vol", "Historical (21-day) volatility", "Volatility forecaster",
  "Naive forecast: next month's volatility = last month's.", related=["ewma"], aliases=["Historical 21d", "hist"],
  results=("vol_summary.csv", "hist"))
E("regime_model", "Regime detector (autoencoder + k-means)", "Regimes",
  "An autoencoder (a neural network that compresses data) squeezes 8 weekly market features (NIFTY return and "
  "volatility, VIX, breadth, dispersion, drawdown…) into a 2-D map; k-means then groups the weeks into 3 clusters, "
  "named Bull / Neutral / Stress from their training statistics. Refit each year on past data only.",
  use="Tells you what kind of market you are in. Measured: next-month NIFTY volatility was 12.6% after Bull weeks vs "
      "20.6% after Stress weeks. Used as an optional exposure filter in the Backtest lab.",
  caveat="It separates future volatility, not direction: Stress weeks were on average followed by rebounds.",
  related=["regime", "india_vix"], aliases=["Market regimes"])
E("regime", "Regime (Bull / Neutral / Stress)", "Regimes",
  "This week's market state from the regime detector.",
  bands=[], use="Bull ≈ calm, rising markets; Neutral ≈ normal; Stress ≈ high volatility and drawdowns. Press less "
                "(smaller positions, wider stops) in Stress.",
  related=["regime_model", "india_vix", "breadth"],
  aliases=["Regime", "Current regime", "REGIME", "CURRENT REGIME"], results=None)
E("regime_stats", "What came next, by regime", "Regimes",
  "Out-of-sample statistics for the weeks the detector put in each regime: how often, NIFTY's next-4-week return, "
  "next-month volatility and how often the next 4 weeks were positive.",
  use="Evidence for what the regime label is (and isn't) good for.",
  related=["regime_model"], aliases=["share", "next 4w NIFTY return", "next-month NIFTY vol", "hit rate (4w > 0)"])
E("rl_agent", "RL allocator (PPO agent)", "Models",
  "A reinforcement-learning agent that decides weekly portfolio weights for 29 NIFTY 50 stocks plus cash. A "
  "permutation-equivariant Transformer encoder reads each stock's features; the policy outputs how much to invest "
  "(Beta) and how to split it (Dirichlet). Trained with PPO in a simulator that charges real Indian costs and taxes, "
  "with a reward based on the investor's risk profile.",
  use="Powers the 'Investor plan': a personalised set of whole-share orders with plain-language reasons.",
  caveat="On validation (2019–20) it roughly matches equal weight (CAGR 18.8% vs 19.4%) with lower costs; it does not "
         "beat minimum-variance or HRP. The plan's data currently ends 17 Sep 2026.",
  related=["baselines"], aliases=["NiveshRL", "SB3-PPO", "RL allocator"], results=("metrics_val.csv", "NiveshRL"))
E("baselines", "Classical portfolio baselines", "Models",
  "Equal weight (1/N), risk parity (weights ∝ 1/volatility), Markowitz max-Sharpe, minimum variance and HRP "
  "(hierarchical risk parity: clusters correlated stocks first). Each is rebalanced under the same costs as the RL agent.",
  use="The honest comparison for the RL allocator. On validation, minimum variance had the best Sharpe (0.85).",
  related=["rl_agent", "sharpe"],
  aliases=["Equal weight", "Risk parity (inv-vol)", "Markowitz max-Sharpe", "Minimum variance", "HRP",
           "Equal weight (same universe)", "Equal weight (universe)"], results=("metrics_val.csv", "Minimum variance"))
E("finbert_model", "FinBERT (model)", "Models",
  "BERT, a Transformer language model, fine-tuned by Prosus AI on financial news and reports to classify sentences as "
  "positive, neutral or negative. Runs on your CPU in the background worker (about 3 minutes for ~1,800 headlines).",
  related=["finbert"], aliases=["FinBERT"])

# =====================================================================================================
# Backtest & risk
# =====================================================================================================
E("cagr", "CAGR (compound annual growth rate)", "Backtest & risk",
  "The constant yearly return that would turn the starting value into the final value.",
  unit="pct", bands=[(0, "losing", ""), (0.10, "below NIFTY", "NIFTY's long-run price return is ~10–12%/yr"),
                     (0.20, "good", ""), (INF, "very high", "check survivorship bias and costs before believing it")],
  caveat="Backtests here use today's NIFTY 200 members, which inflates absolute returns: compare against the "
         "equal-weight row, not NIFTY.", related=["sharpe", "max_drawdown"], aliases=["CAGR"])
E("sharpe", "Sharpe ratio", "Backtest & risk",
  "Return above the risk-free rate (6.5% here) per unit of volatility.",
  bands=[(0, "worse than cash", ""), (0.5, "poor", ""), (1.0, "decent", "NIFTY is ~0.4–0.6 long-run"),
         (1.5, "good", ""), (INF, "excellent", "rare and fragile out of sample")],
  use="The standard risk-adjusted return measure.", related=["sortino", "calmar", "info_ratio"],
  aliases=["Sharpe", "Rolling 1y Sharpe", "Rolling 1-year Sharpe"])
E("sortino", "Sortino ratio", "Backtest & risk",
  "Like Sharpe but only penalises downside volatility.", bands=[(1, "modest", ""), (2, "good", ""), (INF, "excellent", "")],
  related=["sharpe"], aliases=["Sortino"])
E("volatility_ann", "Volatility (annualised)", "Backtest & risk",
  "Standard deviation of daily returns × √252: how bumpy the ride is.", unit="pct",
  bands=[(0.12, "low", ""), (0.20, "moderate", "close to NIFTY's long-run ~16–20%"), (INF, "high", "")],
  related=["sharpe", "vol_60d"], aliases=["Vol", "Volatility", "Strategy ann. vol"])
E("max_drawdown", "Max drawdown", "Backtest & risk",
  "The worst peak-to-trough fall of the portfolio value.",
  unit="pct", bands=[(-0.40, "severe", "about what NIFTY fell in 2008"), (-0.25, "large", ""), (-0.10, "moderate", ""),
                     (INF, "mild", "")],
  use="The number most investors actually feel; ask whether you could sit through it.",
  related=["calmar", "max_dd_1y"], aliases=["Max drawdown", "MaxDD", "Strategy drawdown %", "NIFTY drawdown %"])
E("calmar", "Calmar ratio", "Backtest & risk", "CAGR ÷ |max drawdown|.",
  bands=[(0.5, "weak", ""), (1.0, "decent", ""), (INF, "strong", "")], related=["max_drawdown"], aliases=["Calmar"])
E("alpha", "Alpha and beta vs NIFTY", "Backtest & risk",
  "From regressing the strategy's returns on NIFTY's: beta = market exposure, alpha = the annual return left over.",
  unit="pct", bands=[(0, "negative alpha", ""), (0.03, "small", ""), (INF, "large", "")],
  caveat="Survivorship bias inflates alpha in these backtests.", related=["beta", "info_ratio"],
  aliases=["Alpha vs NIFTY", "Alpha"])
E("info_ratio", "Information ratio", "Backtest & risk", "Excess return over NIFTY ÷ tracking error.",
  bands=[(0.3, "weak", ""), (0.7, "good", ""), (INF, "excellent", "")], related=["alpha"],
  aliases=["Info ratio", "InfoRatio"])
E("turnover", "Turnover per year", "Backtest & risk",
  "How much of the portfolio is traded per year (one-way): 1.0 = the whole portfolio replaced once a year.",
  bands=[(1, "low", "cheap to run"), (4, "moderate", ""), (INF, "high", "costs and taxes eat returns")],
  related=["cost_drag"], aliases=["Turnover / yr", "Turnover/yr", "turnover"])
E("cost_drag", "Cost drag per year", "Backtest & risk",
  "Return lost to trading costs per year: STT, stamp duty, exchange and SEBI fees, GST, DP charges and slippage.",
  unit="pct", bands=[(0.005, "small", ""), (0.015, "noticeable", ""), (INF, "large", "monthly turnover strategies pay "
                                                                                       "2–3%/yr here")],
  related=["turnover", "india_costs"], aliases=["Cost drag / yr", "Costs(Rs)", "cost"])
E("india_costs", "Indian trading costs", "Desk",
  "Delivery trades in India pay STT (0.1% buy and sell), stamp duty (0.015% on buys), exchange and SEBI fees, 18% GST "
  "on brokerage and fees, and a flat DP charge (~₹16) per stock sold, plus capital-gains tax (STCG 20%, LTCG 12.5% "
  "above ₹1.25 lakh/yr).",
  use="Every backtest and the RL simulator charge these, which is why high-turnover strategies look worse here than "
      "in most published research.", related=["cost_drag"], aliases=["TaxLiability(Rs)"])
E("hit_rate", "Hit rate", "Backtest & risk", "Share of periods (weeks here) with a positive return.", unit="pct",
  bands=[(0.5, "low", ""), (0.6, "typical", ""), (INF, "high", "")], aliases=["Hit rate (wk)", "HitRate", "hit"])
E("avg_holdings", "Average holdings", "Backtest & risk", "Average number of stocks held.", aliases=["Avg holdings",
                                                                                                     "holdings"])
E("var", "VaR / CVaR (95%)", "Backtest & risk",
  "Value at Risk: the daily loss exceeded on only 5% of days. CVaR: the average loss on those worst 5% days.",
  unit="pct", use="Sizing for bad days.", related=["max_drawdown"], aliases=["VaR95", "CVaR95"])
E("luck_test", "Luck test (random-pick portfolios)", "Backtest & risk",
  "Runs the same strategy rules (number of stocks, weights, caps, rebalance dates, costs, overlays) hundreds of times "
  "but picks the stocks at random from the same eligible set each time. The spread of those results is what luck alone "
  "produces.",
  how="200 paths, seed 0 (reproducible). The strategy's percentile = share of random portfolios it beat.",
  bands=[(0.5, "worse than typical luck", ""), (0.75, "inside the luck range", "not clearly better than random"),
         (0.95, "better than most luck", ""), (INF, "better than almost all luck", "selection very likely adds value")],
  unit="pct", use="Separates skill from a rising market: a strategy can beat NIFTY simply because its universe did.",
  caveat="Still shares the universe's biases; read it on point-in-time members (see the Survivorship report).",
  related=["return_dd", "cagr", "survivorship"], aliases=["Luck test", "Beats % of random (CAGR)",
                                                        "Beats % of random (Return/DD)"])
E("return_dd", "Return ÷ drawdown", "Backtest & risk",
  "CAGR divided by the maximum drawdown (as a positive number): return earned per unit of the worst fall.",
  bands=[(0.3, "poor", "NIFTY buy-and-hold is about 0.3 over 2014–2026"), (0.6, "fair", ""), (1.0, "good", ""),
         (INF, "excellent", "")],
  use="Compares a strategy that is sometimes in cash with an always-invested index fairly.",
  related=["calmar", "max_drawdown", "luck_test"], aliases=["Return/DD"])
E("survivorship", "Survivorship bias and point-in-time members", "Backtest & risk",
  "Testing on today's index members only sees companies that survived and stayed in the index; those that collapsed or "
  "were demoted are missing, which inflates every backtest (momentum most).",
  how="This app rebuilds NIFTY 200 membership from 9 archived NSE snapshots (2014–2026). A stock counts only from the "
      "snapshot it appears in (stale between snapshots, never anticipated); renames are matched by ISIN. Results are "
      "shown for both universes over the same months in report/results/survivorship.md.",
  caveat="About 7% of member-days have no Yahoo prices (mostly delisted or merged companies), so the point-in-time "
         "results are still slightly flattered. Delisted holdings are carried at their last price.",
  related=["luck_test", "cagr"], aliases=["point-in-time members", "today's members (biased)", "Survivorship report"])
E("trade_plan", "Trade plan (entry, stop, targets)", "Desk",
  "A rule-based plan for one stock: where to buy, where you are wrong (stop), where to take profit.",
  how="Entry = last close (you'd buy at the next open). Stop just under the lowest low of the last 10 days, kept between "
      "2× and 2.5× ATR(14) below entry and never more than 8% away. R = entry − stop. T1 = entry + 1.5R, "
      "T2 = entry + 2.5R. Suggested: book a third at T1, move the stop to entry, trail the rest 3 × ATR under the highest "
      "close.",
  use="Turns a watch list into a concrete, risk-limited order and tells you in advance what a loss costs.",
  caveat="Rules, not predictions: on gaps the stop can fill worse than planned. Not advice; nothing is ordered.",
  related=["r_multiple", "position_size", "risk_state", "atr_pct"],
  aliases=["Entry from", "Entry", "Stop", "T1", "T2", "Trade plan", "Capital planner"])
E("stop_pct", "Stop distance %", "Desk", "How far the stop is below entry, as a % of the entry price.",
  unit="pct", bands=[(0.03, "tight", "small loss per share; easier to get stopped by noise"),
                     (0.06, "normal", ""), (0.08, "wide", "fewer shares for the same rupee risk")],
  related=["trade_plan"], aliases=["Stop %"])
E("position_size", "Position size", "Desk",
  "How many shares to buy so that hitting the stop loses a fixed share of your capital.",
  how="Qty = capital × risk% × risk-state multiplier ÷ R, rounded down, and never more than 20% of capital in one stock. "
      "'₹ risk' = Qty × R: what you lose if the stop is hit.",
  use="Keeps every loss the same size whatever the stock's price or volatility; the single most important risk rule.",
  related=["trade_plan", "risk_state"], aliases=["Qty", "Amount", "₹ risk", "₹ at T1", "₹ at T2", "Positions",
                                                  "Invested", "Total risk", "At T1", "At T2", "Risk per trade",
                                                  "Max positions"])
E("risk_state", "Market risk state", "Desk",
  "Whether to trade at normal or half risk today.",
  how="Half risk (×0.5) when NIFTY is below its 200-day average, when fewer than half of NIFTY 200 stocks are above "
      "their 50-day average (narrow market), or in a Stress regime; otherwise normal (×1).",
  use="Sizes down automatically in weak, narrow markets. It never changes which stocks qualify, only how much you buy.",
  related=["position_size", "breadth", "regime"], aliases=["Risk state", "Risk multiplier"])
E("r_multiple", "R multiple (after costs)", "Desk",
  "A trade's result measured in units of what you risked: (exit − entry) ÷ (entry − stop), after delivery costs.",
  bands=[(-1.2, "worse than the stop", "a gap or slippage past the stop"), (0, "loss", ""), (1, "small win", ""),
         (2, "good win", ""), (INF, "big win", "")],
  use="Compares trades of any size or price fairly; average R × number of trades is your edge.",
  related=["trade_plan", "hit_rate"], aliases=["R (after costs)", "Average R", "Median R", "Total R", "Closed trades",
                                                "Win rate"])
E("exit_line", "Holding exit line", "Desk",
  "A trailing exit for stocks you already own: close − 3 × ATR(14), which is only ever raised, never lowered.",
  bands=[], use="Call: EXIT when the price is at or below it; REVIEW when the stock is on tomorrow's 'watch for "
                "weakness' list; otherwise HOLD.",
  caveat="A mechanical trailing stop, not a forecast. Taxes (STCG/LTCG) apply when you sell.",
  related=["atr_pct", "trade_plan"], aliases=["Exit line", "To exit line", "Call", "EXIT calls"])
E("holdings", "Holdings (Kite import)", "Desk",
  "Your portfolio imported from a Zerodha Kite holdings CSV (Portfolio → Holdings → Download), stored only on this PC.",
  related=["exit_line"], aliases=["Holdings", "Avg cost", "P&L", "P&L %", "Unrealised P&L", "Qty (holding)"])
E("journal", "Trade journal", "Desk", "Trades you actually took, with entry, stop, exit and quantity; scored in R "
  "after costs exactly like the backtests. It never feeds any model.",
  related=["r_multiple"], aliases=["Entry date", "Exit date", "Exit"])
E("intraday_agent", "Intraday paper agent", "Desk",
  "A virtual pool (default Rs 1,00,000) that trades liquid NSE stocks intraday on its own during market hours and learns "
  "from every result. Paper only: no order is ever placed.",
  how="Every 5 minutes: the most active ('in play') stocks from 934 liquid NSE names, six rule-based setups (opening-"
      "range breakout, VWAP reclaim/rejection, 9/20 EMA pullback, previous-day high/low break, gap-and-go, NR7) on "
      "finished 5-minute bars, an ML probability and a learned TAKE/HALF/SKIP, then fills at the next bar's open within "
      "fixed guardrails (at most 1% of the pool at risk per trade, at most 5 positions, stop after a -3% day, "
      "square-off 15:15).",
  use="An honest experiment in whether intraday rules plus learning can beat costs. The first 52-day replay lost money "
      "(-13.6% with the cost rule, gross also negative) while beating a random control; live paper trading is the test.",
  caveat="The +10% per day target has not been reached on any day. Intraday trading after costs is a negative-sum game "
         "for most participants; no verdict before 100 closed trades.",
  related=["agent_bandit", "intraday_costs", "random_control", "speculative_tax"],
  aliases=["Agent", "Pool", "Today P&L", "After tax", "Intraday agent", "Replay", "Gross", "Costs", "Trades",
           "vs random control", "Paper pool: agent vs random control (₹)",
           "Replay: paper pool, agent vs random control (₹)"])
E("agent_trade", "Agent trade / decision", "Desk",
  "One paper trade or decision: the stock, the setup that fired, long or short, the decision (TAKE full size, HALF "
  "size, SKIP, or BLOCKED by a guardrail), entry (next bar's open plus slippage), stop, target (2R), quantity, exit and "
  "its reason (target / stop / square-off), and why the agent decided as it did.",
  related=["intraday_agent", "agent_bandit"])
E("intraday_costs", "Intraday costs (gross to net)", "Desk",
  "Gross = price P&L. Costs = brokerage (Rs 20 or 0.03% per order, whichever is lower), STT 0.025% on the sell side, "
  "NSE transaction 0.00297%, SEBI Rs 10/crore, stamp duty 0.003% on buys, 18% GST on brokerage + exchange + SEBI, and "
  "0.03% slippage per side. Net = gross - costs.",
  use="With tight intraday stops, costs are often 0.2-0.3R per trade: more than the setups' edge. The agent skips "
      "signals whose costs would exceed 0.2R.",
  related=["speculative_tax", "intraday_agent"])
E("speculative_tax", "Speculative (intraday) tax", "Desk",
  "Intraday equity profit is speculative business income, taxed at your income-tax slab plus 4% cess. Speculative "
  "losses can only be set off against speculative gains (carried forward up to 4 years).",
  how="Computed on the financial year's net speculative profit so far (zero if the year is net negative); the slab is "
      "editable on the agent screen.", related=["intraday_costs"], aliases=["tax_accrued"])
E("agent_bandit", "Learning buckets (bandit)", "Desk",
  "Similar signals are grouped (setup x long/short x time of day x with/against NIFTY's trend, then broader groups). "
  "Each group's average net R decides: SKIP when clearly negative (t-stat below -2 with at least 20 outcomes), HALF "
  "when uncertain, TAKE otherwise or while still learning. Skipped signals are still followed, so wrong skips are "
  "learned.",
  bands=[(-2, "SKIP", "clearly loses after costs"), (0, "HALF / uncertain", ""), (INF, "TAKE", "")],
  use="Learns which setups and conditions lose money and stops taking them, within guardrails it can never change.",
  related=["intraday_agent", "agent_ml"])
E("agent_ml", "ML probability (intraday)", "Desk",
  "LightGBM's estimate that a signal ends in profit after costs, from features known at the signal bar. Trained only on "
  "earlier days; used once there are at least 300 outcomes. Below 40% the agent skips.",
  unit="pct", bands=[(0.4, "skip", ""), (0.5, "weak", ""), (INF, "favourable", "")], related=["agent_bandit"])
E("random_control", "Random control account", "Desk",
  "A second paper account that sees the same signals but takes each on a coin flip, with the same guardrails and "
  "costs. If the agent can't beat it, its learning isn't adding anything.",
  related=["luck_test", "intraday_agent"], aliases=["Random control", "control_net"])
E("agent_day", "Agent trading day", "Desk",
  "One day of the paper account: pool at start and end, gross P&L, costs, net P&L, tax accrued, trades, signals seen "
  "and skipped, the random control's net, and whether the day reached the +10% target.",
  related=["intraday_agent"], aliases=["Days"])
E("forward_tracker", "Track record (forward tracker)", "Desk",
  "Whether the app's suggestions worked: every 'watch for strength' pick is followed with the swing rules (buy at the "
  "next open; plan stop; a third booked at T1 = 1.5R with the stop moved to entry; the rest trailed 3 x ATR; 60-day "
  "cap; after delivery costs) next to an equal-size random sample of that day's other stocks.",
  how="Forward: real saved daily runs, as days accumulate. Historical replay: the same list rebuilt every 5th trading day "
      "2015 to today on point-in-time NIFTY 200 members from the walk-forward next-day model and technicals (no news). "
      "A stock counts once per 30 days per group; picks carry a hash of the rules so rule changes are visible.",
  use="Measured on point-in-time history: the monitor list averaged +0.133R per trade vs +0.055R for random picks (edge "
      "+0.08R, t-stat 2.4 over ~9,000 trades): a modest but real edge. The median trade still loses; winners are bigger.",
  caveat="No verdict on the forward record before 100 closed trades per group; one good or bad month says little.",
  related=["r_multiple", "random_control", "trade_plan"],
  aliases=["Track record", "Edge vs random", "Saved days", "Picks followed", "Closed", "Rules", "t-statistic",
           "Period", "Forward tracker", "Cumulative R after costs: monitor list vs random control"])
E("sentiment_test", "Sentiment forward test", "News & analysts",
  "FinBERT sentiment buckets (negative below -0.2, neutral, positive above +0.2) on each saved day, against the next 1 "
  "and 5 trading days' return minus the median stock.",
  use="The only way to learn whether news sentiment predicts anything here: it can't be backtested because there is no "
      "news history, so it is measured forward as days accumulate.",
  caveat="Meaningless with a handful of days; read it after months of daily runs.",
  related=["finbert", "sentiment_adj"], aliases=["avg excess", "hit (beat median)"])
E("range_model", "Who will move tomorrow (range model)", "Model outputs",
  "A model of each stock's next-day trading range (high minus low, as a % of price). It ranks stocks by how much they "
  "are likely to move tomorrow, not by direction.",
  how="LightGBM on today's range, 5- and 20-day average range, ATR %, gap, absolute return, volume vs normal, NR7/NR4, "
      "60-day volatility, VIX and NIFTY's move; walk-forward by year on a rolling 5-year window, point-in-time members. "
      "'Expected range %ile' = where the stock ranks among all stocks (100% = biggest expected move).",
  bands=[(0.5, "below-average mover", ""), (0.8, "above average", ""), (INF, "likely big mover", "")],
  unit="pct",
  use="Choosing what to watch intraday: volatility clusters, so range is far more predictable than direction (IC 0.51 vs "
      "0.06 for direction). The top 20 averaged a 4.8% next-day range vs 3.1% for the average stock.",
  caveat="Big movers move both ways: a range forecast is not a buy or sell signal.",
  related=["atr_pct", "vol_60d", "intraday_agent"], results=("range_summary.csv", "model"),
  aliases=["Expected range %ile", "Avg range (20d)", "Today's range", "Who will move", "Volume / avg",
           "Who will move tomorrow (expected range, not direction)"])
E("pattern_vcp", "VCP (volatility contraction pattern)", "Technicals",
  "Three consecutive 15-day windows whose high-low range keeps shrinking (the last under 10%), with volume drying up, "
  "above the 200-day average: sellers are running out.",
  use="Measured on point-in-time NIFTY 200 members 2014-2026 (non-overlapping 20-day periods, before costs): +0.20% vs "
      "the average stock, t 0.6: no reliable edge. It describes a setup; it does not predict one.",
  caveat="Rules describe price structure; a pattern is a context, not a forecast. An earlier measurement against the "
         "median stock showed large 'edges' that were an artefact of skewed returns (the average stock beats the median).",
  related=["pattern_state", "pattern_squeeze", "breakout"], results=("patterns_edge.csv", "VCP (tightening)"),
  aliases=["VCP (tightening)", "VCP"])
E("pattern_flat", "Flat base", "Technicals",
  "At least 5 weeks inside a range of 15% or less, closing within 5% of the top, above the 200-day average.",
  use="Measured: -0.02% vs the average stock over 20 days (t -0.1): no edge.", related=["pattern_state"],
  results=("patterns_edge.csv", "Flat base"), aliases=["Flat base"])
E("pattern_box", "52-week-high box", "Technicals",
  "Within 3% of the 52-week high with the last 20 days inside a box of 12% or less (Darvas-style).",
  use="Measured: -0.14% vs the average stock over 20 days (t -0.4): no edge.", related=["from_52w_high"],
  results=("patterns_edge.csv", "52-week-high box"), aliases=["52-week-high box"])
E("pattern_nr7", "NR7 (narrowest range of 7 days)", "Technicals",
  "Today's high-low range is the narrowest of the last 7 days: a quiet day.",
  use="Measured: +0.04% vs the average stock over 20 days (t 0.2): no edge. For intraday traders it marks a quiet "
      "stock: the next day's range is usually smaller, not bigger.", results=("patterns_edge.csv", "NR7 (quiet day)"),
  related=["pattern_squeeze"],
  aliases=["NR7 (quiet day)", "NR7"])
E("pattern_pocket", "Pocket pivot", "Technicals",
  "An up day whose volume beats every down day's volume of the previous 10 days, closing above the 10- and 50-day "
  "averages.",
  use="Measured: -0.02% vs the average stock over 20 days (t 0.0), beating the median only 43% of the time: no edge.",
  results=("patterns_edge.csv", "Pocket pivot"), related=["vol_ratio"], aliases=["Pocket pivot"])
E("pattern_squeeze", "Bollinger squeeze", "Technicals",
  "Bollinger Bands (20 days, 2 standard deviations) sitting inside Keltner Channels (20 days, 1.5 x ATR): volatility is "
  "unusually compressed, which tends to be followed by a larger move.",
  use="Measured: +0.14% vs the average stock over 20 days (t 1.3): not statistically reliable.",
  results=("patterns_edge.csv", "Bollinger squeeze"), related=["bb_width", "pattern_nr7"],
  aliases=["Bollinger squeeze", "Volatility squeeze"])
E("pattern_state", "Pivot distance and breakout state", "Technicals",
  "Pivot = the 20-day high before today (the top of the base). Distance = close / pivot - 1.",
  bands=[(-0.10, "early base", "well below the pivot"), (-0.03, "building", ""),
         (0.0, "near pivot", "close to breaking out"), (0.03, "breakout", "just above the pivot"),
         (0.08, "confirmed", ""), (INF, "extended", "far above the pivot: chasing risk")],
  unit="pct", use="'Failed' = broke above the pivot within the last 10 days and is now more than 2% back below it.",
  related=["pattern_flat", "pattern_vcp", "breakout"],
  aliases=["Pivot distance", "Patterns (count)", "breakout_state", "Base near pivot"])
E("delivery_pct", "Delivery %", "Technicals",
  "The share of a day's traded quantity that was actually delivered into demat accounts (from NSE's daily bhavcopy), "
  "rather than squared off intraday.",
  bands=[(0.30, "mostly intraday", "speculative churn"), (0.50, "mixed", ""), (0.70, "high delivery", "investors taking "
         "stock"), (INF, "very high", "")],
  unit="pct", use="Rising delivery on up days and falling on down days is read as genuine buying. 'Up - down days' = "
                  "average delivery on up days minus down days over the last 25 sessions.",
  caveat="Measured only forward (the app keeps the last 25 sessions); no long history to test it against returns yet.",
  related=["volume_phase", "vol_ratio"], aliases=["Delivery % (today)", "Delivery % (20d avg)", "Delivery: up − down days"])
E("volume_phase", "Volume phase (Accumulation / Neutral / Distribution)", "Technicals",
  "A 0-100 score of whether volume has been flowing in or out: 50 plus up/down volume (50 days, ±25), Chaikin money flow "
  "(±15), accumulation minus distribution days (±10), OBV change (±10) and delivery on up vs down days (±10).",
  bands=[(40, "Distribution", "more volume on down days"), (60, "Neutral", ""), (INF, "Accumulation", "more volume on up days")],
  use="Describes recent volume behaviour.",
  caveat="Measured on point-in-time members 2014-2026 (without delivery, non-overlapping 20-day periods): no edge. "
         "Accumulation -0.07% and Distribution +0.17% vs the average stock (t -0.8 / 1.1). It cannot see real orders.",
  related=["delivery_pct", "vol_ratio"], results=("volume_phase_edge.csv", "Accumulation"),
  aliases=["Volume phase score", "vol_phase"])
E("weight", "Portfolio weight", "Backtest & risk", "Share of the portfolio in each stock at the last rebalance.",
  unit="pct", aliases=["Weight"])
E("rebalance", "Rebalance log", "Backtest & risk", "One row per rebalance: turnover traded, costs paid (₹), "
  "portfolio value (₹), share invested (the rest is cash) and number of holdings.", aliases=["invested"],
  related=["turnover", "cost_drag"])
E("annual_returns", "Return by regime / period", "Backtest & risk",
  "Annualised average return of the strategy and NIFTY on the days in each regime or period.",
  aliases=["Strategy ann. return", "NIFTY ann. return", "mean"], related=["regime"])

# =====================================================================================================
# Volatility forecaster metrics
# =====================================================================================================
E("rmse_logvol", "RMSE of log-volatility", "Volatility forecaster",
  "Root-mean-squared error between log(forecast) and log(realised) volatility: a percentage-like accuracy measure.",
  bands=[(0.30, "very good", ""), (0.36, "good", "the LSTM scores 0.337 here"), (0.42, "baseline", "GARCH and "
                                                                                                    "historical ≈ 0.40"),
         (INF, "poor", "")], use="Lower is better.", related=["qlike", "vol_lstm"],
  aliases=["RMSE log-vol", "Forecast error by year: RMSE of log-volatility (lower = better)"])
E("mae_vol", "MAE of volatility", "Volatility forecaster", "Average absolute forecast error in volatility points "
  "(0.083 = 8.3 percentage points).", unit="pct", aliases=["MAE vol"], related=["rmse_logvol"])
E("qlike", "QLIKE", "Volatility forecaster",
  "A loss function that punishes under-forecasting risk more than over-forecasting; the academic standard for "
  "comparing volatility forecasts.", bands=[(0.3, "good", ""), (0.5, "fair", ""), (INF, "poor", "")],
  related=["rmse_logvol"], aliases=["QLIKE"])
E("corr_vol", "Correlation (forecast vs realised)", "Volatility forecaster",
  "How well forecasts rank stocks by future volatility (1 = perfect ordering).", aliases=["Corr"])
E("bias_vol", "Bias (forecast ÷ realised)", "Volatility forecaster",
  "Median forecast ÷ realised volatility. 1.00 = unbiased; above 1 = forecasts too high.",
  bands=[(0.9, "under-forecasts", "dangerous: risk looks smaller than it is"), (1.1, "about right", ""),
         (INF, "over-forecasts", "conservative")], aliases=["Bias (f/r)"])

# =====================================================================================================
# Charts
# =====================================================================================================
E("chart_heatmap", "Sector heatmap", "Charts",
  "Every NIFTY 200 stock as a tile, grouped by NSE sector, coloured by return over the chosen window (green up, red "
  "down; brighter = bigger move).",
  reading="Look for colour clustering: a whole sector red means a sector story (rates, policy, commodity prices), "
          "scattered colours mean stock-specific news. Hover a tile for its name and return; click to open the stock.",
  related=["ret_1d", "breadth"], aliases=["Heatmap", "HEATMAP"])
E("chart_sectors", "Best and worst sectors", "Charts",
  "Today's three strongest and weakest NSE sectors by average stock return.",
  reading="Sector rotation: money leaving one group and entering another often shows up here before it shows in the "
          "index.", aliases=["Best and worst sectors (1D %)"], related=["chart_heatmap"])
E("chart_ic_year", "IC by year", "Charts",
  "Each model's average monthly IC in each calendar year (one bar per model per year).",
  reading="Consistency matters more than any single tall bar. A model whose bars are mostly above zero has a durable "
          "signal; bars below zero mark years when its ranking pointed the wrong way (e.g. sharp reversals).",
  related=["ic", "ffnn", "transformer"], aliases=["Mean monthly IC by year (higher = better ranking)"])
E("chart_decile", "Returns by score decile", "Charts",
  "Average next-month return of the stocks in each tenth of a model's ranking, from D1 (least liked) to D10 (most liked).",
  reading="A working model slopes upward from D1 to D10; the steeper and smoother, the better. A flat or zig-zag line "
          "means its ranking isn't useful. Here every model's D10 beats its D1, momentum and the Transformer most "
          "clearly.", related=["decile", "ic"],
  aliases=["Average next-month return by score decile (%), D1 = worst, D10 = best"])
E("chart_spread", "Cumulative top-minus-bottom decile", "Charts",
  "Growth of ₹1 invested in each month's top decile minus the bottom decile, before costs, on a log scale.",
  reading="A steadily rising line = a persistent edge; flat stretches = periods the signal stopped working; drops = "
          "it reversed. Not investable as-is (shorting and costs ignored).", related=["decile"],
  aliases=["Cumulative top-minus-bottom decile (log, before costs)"])
E("chart_growth", "Growth of ₹1", "Charts",
  "The value of ₹1 invested in each strategy over time, after all costs, on a log scale (equal vertical distances = "
  "equal percentage moves).",
  reading="Compare slopes, not end points alone: a line that rises steadily is better than one that gets to the same "
          "place through deep falls. Always compare against the equal-weight line on the same universe.",
  related=["cagr", "max_drawdown"], aliases=["Growth of ₹1 after all costs (log)", "Growth of ₹1 after costs (validation)"])
E("chart_drawdown", "Drawdown chart", "Charts",
  "How far below its previous peak each line is, at every date.",
  reading="Depth = how much you would have lost from the top; width = how long recovery took. Compare with NIFTY's "
          "drawdowns in the same crises.", related=["max_drawdown"], aliases=["Drawdown", "Drawdown (%)"])
E("chart_regime", "NIFTY coloured by regime", "Charts",
  "Weekly NIFTY level with each week's dot coloured by the regime the detector assigned out of sample.",
  reading="Red (Stress) clusters around crashes and volatile spells; green (Bull) around calm uptrends. Notice that "
          "red stretches are often followed by recoveries: the label is about volatility, not direction.",
  related=["regime_model"], aliases=["NIFTY 50, each week coloured by its out-of-sample regime"])
E("chart_vol_year", "Volatility forecast error by year", "Charts",
  "Each volatility model's RMSE of log-volatility per year (lower = better).",
  reading="A good forecaster stays lowest in most years, not only on average. Spikes mark years with sudden volatility "
          "changes (e.g. 2020) that every model struggled with.", related=["rmse_logvol", "vol_lstm"],
  aliases=["Forecast error by year"])
E("market_habits", "Market habits", "Desk",
  "Statistics measured from NIFTY's own history: average return and up-day share by weekday, what happened after "
  "±2% days and 3-day streaks, and intraday patterns from 60 days of 5-minute bars.",
  use="Context for planning the day (e.g. 'the first hour set the day's direction 71% of recent sessions').",
  caveat="Historical tendencies, small edges, can change. Not signals.", aliases=["Market habits (NIFTY, measured from "
                                                                                    "history, not opinions)"])
E("checklist", "Pre-entry checklist", "Desk",
  "A Go / Wait / No-go verdict from the app's own rules before a swing entry: position size possible, market risk "
  "state, the stock's long-term trend, results within 7 days, stop width, liquidity, whether it is stretched (RSI > 75 "
  "or extended from its pivot), the model views, and how similar past setups turned out.",
  how="No-go = a blocking problem (e.g. the capital cannot buy one share at this risk). Wait = two or more cautions. "
      "Go = at most one caution.",
  use="Slows down an impulsive entry and shows in one place what to look at. Most good entries have one caution or "
      "none; buying with several is a decision to make consciously.",
  caveat="The rules are simple and partly overlap. A Go is not a forecast that the trade will work; the replay's median "
         "trade loses even when the average wins.", related=["trade_plan", "similar_setups", "risk_state"])
E("similar_setups", "Similar past setups", "Desk",
  "Closed trades from the point-in-time replay of the same swing-plan rules (2015-2026, ~9,000 trades in NIFTY 200 "
  "stocks, model picks and random picks together) that started in the same context as this stock today: NIFTY above "
  "or below its 200-day average, the stock above or below its own, its RSI(14) zone and how far it was from its "
  "20-day high.",
  how="If fewer than 30 trades match, the least important tag is dropped (market, then distance from the high, then "
      "RSI zone) and the note says so. Outcomes are R after delivery costs: −1R ≈ the stop was hit.",
  bands=[(-0.1, "worse than average", "trades like this lost on average"), (0.1, "about average", ""),
         (INF, "better than average", "")],
  use="Shows the realistic spread of outcomes for this kind of entry: how often it wins, how big the typical loss and "
      "win are, and whether this context did better or worse than all trades.",
  caveat="Similar context, not the same stock or the same chart. Hundreds of trades still carry noise (see the t-stat: "
         "below 2 is not reliable). Past distributions shift with markets.",
  related=["r_multiple", "forward_tracker", "checklist"],
  aliases=["Similar trades", "Average days held", "Similar setups", "All trades", "Outcome",
           "What happened to similar past setups"])
E("scenario_pnl", "Trade scenarios", "Desk",
  "The rupee result of each way the swing plan can end, for the planned quantity: stopped out, a gap 3% through the "
  "stop, sold flat, T1 then stopped at entry, T1 then T2, and all at T2.",
  how="Gross = (exit − entry) × shares, with a third sold at T1 where the plan says so. Costs = brokerage, STT, "
      "exchange, SEBI, stamp duty, DP charge, GST and slippage (configs/costs_india.yaml). Tax = short-term capital "
      "gains on a net gain. Net R = net ÷ (R × shares).",
  use="Shows that a 'flat' trade still costs money, how much worse a gap is than the planned stop, and what costs and "
      "tax take out of a win, before you decide whether the trade is worth it.",
  caveat="Real fills differ; the gap scenario is an example, not the worst case.",
  related=["trade_plan", "stcg", "r_multiple"],
  aliases=["Exit price", "Gross ₹", "Costs ₹", "Net ₹", "Net R", "% of capital",
           "Scenarios: what each ending is worth after costs and tax"])
E("stcg", "Short-term capital gains tax (STCG)", "Desk",
  "Tax on profits from listed shares held for 12 months or less (STT paid): 20% plus 4% cess = 20.8%, for sales from "
  "23 July 2024. Held longer, gains are long-term (12.5% above ₹1.25 lakh a year).",
  how="The scenario table applies 20.8% to a scenario's net gain; losses show no tax.",
  use="A short-term winner keeps about 79% of its pre-tax profit. Losses can be set off against gains in the same "
      "year and carried forward 8 years.",
  caveat="Simplified, not tax advice: surcharge, your other gains and losses, and intraday (speculative) trades taxed at "
         "your slab are not included.", related=["scenario_pnl"], aliases=["Tax ₹", "STCG"])
E("custom_alerts", "Alert rules", "Desk",
  "Your own conditions on a NIFTY 200 stock, checked every 15 seconds: price at or above/below a level, the day's move, "
  "price above/below its N-day average, RSI(14), today's volume vs its 20-day average, or a new 52-week high/low. Each "
  "fires a Windows notification and is logged; 'once' rules then switch off, 'once a day' rules re-arm the next day.",
  how="Uses the live stream during market hours (today's live price stands in for today's close in the SMA, RSI and "
      "52-week checks); when the market is closed, the last close. A rule whose input is unavailable (e.g. no live "
      "volume) never fires.",
  use="Lets the app watch levels from your trade plan (entry, stop, targets) so you do not have to.",
  caveat="The Yahoo stream can lag or drop; an alert is a prompt to look, not an execution. Nothing is ordered.",
  related=["alerts", "paper_trading"], aliases=["Rule", "Repeat", "Status", "Last fired", "Now", "Your rules"])
E("paper_trading", "Paper trading", "Desk",
  "A pretend delivery account in My desk (starting from your desk capital) where you place orders by hand and they "
  "fill against the live stream: market orders at the next live price plus half the spread, limit buys when the price "
  "trades at or below the limit. A buy can carry a stop and a target that close the whole position.",
  how="A stop fills at the live price that crossed it (so a gap fills worse than the stop, as in real life); a target at "
      "the target or better. Round-trip delivery costs come from configs/costs_india.yaml. R uses the stop set at entry.",
  use="Practise following the trade plan (sizing, stops, not moving targets) and build a record in R before risking "
      "money. 'Paper trade this plan' on a stock page fills the ticket from its swing plan.",
  caveat="Paper fills are kinder than real ones (no queue, no partial fills, no slippage beyond half the spread) and "
         "there is no emotional cost. Orders placed while the market is closed wait for the next live price. Nothing "
         "is ever sent to a broker.",
  related=["trade_plan", "r_multiple", "journal", "custom_alerts"],
  aliases=["Paper equity", "Paper cash", "Unrealised P&L", "Realised P&L", "Closed paper trades", "Side", "Type",
           "Limit", "Target", "Placed", "Avg", "P&L ₹", "P&L %", "Open R", "Opened", "Closed", "Reason", "Entry", "Exit",
           "Order ticket", "Open orders", "Positions", "Closed paper trades"])
E("portfolio_risk", "Portfolio risk and allocation", "Desk",
  "How your current holdings behave together: value, beta (how much the portfolio moves when NIFTY moves 1%), "
  "annualised volatility, 1-day value at risk, sector weights, the largest holding and the 'effective' number of "
  "holdings.",
  how="Daily returns of today's holdings at today's weights over the last year (as if you had held them all year). "
      "VaR = the loss exceeded on only 5% of those days (historical). Effective holdings = 1 ÷ Σ weight²: ten equal "
      "holdings = 10, one 50% position among ten ≈ 3.5.",
  bands=[(0.8, "defensive", "beta below 0.8"), (1.2, "market-like", ""), (INF, "aggressive", "beta above 1.2")],
  use="Shows whether the portfolio is really diversified or a bet on one sector or one stock, and the size of an "
      "ordinary bad day in rupees before it happens.",
  caveat="Backward-looking: crashes exceed the VaR, and correlations rise in sell-offs. Holdings outside the NIFTY 200 "
         "have no price history here and are left out.",
  related=["beta", "risk_contrib", "max_drawdown"],
  aliases=["Portfolio value", "Portfolio beta", "Portfolio volatility", "1-day VaR (95%)", "Largest holding",
           "Effective holdings", "Weight", "Value ₹", "Risk and allocation"])
E("risk_contrib", "Share of portfolio risk", "Desk",
  "Each holding's share of the portfolio's total variance: weight × its covariance with the portfolio ÷ portfolio "
  "variance. The shares add up to 100%.",
  use="A 10% position can carry 25% of the risk if it is volatile and moves with the rest; trimming it reduces risk "
      "more than trimming a calm, unrelated stock.", related=["portfolio_risk"], aliases=["Share of risk"])
E("xirr", "XIRR", "Desk",
  "The annualised return of all your trades, taking timing into account: every buy is money in, every sale and "
  "today's value of what you still hold is money out, each on its own date.",
  how="Solves Σ cash flow ÷ (1 + r)^(years since the first trade) = 0 for r. Before brokerage and taxes.",
  bands=[(0.0, "losing money", ""), (0.12, "below a typical index fund", "NIFTY's long-run average is ~11-13%/yr"),
         (INF, "beating a typical index fund", "")], unit="pct",
  use="The fair way to compare your trading with an index fund or FD, because it accounts for when money went in.",
  caveat="Needs the full history: sales of shares bought before the tradebook starts are left out. Short histories "
         "give extreme annualised numbers.", related=["cagr", "cg_tax"], aliases=["XIRR"])
E("cg_tax", "Capital-gains tax on shares", "Desk",
  "Profit from selling listed shares is taxed by holding period. Held 12 months or less: short-term (STCG) at 20%. "
  "Held longer: long-term (LTCG) at 12.5% on the year's net gains above ₹1,25,000. Plus 4% cess. These rates apply to "
  "sales from 23 July 2024.",
  how="Sales are matched to the oldest buys first (FIFO). Within a financial year (April-March) short-term losses can "
      "offset short- or long-term gains; long-term losses only long-term gains. Net losses carry forward 8 years.",
  use="Know the tax a sale will trigger before you sell, and keep enough cash for advance tax.",
  caveat="An estimate for understanding, not tax advice: no surcharge, no grandfathering of gains before 31 Jan 2018, "
         "no adjustment of lots for bonuses/splits, and costs are ignored. Check with a tax professional.",
  related=["stcg", "tax_harvest", "xirr"],
  aliases=["Realised gains", "Taxable STCG", "Taxable LTCG", "Estimated tax", "LTCG allowance left", "Bought", "Sold",
           "Buy ₹", "Sell ₹", "Gain ₹", "Days held", "Term", "FY", "Returns and capital-gains tax",
           "Realised sales matched to buys (FIFO)"])
E("tax_harvest", "Tax ideas (harvesting)", "Desk",
  "Ways the timing of a sale changes this year's tax: booking a loss that offsets taxable gains, selling and rebuying a "
  "long-term winner to use the tax-free ₹1.25 lakh LTCG allowance (it resets your cost price higher), or waiting a "
  "few days for a short-term winner to turn long-term.",
  how="'Est. tax effect' = the tax saved (or extra tax from selling early) at 20% / 12.5% plus cess, for this "
      "financial year's gains so far.",
  use="Worth checking before March 31. India has no wash-sale rule for shares, but a sale must be genuine and costs "
      "apply on both legs.",
  caveat="Suggestions to look at, not advice: a tax saving never justifies holding a bad position or selling a good "
         "one, and the estimate ignores costs and your other income.", related=["cg_tax", "stcg"],
  aliases=["Idea", "Unrealised ₹", "Est. tax effect ₹", "Tax ideas for this year (not advice)"])
E("pnl_calendar", "Realised P&L calendar", "Desk",
  "Your realised gains and losses by month and year, from the tradebook matched FIFO (before costs).",
  use="Shows streaks and seasonality in your own results: are losses clustered after big wins? Did one month make or "
      "break the year?", related=["cg_tax", "journal"],
  aliases=["Realised P&L by month (₹, FIFO, before costs)"])
E("stacked_model", "Stacked next-day model", "Model outputs",
  "Four models predict tomorrow at the close: LightGBM and a sequence network (P(the stock beats tomorrow's median "
  "return)), a logistic regression (a stable linear view of the same), and the range model (how big tomorrow's "
  "high-low range may be). A fifth, 'meta' model learns how much to trust each and outputs one stacked P(up).",
  how="Base models are refit at the start of every month on the last 4 years up to 3 months before, and calibrated on "
      "those 3 months (the sequence net every quarter, because it is slow on a CPU). The meta model is a logistic "
      "regression over each base model's daily rank (plus direction × range), trained only on the base models' previous "
      "12 months of out-of-sample predictions, so it never sees a prediction a base model made on its own training data. "
      "The columns 'LightGBM … Range model' show each base model's rank today (100% = most bullish / most volatile).",
  bands=[(0.47, "Down", "below-median day expected"), (0.53, "Flat", "no clear view"), (INF, "Up", "")], unit="pct",
  use="One combined view per stock each evening, and a list of stocks whose outlook changed, to decide what to look at "
      "before tomorrow's open.",
  caveat="Next-day returns are mostly noise: even good models are right only slightly more than half the time, and the "
         "top picks lose after trading costs in this app's own tests. Its measured record is in "
         "report/results/refit_comparison.md. A forecast, not advice.",
  related=["p_up", "range_model", "pattern_change", "model_health"],
  aliases=["Stocks scored", "Up patterns", "Down patterns", "Heaviest meta weight", "LightGBM", "Sequence net",
           "Logistic", "Range model", "Stacked", "Pattern changes",
           "Stacked next-day model: pattern changes since the last close",
           "All stocks: stacked P(up), pattern and each model's vote"])
E("pattern_change", "Pattern change", "Model outputs",
  "A stock's stacked pattern = direction (Up when stacked P(up) ≥ 53%, Down ≤ 47%, else Flat) × expected size "
  "(volatile = top 30% of predicted ranges, quiet = bottom 30%, else normal). A change is any difference from the "
  "previous close; a notable change is a direction flip (Up↔Down) or a size jump (quiet↔volatile).",
  use="Highlights stocks whose outlook moved overnight, e.g. 'Down · quiet → Up · volatile' (the models now expect a "
      "large up-move): candidates to check on the chart and the trade plan.",
  caveat="Many small changes are just noise around the thresholds; the notable ones and a large Δ P(up) matter more.",
  related=["stacked_model"], aliases=["Notable changes", "Pattern now", "Pattern before", "P(up) now", "P(up) before",
                                      "Δ P(up)", "Change", "Notable changes tonight"])
E("model_health", "Model health (drift monitor)", "Model quality",
  "Each model's rank correlation (IC) between its prediction and the next day's actual returns, averaged over the "
  "last 60 trading days.",
  bands=[(0.0, "no skill / wrong way", "ignore the model for now"), (0.03, "weak", ""), (INF, "useful", "")],
  reading="Watch the trend, not single days: a model whose line falls below its own usual range is drifting (the "
          "market changed in a way it has not learned). The app flags a model whose latest value is below its own 10th "
          "percentile.", use="Tells you when to trust the stacked view less.", related=["ic", "stacked_model"],
  aliases=["Rolling 60-day rank IC with next-day returns (higher = better; 0 = no skill)",
           "Model health: rolling 60-day skill (rank IC with next-day returns)"])
E("watchlist_tier", "Watchlist tier", "Desk", "★ Must have = your highest-priority stocks; ☆ Preferred = on your radar.",
  aliases=["Tier"], related=["alerts"])
E("alerts", "Watchlist alerts", "Desk",
  "Fire once per day per stock and reason: price at or beyond your buy/sell target, a ±3% day, a news-sentiment flip, "
  "results within 7 days, or entering/leaving tomorrow's top list.",
  aliases=["Alerts", "Buy ≤", "Sell ≥"], related=["watchlist_tier"])
E("price", "Price / last", "Desk", "The latest traded price: live from the stream during market hours, otherwise the "
  "last close.", aliases=["Price", "Last", "close"])


# =====================================================================================================
# Stock deep-dive
# =====================================================================================================
E("report_card", "Report card", "Stock deep-dive",
  "Five scores from 0 to 100 that sum up a stock against its own NSE industry: Valuation (P/E, P/B, EV/EBITDA), "
  "Quality (ROE, operating margin, debt/equity), Growth (revenue and earnings growth), Momentum (3- and 6-month "
  "return, distance above the 200-day average) and Risk (volatility, beta, 1-year max drawdown).",
  how="For each measure, the share of NIFTY 200 stocks in the same industry the stock beats (lower is better for "
      "multiples, debt, volatility and beta); the category score is the average. Loss-making companies get no "
      "valuation percentile. Industries with fewer than 4 members are compared with the whole NIFTY 200.",
  bands=[(34, "weak for its group", "bottom third of its peers"), (66, "middling", ""),
         (INF, "strong for its group", "top third of its peers")],
  use="A fast, plain-language first read: is it cheap or expensive, well run, growing, in favour, and how bumpy? The "
      "reasons under each score show which measures drove it.",
  caveat="Relative, not absolute: the best stock in an expensive sector can still be expensive. Built from Yahoo "
         "fundamentals (can be stale or missing) and today's prices. Not a rating or a recommendation; no backtest "
         "shows these scores predict returns.",
  related=["pe", "roe", "revenue_growth", "ret_3m", "vol_60d"],
  aliases=["Valuation score", "Quality score", "Growth score", "Momentum score", "Risk score"])
E("why_moving", "Why is it moving?", "Stock deep-dive",
  "A short explanation of today's move assembled from facts: the move vs NIFTY and its sector, volume vs normal, the "
  "opening gap, results due or just out, the latest headlines and their FinBERT tone, and the breakout state.",
  how="If the stock moved within 0.5% of its sector's average, the move is called sector-wide; if it differs by 1.5% "
      "or more, stock-specific. Volume ≥ 1.5× its 20-day average = heavy (conviction), < 0.8× = light.",
  use="Separates 'the whole sector moved' from 'something happened to this company', which decides whether news or "
      "the market is the thing to read.",
  caveat="These are coincidences in time, not proven causes. A headline on the same day may not be why it moved.",
  related=["vol_ratio", "gap_pct", "finbert", "pattern_state"])
E("rs_line", "Relative strength line", "Stock deep-dive",
  "The stock's price divided by a benchmark (NIFTY, or an equal-weight average of its industry peers), rebased to 1 a "
  "year ago. 1.10 = it has done 10% better than the benchmark over the year.",
  bands=[(-0.10, "lagging clearly", "more than 10% behind"), (0.0, "lagging", ""), (0.10, "leading", ""),
         (INF, "leading clearly", "more than 10% ahead")], unit="pct",
  reading="Look at the slope, not the price: a rising line means the stock is beating the benchmark even if both are "
          "falling. A stock rising with a falling RS line is just riding the market.",
  use="Leaders tend to keep leading for a while (momentum), and the sector line shows whether it is the company or "
      "the industry that is strong.",
  caveat="Price only (no dividends). Sector line = equal-weight average of today's NIFTY 200 members in that industry.",
  related=["ret_1y", "beta", "momentum_model"],
  aliases=["vs NIFTY (1 year)", "vs sector (1 year)",
           "Relative strength: stock ÷ benchmark, rebased to 1 a year ago (rising = beating it)"])
E("corr_nifty", "Correlation with NIFTY", "Stock deep-dive",
  "How closely the stock's daily moves line up with NIFTY's, from −1 (opposite) to +1 (in lock-step), over 1 year.",
  bands=[(0.3, "independent", "moves mostly on its own news"), (0.6, "partly tied", ""),
         (INF, "tied to the market", "the market explains most of its daily moves")],
  use="Low correlation helps diversify a portfolio; high correlation means market risk dominates. Beta = correlation "
      "× (stock volatility ÷ NIFTY volatility).", related=["beta"])
E("seasonality", "Seasonality", "Stock deep-dive",
  "The stock's average and median return in each calendar month across all the years of price history, the share of "
  "years that month was up, and the average return vs NIFTY in that month.",
  how="Month-end to month-end closes (adjusted for splits and bonuses). 'years' = how many of each month exist.",
  use="Context, e.g. a stock that usually rises into its results month. Look for months that are consistent (high or "
      "low 'up years') and different from NIFTY's.",
  caveat="With ~20 years per month, a 2% average difference is easily luck. Seasonality has no measured edge in this "
         "app; treat it as trivia unless it has an economic reason (e.g. festive demand).",
  related=["market_habits"], aliases=["avg return", "up years", "avg vs NIFTY",
                                      "Seasonality: average return by calendar month"])
E("results_reaction", "Results-day reaction", "Stock deep-dive",
  "How the stock moved on the first session after each quarterly results announcement: the close-to-close move, and "
  "the same move minus NIFTY's.",
  how="Announcement times from Yahoo's earnings calendar, converted to IST; an announcement before 15:30 IST counts "
      "that day, later ones the next trading day. Averages are over all quarters with price data.",
  bands=[(0.02, "calm", "results rarely move it much"), (0.04, "normal", ""),
         (INF, "jumpy", "results often move it a lot: size positions for a gap")], unit="pct",
  use="Tells you how much risk holding through results carries: a stock that averages ±5% on results days can gap "
      "past a stop. Pairs with 'Next results' on the stock header.",
  caveat="Yahoo's dates for NSE stocks are sometimes off by a day; a wrong date mixes in an ordinary day. A few "
         "quarters is a small sample.",
  related=["post_results_drift", "eps_surprise", "next_results"],
  aliases=["Results quarters", "Avg results-day move", "Up reactions", "day move",
           "How it reacted to quarterly results"])
E("post_results_drift", "Post-results drift", "Stock deep-dive",
  "The stock's return over the 20 trading days after the results day, minus NIFTY's over the same days.",
  use="Academic studies find prices keep drifting in the direction of an earnings surprise for weeks (post-earnings "
      "announcement drift). Here it is just this stock's history; compare it with the surprise column.",
  caveat="Not tested as a strategy in this app. The latest quarter has no value until 20 sessions have passed.",
  related=["results_reaction", "eps_surprise"], aliases=["Avg 20-day drift", "next 20d vs NIFTY"])
E("eps_surprise", "EPS surprise", "Stock deep-dive",
  "Reported earnings per share vs the analysts' consensus estimate before the results, in percent.",
  how="From Yahoo's earnings calendar ('EPS Estimate', 'Reported EPS', 'Surprise(%)').",
  bands=[(-5, "big miss", ""), (0, "miss", ""), (5, "beat", ""), (INF, "big beat", "")],
  use="Markets react to surprises, not to absolute numbers: a strong quarter that missed estimates often falls.",
  caveat="Consensus coverage of Indian companies on Yahoo is thin and estimates can be stale. Revenue and guidance "
         "matter too and are not shown.", related=["results_reaction", "eps"],
  aliases=["Beat estimate", "EPS est.", "EPS actual", "surprise %"])
E("dividend_history", "Dividend history", "Stock deep-dive",
  "Every dividend per share Yahoo records for the stock, by ex-date, plus the last 12 months' total and its yield on "
  "the last close.",
  use="Consistency (years paid) says more about a company's cash than one year's yield. You must hold the shares "
      "before the ex-date to receive a dividend; the price usually drops by about the dividend on that day.",
  caveat="In India dividends are taxed at your slab rate. Old dividends are not adjusted for later splits/bonuses.",
  related=["dividend_yield", "payout"], aliases=["Trailing 12m dividend", "Trailing yield", "Years paid",
                                                 "dividend ₹", "Ex-date", "Dividends"])
E("vol_cone", "Volatility cone", "Stock deep-dive",
  "The price ranges the stock would stay inside with about 68% and 95% probability after 5, 10, 20 and 60 trading "
  "days if its volatility stays at the estimate and moves are log-normal with no drift.",
  how="σ = the vol LSTM's next-month forecast if it is under 45 days old, otherwise 60-day realised volatility. "
      "1σ move over h days = σ × √(h/252); bands = price × e^(±σ√(h/252)) and e^(±1.96σ√(h/252)).",
  use="Sanity-check targets and stops: a target outside the 95% band in your time frame needs something unusual to "
      "happen; a stop inside the 68% band will be hit by ordinary noise.",
  caveat="Says nothing about direction. Real returns have fat tails: moves beyond the 95% band happen more often than "
         "5% of the time, especially around results.",
  related=["vol_lstm", "vol_60d", "atr_pct"],
  aliases=["95% low", "68% low", "68% high", "95% high", "1σ move",
           "Volatility cone: where the price may be (not where it will go)"])
E("drawdown_history", "Drawdown history", "Stock deep-dive",
  "The deepest peak-to-trough falls in the stock's price history, when they started and bottomed, and how many "
  "calendar days the price took to get back to the old peak.",
  use="Shows what holding this stock has actually felt like: if it has fallen 50% twice before, it can again. "
      "'not yet' = still below that peak.",
  caveat="Adjusted prices from the data start (often 2005 or listing); price only, no dividends.",
  related=["max_dd_1y", "max_drawdown", "chart_drawdown"],
  aliases=["peak", "trough", "depth", "recovered", "days to recover", "Deepest falls in the price history"])
E("peers", "Sector peers", "Stock deep-dive",
  "The largest NIFTY 200 stocks in the same NSE industry, with returns, valuation, quality and the app's analyst and "
  "next-day scores side by side.", use="Is this stock cheap or expensive, strong or weak, compared with its real "
  "competitors? The report card turns this table into percentiles.", related=["report_card"])


# =====================================================================================================
# Lookup
# =====================================================================================================
def _norm(s: str) -> str:
    s = str(s).replace("ⓘ", "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


_INDEX: dict[str, str] = {}


def _build_index() -> None:
    for k, e in G.items():
        for a in [e.title, k, *e.aliases]:
            _INDEX.setdefault(_norm(a), k)
    try:                                              # every screener column label maps to its entry
        from .research.screener import COLUMNS
        col_map = {"ret_1d": "ret_1d", "ret_1w": "ret_1w", "ret_1m": "ret_1m", "ret_3m": "ret_3m", "ret_6m": "ret_6m",
                   "ret_1y": "ret_1y", "gap_pct": "gap_pct", "rsi14": "rsi", "macd_hist": "macd", "adx14": "adx",
                   "supertrend": "supertrend", "vs_sma20": "vs_sma", "vs_sma50": "vs_sma", "vs_sma200": "vs_sma",
                   "golden_cross_5d": "golden_cross", "death_cross_5d": "golden_cross", "donchian_breakout": "breakout",
                   "from_52w_high": "from_52w_high", "from_52w_low": "from_52w_low", "new_52w_high": "new_52w",
                   "new_52w_low": "new_52w", "bb_pctb": "bb_pctb", "bb_width": "bb_width", "atr_pct": "atr_pct",
                   "vol_60d": "vol_60d", "beta_1y": "beta", "max_dd_1y": "max_dd_1y", "vol_ratio": "vol_ratio",
                   "turnover_cr": "turnover_cr", "marketCap": "market_cap", "trailingPE": "pe", "forwardPE": "pe",
                   "priceToBook": "pb", "enterpriseToEbitda": "ev_ebitda", "trailingEps": "eps",
                   "dividendYield": "dividend_yield", "payoutRatio": "payout", "returnOnEquity": "roe",
                   "returnOnAssets": "roa", "operatingMargins": "operating_margin", "profitMargins": "net_margin",
                   "debtToEquity": "debt_equity", "revenueGrowth": "revenue_growth", "earningsGrowth": "earnings_growth",
                   "heldPercentInsiders": "promoters", "heldPercentInstitutions": "institutions",
                   "analyst_score": "analyst_score", "target_upside": "target_upside",
                   "numberOfAnalystOpinions": "n_analysts", "days_to_earnings": "next_results", "prob_up": "p_up",
                   "monthly_pct": "ranker_pct", "sentiment": "finbert", "sentiment_adj": "sentiment_adj",
                   "n_news": "n_news", "buzz": "buzz"}
        for col, (label, _, _) in COLUMNS.items():
            if col in col_map:
                _INDEX.setdefault(_norm(label), col_map[col])
                _INDEX.setdefault(_norm(col), col_map[col])
    except Exception:
        pass


_build_index()
# Labels that name things rather than measure them: no explanation needed.
IGNORE = {_norm(x) for x in ["Company", "Sector", "Stock", "Note", "Headline", "Source", "When", "Group", "Indicator",
                             "Value", "Name", "Action", "Shares", "Price (Rs)", "Amount (Rs)", "", "Year",
                             "Trading day", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                             "Nov", "Dec", "Full year", "Volume (stock)", "Notes"]}


def lookup(label: str) -> Entry | None:
    """The entry for a label shown anywhere in the app (case-insensitive; ignores ⓘ and units in brackets)."""
    if label is None:
        return None
    n = _norm(label)
    if n in _INDEX:
        return G[_INDEX[n]]
    for cand in (re.sub(r"\s*\(.*?\)\s*", " ", n).strip(), n.rstrip(" %").strip(), n.split(" · ")[0].strip()):
        if cand in _INDEX:
            return G[_INDEX[cand]]
    return None


def band(entry: Entry, value) -> tuple[int, tuple] | None:
    """(index, band) for a numeric value, or None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or not entry.bands:
        return None
    for i, b in enumerate(entry.bands):
        if v < b[0]:
            return i, b
    return len(entry.bands) - 1, entry.bands[-1]


def fmt(entry: Entry, v: float) -> str:
    if v in (INF, -INF):
        return "∞" if v > 0 else "−∞"
    if entry.unit == "pct":
        return f"{v * 100:+.4g}%" if v < 0 else f"{v * 100:.4g}%"
    return f"{v:,.4g}"


def band_ranges(entry: Entry) -> list[tuple[str, str, str]]:
    """Human-readable (range, label, meaning) rows."""
    out, lo = [], None
    for upper, label, meaning in entry.bands:
        if lo is None:
            rng = f"below {fmt(entry, upper)}"
        elif upper == INF:
            rng = f"{fmt(entry, lo)} and above"
        else:
            rng = f"{fmt(entry, lo)} to {fmt(entry, upper)}"
        out.append((rng, label, meaning))
        lo = upper
    return out


def parse_value(text: str):
    """Best-effort number from a displayed value: '+0.88%' -> 0.0088, '₹1,167.70' -> 1167.7, '21.1x' -> 21.1."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).strip().replace(",", "").replace("₹", "").replace("−", "-").replace("–", "")
    m = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*(%|x|/100|L Cr|Cr)?", s)
    if not m:
        return None
    v = float(m.group(1))
    unit = m.group(2)
    if unit == "%":
        return v / 100
    if unit == "L Cr":                                # market caps shown in lakh crore -> crore
        return v * 1e5
    return v


def search(q: str) -> list[Entry]:
    q = _norm(q)
    if not q:
        return list(G.values())
    hits = []
    for e in G.values():
        hay = " ".join([e.title, e.key, *e.aliases, e.what, e.category]).lower()
        if q in hay:
            hits.append((0 if q in e.title.lower() else 1, e.title, e))
    return [e for _, _, e in sorted(hits, key=lambda x: (x[0], x[1]))]
