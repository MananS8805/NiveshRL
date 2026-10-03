"""Recent news headlines per stock from Google News RSS.

yfinance returns no news for NSE tickers, so headlines come from Google News'
public RSS search, restricted to India English and the last ``days`` days.

A headline is kept only if it names the company (short name) or its NSE
symbol, because a search for "Titan" also returns unrelated stories. Duplicate
syndicated copies are dropped by normalised title.
"""
from __future__ import annotations

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import timedelta, timezone
from email.utils import parsedate_to_datetime

import pandas as pd

IST = timezone(timedelta(hours=5, minutes=30))
# Names the press actually uses, where the official name differs.
ALIASES = {
    "SBIN": ["sbi"], "LT": ["l&t", "larsen"], "M&M": ["m&m", "mahindra"], "HDFCBANK": ["hdfc bank"],
    "BAJFINANCE": ["bajaj finance"], "BAJAJFINSV": ["bajaj finserv"], "BAJAJ-AUTO": ["bajaj auto"],
    "HINDUNILVR": ["hul", "hindustan unilever"], "ITC": ["itc"], "ONGC": ["ongc"], "NTPC": ["ntpc"],
    "TCS": ["tcs"], "INFY": ["infosys"], "KOTAKBANK": ["kotak"], "BHARTIARTL": ["airtel"],
    "MARUTI": ["maruti"], "TATASTEEL": ["tata steel"], "TMPV": ["tata motors"], "ADANIENT": ["adani enterprises"],
}
# SEO quote/prediction pages carry no news; they would only dilute sentiment.
_JUNK = re.compile(r"stock price, news|quote and history|price today|price prediction|prediction for|share price live",
                   re.I)
_SUFFIX = re.compile(r"\b(limited|ltd\.?|corporation|corp\.?|company|co\.?|india|of india|\(india\))\s*$", re.I)


def short_name(name: str) -> str:
    """'ICICI Bank Limited' -> 'ICICI Bank'; 'Titan Company Limited' -> 'Titan'."""
    s = str(name or "").strip()
    for _ in range(3):
        s2 = _SUFFIX.sub("", s).strip(" .,&")
        if s2 == s:
            break
        s = s2
    return s


def _key_terms(name: str, symbol: str) -> list[str]:
    sn = short_name(name)
    words = sn.split()
    first = words[0] if words else symbol
    terms = {symbol.lower(), sn.lower(), *ALIASES.get(symbol, [])}
    if len(first) >= 4:                       # 'Reliance', 'Infosys' identify the company on their own
        terms.add(first.lower())
    elif len(words) >= 2:                     # 'L&T', 'SBI Cards' need two words
        terms.add(" ".join(words[:2]).lower())
    return [t for t in terms if t]


def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()[:70]


def fetch_rss(query: str, days: int = 2, timeout: int = 20) -> list[dict]:
    q = urllib.parse.quote(f"{query} when:{days}d")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (NiveshRL research)"})
    xml = urllib.request.urlopen(req, timeout=timeout).read()
    out = []
    for it in ET.fromstring(xml).findall(".//item"):
        raw = it.findtext("title") or ""
        title, _, source = raw.rpartition(" - ")
        if not title:
            title, source = raw, (it.findtext("source") or "")
        try:
            ts = parsedate_to_datetime(it.findtext("pubDate")).astimezone(IST)
        except Exception:
            ts = None
        out.append({"title": title.strip(), "source": source.strip(), "link": it.findtext("link"), "published": ts})
    return out


def stock_news(ticker: str, name: str, days: int = 2) -> pd.DataFrame:
    sym = ticker.replace(".NS", "")
    sn = short_name(name)
    alias = ALIASES.get(sym, [])
    q = f'"{sn}"' + (f' OR "{alias[0]}"' if alias else "")
    items = fetch_rss(f"({q}) share OR stock OR shares", days=days)
    terms = _key_terms(name, sym)
    seen, rows = set(), []
    for it in items:
        t = it["title"].lower()
        if _JUNK.search(t):
            continue                                  # SEO quote page, not news
        if not any(re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", t) for term in terms):
            continue                                  # not about this company (whole words: "Titan" != "Titanic")
        k = _norm(it["title"])
        if k in seen:
            continue                                  # syndicated duplicate
        seen.add(k)
        rows.append({"ticker": ticker, **it})
    return pd.DataFrame(rows, columns=["ticker", "title", "source", "link", "published"])


def all_news(tickers: list[str], names: dict[str, str], days: int = 2, workers: int = 8, progress=None) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    frames = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(stock_news, t, names.get(t, t), days): t for t in tickers}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                frames.append(fut.result())
            except Exception:
                pass                                  # one feed failing never stops the rest
            if progress:
                progress(i, len(futs))
    df = pd.concat([f for f in frames if len(f)], ignore_index=True) if frames else pd.DataFrame()
    return df.sort_values("published", ascending=False) if len(df) else df


def market_news(days: int = 1) -> pd.DataFrame:
    rows = fetch_rss("Sensex OR Nifty stock market India", days=days)
    seen, out = set(), []
    for it in rows:
        k = _norm(it["title"])
        if k not in seen:
            seen.add(k)
            out.append(it)
    return pd.DataFrame(out)
