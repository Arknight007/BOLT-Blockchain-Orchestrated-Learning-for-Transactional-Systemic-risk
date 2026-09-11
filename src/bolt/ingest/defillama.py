"""DefiLlama public endpoints (BOLT_SPEC.md Section 3).

Added because two configured sources turned out to be insufficient for the
2020-2025 window:

* CoinGecko's free tier now refuses any historical range older than 365 days
  (``error_code 10012``), which removes market capitalisation and therefore the
  canonical NVT denominator for the whole study period.
* Binance lists no spot pair for every asset in the frozen universe.

DefiLlama is free, keyless, and serves daily history back to 2017. Both
endpoints used here are public and documented.

The stablecoin series is a genuine improvement rather than a workaround: the
aggregate circulating supply of ALL USD-pegged stablecoins is a better measure of
capital entering or leaving the crypto system than the sum of the two stablecoins
that happen to be in our universe.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from bolt.ingest.cache import CacheKey, DiskCache, cached_fetch
from bolt.ingest.http import RateLimiter, get_json
from bolt.logging_setup import get_logger

log = get_logger(__name__)

COINS_CHART_URL = "https://coins.llama.fi/chart"
STABLECOIN_CHART_URL = "https://stablecoins.llama.fi/stablecoincharts/all"

_PAGE_SPAN = 500          # the API's hard maximum data points per request
_SEARCH_WIDTH = "600"     # seconds of tolerance when snapping to a daily stamp
_MAX_PAGES = 20


def _epoch(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def _fetch_price_pages(coin_key: str, start: date, end: date, limiter: RateLimiter) -> list[dict]:
    """Page through the 500-point limit until the window is covered."""
    points: list[dict] = []
    cursor, stop = _epoch(start), _epoch(end)
    for _ in range(_MAX_PAGES):
        if cursor >= stop:
            break
        payload = get_json(
            f"{COINS_CHART_URL}/{coin_key}",
            params={"start": cursor, "span": _PAGE_SPAN, "period": "1d",
                    "searchWidth": _SEARCH_WIDTH},
            limiter=limiter,
        )
        coins = (payload or {}).get("coins") or {}
        if not coins:
            break
        prices = next(iter(coins.values())).get("prices") or []
        if not prices:
            break
        points.extend(prices)
        last = int(prices[-1]["timestamp"])
        if last <= cursor:
            break
        cursor = last + 86_400
    return points


def fetch_prices(
    cfg_section: dict,
    coingecko_id: str,
    symbol: str,
    start: date,
    end: date,
    cache: DiskCache,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Daily close prices for one asset. Returns an empty frame if unavailable.

    DefiLlama serves prices only, so open/high/low are reconstructed from
    consecutive closes. This is a documented degradation: intraday range is
    genuinely unavailable, so ATR collapses to a close-to-close measure for any
    asset sourced here, and the data card says so.
    """
    limiter = RateLimiter(cfg_section.get("rate_limit_per_minute", 60))
    key = CacheKey("defillama_chart", symbol, str(start), str(end), "1d")
    entry = cached_fetch(
        cache, key,
        lambda: _fetch_price_pages(f"coingecko:{coingecko_id}", start, end, limiter),
        refresh=refresh, url=f"{COINS_CHART_URL}/coingecko:{coingecko_id}",
    )
    points: list[dict] = entry.payload or []
    if not points:
        return pd.DataFrame()

    index = pd.to_datetime([int(p["timestamp"]) for p in points], unit="s", utc=True).normalize()
    close = pd.Series([float(p["price"]) for p in points], index=index).sort_index()
    close = close[~close.index.duplicated(keep="last")]
    close = close.loc[
        (close.index >= pd.Timestamp(start, tz="UTC"))
        & (close.index <= pd.Timestamp(end, tz="UTC"))
    ]
    if close.empty:
        return pd.DataFrame()

    frame = pd.DataFrame({"close": close})
    frame["open"] = frame["close"].shift(1).fillna(frame["close"])
    frame["high"] = frame[["open", "close"]].max(axis=1)
    frame["low"] = frame[["open", "close"]].min(axis=1)
    # DefiLlama's price endpoint carries no volume. Left missing, never zeroed:
    # zero volume is a factual claim about the market that would be false.
    frame["volume"] = float("nan")
    frame["quote_volume"] = float("nan")
    frame["trades"] = float("nan")
    frame.index.name = "timestamp"
    return frame


def fetch_stablecoin_supply(
    cfg: Any, cache: DiskCache, *, refresh: bool = False
) -> pd.Series:
    """Aggregate circulating USD value of all USD-pegged stablecoins, daily."""
    key = CacheKey("defillama_stablecoins", "MARKET", str(cfg.start_date), str(cfg.end_date), "1d")
    entry = cached_fetch(
        cache, key,
        lambda: get_json(STABLECOIN_CHART_URL, params={}, limiter=RateLimiter(30)),
        refresh=refresh, url=STABLECOIN_CHART_URL,
    )
    rows = entry.payload or []
    if not rows:
        raise ValueError("DefiLlama returned no stablecoin history; refusing to continue")

    records = []
    for row in rows:
        total = row.get("totalCirculatingUSD") or {}
        value = total.get("peggedUSD")
        if value is None:
            continue
        records.append((int(row["date"]), float(value)))
    if not records:
        raise ValueError("DefiLlama stablecoin payload carried no peggedUSD totals")

    index = pd.to_datetime([r[0] for r in records], unit="s", utc=True).normalize()
    series = pd.Series([r[1] for r in records], index=index, name="stablecoin_supply")
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]
    log.info("stablecoin supply: %d days, %s..%s, latest $%.1fB",
             len(series), series.index.min().date(), series.index.max().date(),
             series.iloc[-1] / 1e9)
    return series
