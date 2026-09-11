"""Market data ingestion (BOLT_SPEC.md Section 3).

Source chain, in order:

1. **Binance public API** - primary for every asset with a spot pair. It is the
   venue where the trades actually occurred, so its OHLCV is a primary record
   rather than an aggregation, and it carries taker-flow and trade counts that
   the on-chain proxy layer depends on.
2. **DefiLlama** - daily closes for assets Binance does not list. Prices only;
   volume is left missing rather than zeroed.
3. **CoinGecko** - last resort. Its free tier now refuses ranges older than 365
   days, so it cannot cover the study window on its own.

Every series returned carries a timezone-aware UTC ``timestamp`` index and a
``source`` column naming where the row came from, which flows into the data card.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from bolt.config import Asset, BoltConfig
from bolt.ingest.cache import CacheKey, DiskCache, IngestError, cached_fetch
from bolt.ingest.defillama import fetch_prices as fetch_llama_prices
from bolt.ingest.http import RateLimiter, get_json
from bolt.logging_setup import get_logger

log = get_logger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume", "trades"]
EXTRA_COLUMNS = ["taker_buy_base"]
_BINANCE_PAGE_LIMIT = 1000
_MIN_USABLE_DAYS = 90


def _to_millis(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)


def _fetch_binance_klines(
    base_url: str, symbol: str, start: date, end: date, interval: str, limiter: RateLimiter
) -> list[list[Any]]:
    """Page through Binance klines until the window is covered."""
    rows: list[list[Any]] = []
    cursor = _to_millis(start)
    end_ms = _to_millis(end) + 86_399_000
    while cursor <= end_ms:
        page = get_json(
            f"{base_url}/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "startTime": cursor,
                    "endTime": end_ms, "limit": _BINANCE_PAGE_LIMIT},
            limiter=limiter,
        )
        if not page:
            break
        rows.extend(page)
        if len(page) < _BINANCE_PAGE_LIMIT:
            break
        cursor = int(page[-1][0]) + 86_400_000
    return rows


def _binance_frame(rows: list[list[Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=OHLCV_COLUMNS + EXTRA_COLUMNS)
    frame = pd.DataFrame(
        rows,
        columns=["open_time", "open", "high", "low", "close", "volume", "close_time",
                 "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"],
    )
    frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    numeric = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base"]
    frame[numeric] = frame[numeric].astype(float)
    frame["trades"] = frame["trades"].astype(float)
    return frame.set_index("timestamp")[OHLCV_COLUMNS + EXTRA_COLUMNS].sort_index()


def _coingecko_frame(payload: Any) -> pd.DataFrame:
    """Daily frame from CoinGecko market_chart. Closes only; OHLC reconstructed."""
    def series(name: str) -> pd.Series:
        pairs = (payload or {}).get(name) or []
        if not pairs:
            return pd.Series(dtype=float)
        idx = pd.to_datetime([p[0] for p in pairs], unit="ms", utc=True)
        return pd.Series([float(p[1]) for p in pairs], index=idx).sort_index()

    price, volume = series("prices"), series("total_volumes")
    if price.empty:
        return pd.DataFrame()
    frame = pd.DataFrame({"close": price, "volume": volume})
    frame.index.name = "timestamp"
    frame = frame.resample("1D").last().dropna(subset=["close"])
    frame["open"] = frame["close"].shift(1).fillna(frame["close"])
    frame["high"] = frame[["open", "close"]].max(axis=1)
    frame["low"] = frame[["open", "close"]].min(axis=1)
    frame["quote_volume"] = frame["volume"] * frame["close"]
    frame["trades"] = float("nan")
    return frame


def fetch_ohlcv(
    cfg: BoltConfig, asset: Asset, cache: DiskCache, *, refresh: bool = False
) -> pd.DataFrame:
    """Daily OHLCV for one asset via the three-tier source chain.

    Returns an EMPTY frame when no source can cover the asset. The caller decides
    whether that is fatal; for a context asset it is a documented exclusion, and
    either way it is recorded in the data card rather than silently filled.
    """
    ingest = cfg.section("ingest")
    start, end, freq = cfg.start_date, cfg.end_date, cfg.raw["data"]["frequency"]
    sources = ingest["sources"]

    # --- 1. Binance -------------------------------------------------------
    if asset.binance_symbol and sources["binance"]["enabled"]:
        base = sources["binance"]["base_url"]
        limiter = RateLimiter(sources["binance"]["rate_limit_per_minute"])
        key = CacheKey("binance", asset.symbol, str(start), str(end), freq)
        entry = cached_fetch(
            cache, key,
            lambda: _fetch_binance_klines(base, asset.binance_symbol, start, end, freq, limiter),
            refresh=refresh, url=f"{base}/api/v3/klines",
        )
        frame = _binance_frame(entry.payload)
        if len(frame) >= _MIN_USABLE_DAYS:
            frame["source"] = "binance"
            return frame
        log.warning("%s: Binance returned only %d day(s); trying DefiLlama",
                    asset.symbol, len(frame))

    # --- 2. DefiLlama -----------------------------------------------------
    frame = fetch_llama_prices(
        sources.get("coingecko", {}), asset.coingecko_id, asset.symbol,
        start, end, cache, refresh=refresh,
    )
    if len(frame) >= _MIN_USABLE_DAYS:
        frame["taker_buy_base"] = float("nan")
        frame["source"] = "defillama"
        log.info("%s: %d day(s) from DefiLlama (prices only, no volume)",
                 asset.symbol, len(frame))
        return frame

    # --- 3. CoinGecko (365-day ceiling on the free tier) -------------------
    cg = sources["coingecko"]
    if cg["enabled"]:
        limiter = RateLimiter(cg["rate_limit_per_minute"])
        key = CacheKey("coingecko_chart", asset.symbol, str(start), str(end), freq)
        try:
            entry = cached_fetch(
                cache, key,
                lambda: get_json(
                    f"{cg['base_url']}/coins/{asset.coingecko_id}/market_chart",
                    params={"vs_currency": "usd", "days": "365", "interval": "daily"},
                    limiter=limiter,
                ),
                refresh=refresh,
            )
            frame = _coingecko_frame(entry.payload)
        except IngestError as exc:
            log.warning("%s: CoinGecko failed (%s)", asset.symbol, exc)
            frame = pd.DataFrame()
        if not frame.empty:
            frame = frame.loc[
                (frame.index >= pd.Timestamp(start, tz="UTC"))
                & (frame.index <= pd.Timestamp(end, tz="UTC"))
            ]
        if len(frame) >= _MIN_USABLE_DAYS:
            frame["taker_buy_base"] = float("nan")
            frame["source"] = "coingecko"
            log.warning("%s: only %d day(s) from CoinGecko's 365-day free window",
                        asset.symbol, len(frame))
            return frame

    log.error(
        "%s: no source covers %s..%s with at least %d days. The asset is EXCLUDED "
        "from the panel and the exclusion is recorded in DATA_CARD.md. It is not "
        "back-filled, interpolated, or replaced (Rule 12.3).",
        asset.symbol, start, end, _MIN_USABLE_DAYS,
    )
    return pd.DataFrame()
