"""Sentiment and event ingestion (BOLT_SPEC.md Section 3).

The publication timestamp is load-bearing. Guard 2 rejects any headline whose
``published_at`` is later than the feature date it would contribute to: text
published *after* a crash trivially predicts it, and that is the single most
common silent failure in this literature.

Sources, in order of preference:

* **alternative.me Fear and Greed Index** - free, keyless, daily, with history
  back to 2018. A real composite of volatility, momentum, social volume and
  market dominance, published once per day.
* **CryptoPanic** - headline text with publication timestamps. Requires an API
  key; when one is absent the pipeline records the gap rather than inventing
  headlines or treating silence as neutral sentiment.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from bolt.config import BoltConfig
from bolt.ingest.cache import CacheKey, DiskCache, cached_fetch
from bolt.ingest.http import RateLimiter, get_json
from bolt.logging_setup import get_logger

log = get_logger(__name__)

FEAR_GREED_URL = "https://api.alternative.me/fng/"
HEADLINE_COLUMNS = ["published_at", "title", "source", "asset"]


def fetch_fear_greed(cfg: BoltConfig, cache: DiskCache, *, refresh: bool = False) -> pd.Series:
    """Daily Fear and Greed index, 0-100, indexed by tz-aware UTC timestamp."""
    key = CacheKey("fear_greed", "MARKET", str(cfg.start_date), str(cfg.end_date), "1d")
    entry = cached_fetch(
        cache,
        key,
        lambda: get_json(
            FEAR_GREED_URL, params={"limit": 0, "format": "json"}, limiter=RateLimiter(30)
        ),
        refresh=refresh,
        url=FEAR_GREED_URL,
    )
    rows = (entry.payload or {}).get("data") or []
    if not rows:
        raise ValueError("Fear and Greed returned no data; refusing to continue (Rule 12.3)")

    index = pd.to_datetime([int(r["timestamp"]) for r in rows], unit="s", utc=True)
    series = pd.Series([float(r["value"]) for r in rows], index=index, name="fear_greed")
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]
    log.info(
        "fear and greed: %d days, %s..%s",
        len(series), series.index.min().date(), series.index.max().date(),
    )
    return series


def _normalise_timestamp(value: Any) -> pd.Timestamp | None:
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _headline_frame(payload: Any) -> pd.DataFrame:
    rows = (payload or {}).get("results") or []
    records = []
    for row in rows:
        stamp = _normalise_timestamp(row.get("published_at") or row.get("created_at"))
        if stamp is None:
            continue  # Guard 2: a headline without a timestamp cannot be placed in time
        records.append(
            {
                "published_at": stamp,
                "title": row.get("title", ""),
                "source": (row.get("source") or {}).get("domain", "cryptopanic"),
                "asset": ",".join(c.get("code", "") for c in (row.get("currencies") or [])),
            }
        )
    frame = pd.DataFrame(records, columns=HEADLINE_COLUMNS)
    dropped = len(rows) - len(frame)
    if dropped:
        log.warning("dropped %d headline(s) with no publication timestamp (Guard 2)", dropped)
    if frame.empty:
        return frame
    return frame.sort_values("published_at").reset_index(drop=True)


def fetch_headlines(cfg: BoltConfig, cache: DiskCache, *, refresh: bool = False) -> pd.DataFrame:
    """Timestamped headlines from CryptoPanic.

    Returns an EMPTY frame with the correct columns when no API key is set. The
    caller records that gap in the data card; it never fabricates headlines and
    never treats absence as neutral sentiment.
    """
    settings = cfg.section("ingest")["sources"]["cryptopanic"]
    token = os.environ.get(settings["api_key_env_var"], "").strip()
    if not settings["enabled"] or not token:
        log.warning(
            "CryptoPanic: no API key in $%s - headline features will be unavailable and "
            "recorded as a gap in DATA_CARD.md (Rule 12.3: no silent zero-fill)",
            settings["api_key_env_var"],
        )
        return pd.DataFrame(columns=HEADLINE_COLUMNS)

    key = CacheKey("cryptopanic", "MARKET", str(cfg.start_date), str(cfg.end_date), "1d")
    entry = cached_fetch(
        cache,
        key,
        lambda: get_json(
            settings["base_url"] + "/posts/",
            params={"auth_token": token, "public": "true"},
            limiter=RateLimiter(settings["rate_limit_per_minute"]),
        ),
        refresh=refresh,
    )
    return _headline_frame(entry.payload)
