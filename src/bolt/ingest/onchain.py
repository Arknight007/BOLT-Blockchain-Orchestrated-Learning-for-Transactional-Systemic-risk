"""On-chain metrics (BOLT_SPEC.md Section 3).

No Etherscan key is configured for this build, so the four ledger-native metrics
are implemented as **documented proxies**, and every one of them is registered in
``bolt.ingest.provenance`` with what it stands in for and where it falls short.
``DATA_CARD.md`` reproduces that table. Nothing is silently substituted.

Stablecoin supply is the exception: aggregate USDT and DAI market capitalisation
is a real, directly observed quantity, and it is the single most informative free
on-chain-adjacent series for crash work. Stablecoin supply contracting while
prices fall is the signature of capital leaving the system rather than rotating
within it.
"""

from __future__ import annotations

import pandas as pd

from bolt.config import Asset, BoltConfig
from bolt.ingest.cache import DiskCache
from bolt.logging_setup import get_logger

log = get_logger(__name__)

ONCHAIN_RAW_COLUMNS = ["taker_net_ratio", "mean_trade_size", "trade_count", "close", "quote_volume"]


def build_onchain_raw(
    cfg: BoltConfig,
    asset: Asset,
    ohlcv: pd.DataFrame,
    cache: DiskCache,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Raw on-chain-proxy inputs for one asset, aligned to the OHLCV index.

    The feature layer turns these into the five configured on-chain features.
    Keeping the raw quantities separate means each proxy definition lives in
    exactly one place, next to its provenance entry.
    """
    frame = pd.DataFrame(index=ohlcv.index)

    # Aggressive-flow imbalance: taker buys minus taker sells over total volume.
    # Bounded in [-1, 1]; negative means sellers were hitting bids all day.
    if "taker_buy_base" in ohlcv.columns and ohlcv["taker_buy_base"].notna().any():
        volume = ohlcv["volume"].replace(0.0, float("nan"))
        taker_sell = volume - ohlcv["taker_buy_base"]
        frame["taker_net_ratio"] = (ohlcv["taker_buy_base"] - taker_sell) / volume
    else:
        frame["taker_net_ratio"] = float("nan")
        log.warning(
            "%s: no taker-flow data (CoinGecko-sourced asset); exchange_netflow_7 "
            "will be missing and carded as such, never zero-filled",
            asset.symbol,
        )

    # Mean trade size is the whale-activity proxy; trade count the activity proxy.
    if "trades" in ohlcv.columns and ohlcv["trades"].notna().any():
        trades = pd.to_numeric(ohlcv["trades"], errors="coerce").replace(0, float("nan"))
        frame["mean_trade_size"] = ohlcv["quote_volume"] / trades
        frame["trade_count"] = trades.astype(float)
    else:
        frame["mean_trade_size"] = float("nan")
        frame["trade_count"] = float("nan")

    # NVT's denominator. CoinGecko's free tier no longer serves historical market
    # capitalisation, so the ratio is built from price and turnover instead; the
    # provenance registry records exactly what that changes.
    frame["close"] = ohlcv["close"]
    frame["quote_volume"] = ohlcv["quote_volume"]
    return frame


def fetch_stablecoin_supply(
    cfg: BoltConfig, cache: DiskCache, *, refresh: bool = False
) -> pd.Series:
    """Aggregate circulating USD value of all USD-pegged stablecoins, daily.

    Sourced from DefiLlama rather than summing our two universe stablecoins: the
    market-wide aggregate is the better measure of capital entering or leaving
    the crypto system, and it is a real observation rather than a proxy.
    """
    from bolt.ingest.defillama import fetch_stablecoin_supply as _llama_supply

    return _llama_supply(cfg, cache, refresh=refresh)
