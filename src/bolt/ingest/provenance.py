"""Per-column data provenance (BOLT_SPEC.md Section 3).

The spec is explicit: where a metric is unavailable free, it is implemented as a
DOCUMENTED proxy and recorded in ``data/processed/DATA_CARD.md`` -- never a
silent substitution. This module is the record that makes that promise checkable.

A panel will ask "is this real on-chain data?". The answer must be a table, not
a recollection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["Provenance", "ColumnSource", "REGISTRY", "register", "describe"]


class Provenance(str, Enum):
    """How a column's values came to exist."""

    PRIMARY = "PRIMARY"    # straight from the venue or ledger that produced it
    DERIVED = "DERIVED"    # a deterministic function of PRIMARY columns
    PROXY = "PROXY"        # a stand-in for an unavailable metric -- must state why


@dataclass(frozen=True, slots=True)
class ColumnSource:
    column: str
    source: str
    provenance: Provenance
    definition: str
    proxy_for: str | None = None
    limitation: str | None = None

    def __post_init__(self) -> None:
        if self.provenance is Provenance.PROXY and not (self.proxy_for and self.limitation):
            raise ValueError(
                f"{self.column}: a PROXY column must declare proxy_for and limitation. "
                f"An undocumented substitution is exactly what the data card exists to prevent."
            )


REGISTRY: dict[str, ColumnSource] = {}


def register(source: ColumnSource) -> ColumnSource:
    REGISTRY[source.column] = source
    return source


def describe(column: str) -> ColumnSource | None:
    return REGISTRY.get(column)


# --- market: primary records from the venue where trades occurred -----------
for _col, _desc in [
    ("open", "first trade price of the UTC day"),
    ("high", "highest trade price of the UTC day"),
    ("low", "lowest trade price of the UTC day"),
    ("close", "last trade price of the UTC day"),
    ("volume", "base-asset volume traded"),
    ("quote_volume", "quote-asset (USDT) volume traded"),
    ("trades", "count of individual trades executed"),
]:
    register(ColumnSource(_col, "Binance public API (klines)", Provenance.PRIMARY, _desc))

register(ColumnSource(
    "stablecoin_supply", "DefiLlama stablecoins API", Provenance.PRIMARY,
    "aggregate circulating USD value of all USD-pegged stablecoins",
))
register(ColumnSource(
    "taker_buy_base", "Binance public API (klines)", Provenance.PRIMARY,
    "base volume where the buyer was the taker -- i.e. aggressive buying",
))

# --- on-chain: proxies, because no Etherscan key is configured --------------
register(ColumnSource(
    "exchange_netflow_7", "Binance public API (taker flow)", Provenance.PROXY,
    "7-day sum of (taker buy volume - taker sell volume) divided by total volume",
    proxy_for="net ERC-20/BTC transfer volume into labelled exchange wallets",
    limitation=(
        "Measures aggressive order flow ON an exchange, not deposits INTO one. "
        "It captures the same economic pressure (willingness to sell at market) "
        "but misses coins moved to an exchange and not yet sold, which is the "
        "leading half of the true signal."
    ),
))
register(ColumnSource(
    "whale_tx_count", "Binance public API (klines)", Provenance.PROXY,
    "z-score of mean trade size (quote_volume / trades) over a trailing window",
    proxy_for="count of on-chain transfers above a large-value threshold",
    limitation=(
        "A rising mean trade size indicates large actors are active, but cannot "
        "distinguish one whale from many co-ordinated mid-size traders, and sees "
        "only exchange activity rather than the whole ledger."
    ),
))
register(ColumnSource(
    "active_addresses", "Binance public API (klines)", Provenance.PROXY,
    "daily count of executed trades, as a network-activity level",
    proxy_for="count of distinct addresses transacting on-chain that day",
    limitation=(
        "Exchange trade count moves with on-chain activity but is dominated by "
        "high-frequency participants and excludes all off-exchange settlement. "
        "Correlated with the true metric, not equal to it."
    ),
))
register(ColumnSource(
    "stablecoin_netflow", "DefiLlama stablecoins API", Provenance.DERIVED,
    "7-day fractional change in the aggregate circulating USD value of all "
    "USD-pegged stablecoins",
))
register(ColumnSource(
    "nvt_ratio", "Binance (price and turnover)", Provenance.PROXY,
    "log of close price divided by trailing mean quote volume",
    proxy_for="network value to on-chain transaction volume",
    limitation=(
        "Two substitutions, both forced by CoinGecko's free tier refusing "
        "historical market capitalisation (error 10012, 365-day ceiling). Price "
        "stands in for network value, and exchange turnover for on-chain settled "
        "value. Within one asset's history supply moves slowly, so price tracks "
        "market cap up to a slowly-varying factor -- but the resulting level is "
        "NOT comparable across assets, only across time within an asset."
    ),
))

# --- sentiment -------------------------------------------------------------
register(ColumnSource(
    "fear_greed", "alternative.me Fear & Greed Index", Provenance.PRIMARY,
    "daily composite index, 0 (extreme fear) to 100 (extreme greed)",
))
