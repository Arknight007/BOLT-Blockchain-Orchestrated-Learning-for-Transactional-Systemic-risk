"""On-chain features (BOLT_SPEC.md Section 5).

Built from the raw proxy inputs assembled in ``ingest/onchain.py``. Each output
column has a matching entry in ``bolt.ingest.provenance`` stating whether it is
a real observation or a documented proxy, and that table is reproduced verbatim
in ``DATA_CARD.md``.

All windows are trailing (Guard 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bolt.logging_setup import get_logger

log = get_logger(__name__)

ONCHAIN_COLUMNS = [
    "exchange_netflow_7", "whale_tx_count", "active_addresses",
    "stablecoin_netflow", "nvt_ratio",
]


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return (series - mean) / std.replace(0.0, np.nan)


def compute_onchain(
    raw: pd.DataFrame,
    stablecoin_supply: pd.Series,
    params: dict,
) -> pd.DataFrame:
    """The five configured on-chain features for one asset.

    Args:
        raw: output of ``ingest.onchain.build_onchain_raw`` for this asset.
        stablecoin_supply: aggregate USDT + DAI market cap, reindexed to ``raw``.
        params: ``features.params`` from the configuration.
    """
    window = params["onchain_netflow_window"]
    out = pd.DataFrame(index=raw.index)

    # Aggressive selling pressure, summed over a trailing week. Sign convention:
    # POSITIVE means net coins flowing toward sellers, i.e. rising crash risk.
    out["exchange_netflow_7"] = (
        (-raw["taker_net_ratio"]).rolling(window, min_periods=window).mean()
    )

    # Whale proxy: how unusual today's mean trade size is against its own history.
    out["whale_tx_count"] = _rolling_z(raw["mean_trade_size"], params["contagion_window"])

    # Activity proxy, log-scaled because trade counts span orders of magnitude.
    out["active_addresses"] = np.log1p(raw["trade_count"])

    # Real signal: stablecoin supply contracting means capital is leaving.
    supply = stablecoin_supply.reindex(raw.index)
    out["stablecoin_netflow"] = supply.pct_change(window, fill_method=None)

    # Valuation to turnover. Canonical NVT divides network value by on-chain
    # settled value; here price stands in for network value and exchange turnover
    # for settlement. Within one asset's own history, supply moves slowly enough
    # that price tracks market capitalisation up to a slowly-varying factor --
    # which is why this is a per-asset time-series feature and must not be
    # compared ACROSS assets. The provenance registry states the limitation.
    nvt_window = params["nvt_window"]
    turnover = raw["quote_volume"].rolling(nvt_window, min_periods=nvt_window).mean()
    out["nvt_ratio"] = np.log(
        raw["close"] / turnover.replace(0.0, np.nan).replace(0, np.nan)
    )

    return out[ONCHAIN_COLUMNS]
