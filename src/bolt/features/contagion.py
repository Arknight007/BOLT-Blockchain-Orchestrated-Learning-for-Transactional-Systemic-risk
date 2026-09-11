"""Cross-asset contagion features - G2 (BOLT_SPEC.md Section 5).

The base paper models Bitcoin in isolation and names cross-chain dynamics as its
own future work. This module is that work.

Rising mean pairwise correlation means diversification is disappearing: assets
that normally move independently start moving together, which is a recognised
stress signature in equity and credit markets and is the mechanism by which a
single failure becomes systemic. Eigenvector centrality on the correlation graph
asks a sharper question - not "is everything correlated?" but "which asset is at
the centre of the correlation structure right now?".

All windows are trailing 30 days (Guard 1). The correlation graph is cached per
date because eigenvector centrality is the expensive step.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd

from bolt.logging_setup import get_logger

log = get_logger(__name__)

CONTAGION_COLUMNS = [
    "mean_pairwise_corr_30", "corr_dispersion_30", "btc_lead_lag_5", "eigen_centrality_30",
]

MIN_ASSETS_FOR_CORRELATION = 3


def returns_matrix(panel: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Wide (date x asset) log-return matrix from a long (date, asset) panel."""
    wide = panel[price_col].unstack("asset").sort_index()
    return np.log(wide / wide.shift(1))


def _offdiagonal(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return matrix[mask]


def eigen_centrality(corr: pd.DataFrame, min_edge_weight: float) -> pd.Series:
    """Eigenvector centrality of the absolute-correlation graph.

    Edges below ``min_edge_weight`` are dropped so the graph reflects genuine
    co-movement rather than the dense noise floor that makes every node equally
    central.
    """
    graph = nx.Graph()
    graph.add_nodes_from(corr.columns)
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            weight = abs(float(corr.loc[a, b]))
            if np.isfinite(weight) and weight >= min_edge_weight:
                graph.add_edge(a, b, weight=weight)

    if graph.number_of_edges() == 0:
        return pd.Series(0.0, index=corr.columns)
    try:
        scores = nx.eigenvector_centrality_numpy(graph, weight="weight")
    except (nx.NetworkXException, np.linalg.LinAlgError):
        # Disconnected or degenerate graph: fall back to weighted degree, which
        # answers the same question less sharply rather than failing the run.
        degree = dict(graph.degree(weight="weight"))
        total = sum(degree.values()) or 1.0
        scores = {k: v / total for k, v in degree.items()}
    return pd.Series(scores).reindex(corr.columns).fillna(0.0)


def lead_lag(
    leader: pd.Series, follower: pd.Series, max_lag: int
) -> float:
    """Lag in 1..max_lag at which ``leader`` best predicts ``follower``.

    Returns the signed lag weighted by its correlation, so a strong one-day lead
    and a weak five-day lead are not conflated. Zero means no usable lead.
    """
    best_lag, best_corr = 0, 0.0
    for lag in range(1, max_lag + 1):
        shifted = leader.shift(lag)
        pair = pd.concat([shifted, follower], axis=1).dropna()
        if len(pair) < max_lag + 2:
            continue
        corr = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
        if np.isfinite(corr) and abs(corr) > abs(best_corr):
            best_lag, best_corr = lag, corr
    return float(best_lag) * best_corr


def compute_contagion(
    panel: pd.DataFrame,
    params: dict,
    primary_asset: str = "BTC",
) -> pd.DataFrame:
    """Contagion features for every (date, asset) in the panel.

    ``mean_pairwise_corr_30``, ``corr_dispersion_30`` and ``btc_lead_lag_5`` are
    market-wide: identical for every asset on a given date. ``eigen_centrality_30``
    is per-asset, since it asks where each asset sits in the structure.

    Returns:
        A frame indexed like ``panel`` with :data:`CONTAGION_COLUMNS`.
    """
    window = params["contagion_window"]
    max_lag = params["lead_lag_max_lag"]
    min_edge = params["corr_graph_min_edge_weight"]

    returns = returns_matrix(panel)
    dates = returns.index
    out = pd.DataFrame(index=panel.index, columns=CONTAGION_COLUMNS, dtype=float)

    market_rows: dict[pd.Timestamp, tuple[float, float, float]] = {}
    centrality_rows: dict[pd.Timestamp, pd.Series] = {}

    for position in range(window, len(dates)):
        date = dates[position]
        block = returns.iloc[position - window + 1: position + 1]
        usable = block.dropna(axis=1, thresh=max(3, window // 2))
        if usable.shape[1] < MIN_ASSETS_FOR_CORRELATION:
            continue

        corr = usable.corr()
        values = _offdiagonal(corr.to_numpy())
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue

        lead = 0.0
        if primary_asset in usable.columns:
            others = [c for c in usable.columns if c != primary_asset]
            if others:
                lags = [lead_lag(usable[primary_asset], usable[c], max_lag) for c in others]
                lead = float(np.nanmean(lags)) if lags else 0.0

        market_rows[date] = (float(values.mean()), float(values.std()), lead)
        centrality_rows[date] = eigen_centrality(corr, min_edge)

    if not market_rows:
        log.warning("contagion: no date had %d usable assets; columns left missing",
                    MIN_ASSETS_FOR_CORRELATION)
        return out

    market = pd.DataFrame.from_dict(
        market_rows, orient="index",
        columns=["mean_pairwise_corr_30", "corr_dispersion_30", "btc_lead_lag_5"],
    )
    centrality = pd.DataFrame(centrality_rows).T

    panel_dates = out.index.get_level_values("date")
    panel_assets = out.index.get_level_values("asset")
    for column in ["mean_pairwise_corr_30", "corr_dispersion_30", "btc_lead_lag_5"]:
        out[column] = market[column].reindex(panel_dates).to_numpy()

    stacked = centrality.stack()
    stacked.index.names = ["date", "asset"]
    out["eigen_centrality_30"] = stacked.reindex(out.index).to_numpy()

    log.info("contagion: computed over %d date(s), %d asset(s)",
             len(market_rows), centrality.shape[1])
    return out
