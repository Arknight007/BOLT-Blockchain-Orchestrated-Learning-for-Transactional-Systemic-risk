"""Nearest historical regime retrieval (BOLT_SPEC.md Section 8).

Answers "when did the market last look like this, and what happened next?".

``exclude_days`` is load-bearing. Without it the nearest neighbours of today are
yesterday and the day before, because consecutive 30-day windows share 29 days of
data. A window from last week is not a historical precedent, it is the same
observation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bolt.logging_setup import get_logger

log = get_logger(__name__)


def _standardise(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (matrix - mean) / scale, mean, scale


def _cosine_similarity(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    numerator = matrix @ vector
    denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    return numerator / np.where(denominator > 1e-12, denominator, 1.0)


def _mahalanobis_similarity(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Negative Mahalanobis distance, rescaled to a similarity in (0, 1]."""
    covariance = np.cov(matrix, rowvar=False)
    covariance += np.eye(covariance.shape[0]) * 1e-6      # ridge for invertibility
    inverse = np.linalg.pinv(covariance)
    deltas = matrix - vector
    distances = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", deltas, inverse, deltas), 0.0))
    return 1.0 / (1.0 + distances)


def nearest_historical_analogue(
    current_vector: np.ndarray,
    history_matrix: np.ndarray,
    meta: pd.DataFrame,
    prices: pd.Series | None = None,
    k: int = 3,
    exclude_days: int = 90,
    metric: str = "cosine",
    as_of: pd.Timestamp | None = None,
    outcome_window_days: int = 30,
) -> list[dict]:
    """Find the ``k`` most similar historical windows and report what followed.

    Args:
        current_vector: the query window, flattened or per-feature.
        history_matrix: ``(N, D)`` historical windows in the same space.
        meta: window metadata aligned to ``history_matrix`` rows.
        prices: close prices indexed by date, for the outcome. When absent the
            outcome is reported as unavailable rather than guessed.
        k: how many analogues to return.
        exclude_days: minimum separation from ``as_of``, so overlapping windows
            are not returned as precedents.
        metric: ``cosine`` or ``mahalanobis``.
        as_of: the query date; defaults to the latest date in ``meta``.
        outcome_window_days: how far forward to measure what happened next.

    Returns:
        ``[{date, asset, similarity, what_happened_next_30d, ...}, ...]``
    """
    if len(history_matrix) != len(meta):
        raise ValueError(
            f"history_matrix has {len(history_matrix)} rows but meta has {len(meta)}"
        )

    dates = pd.to_datetime(meta["window_end"], utc=True)
    if as_of is None:
        as_of = dates.max()
    else:
        as_of = pd.Timestamp(as_of)
        as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")

    # Exclude anything within exclude_days of the query: an overlapping window is
    # the same observation, not a precedent.
    separation = (as_of - dates).abs()
    eligible = (separation >= pd.Timedelta(exclude_days, "D")).to_numpy()
    if not eligible.any():
        log.warning("no window is at least %d days from %s; no analogue returned",
                    exclude_days, as_of.date())
        return []

    standardised, mean, scale = _standardise(history_matrix)
    query = (np.asarray(current_vector, dtype=float).ravel() - mean) / scale

    if metric == "cosine":
        similarity = _cosine_similarity(query, standardised)
    elif metric == "mahalanobis":
        similarity = _mahalanobis_similarity(query, standardised)
    else:
        raise ValueError(f"unknown metric {metric!r}; expected 'cosine' or 'mahalanobis'")

    similarity = np.where(eligible, similarity, -np.inf)
    order = np.argsort(similarity)[::-1][:k]

    results = []
    for position in order:
        if not np.isfinite(similarity[position]):
            continue
        row = meta.iloc[position]
        results.append({
            "date": pd.Timestamp(row["window_end"]).date().isoformat(),
            "asset": row["asset"],
            "similarity": float(similarity[position]),
            "what_happened_next_30d": _outcome(
                prices, row["asset"], _as_utc(row["window_end"]), outcome_window_days
            ),
        })
    return results


def _as_utc(value) -> pd.Timestamp:
    """Normalise to a tz-aware UTC timestamp, whatever form it arrives in."""
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _outcome(prices: pd.Series | None, asset: str, date: pd.Timestamp, days: int) -> dict:
    """What the price did in the ``days`` after ``date``.

    Returns ``{"available": False}`` when prices were not supplied. It never
    invents an outcome: an analogue whose consequence is unknown is still a
    useful analogue, and a fabricated consequence is not.
    """
    if prices is None:
        return {"available": False, "reason": "price series not supplied"}
    try:
        series = prices.xs(asset, level="asset") if isinstance(prices.index, pd.MultiIndex) else prices
    except KeyError:
        return {"available": False, "reason": f"no prices for {asset}"}

    series = series.sort_index()
    window = series.loc[date: date + pd.Timedelta(days, "D")]
    if len(window) < 2:
        return {"available": False, "reason": "insufficient forward data"}

    start_price = float(window.iloc[0])
    minimum = float(window.min())
    end_price = float(window.iloc[-1])
    return {
        "available": True,
        "max_drawdown": (start_price - minimum) / start_price,
        "return": (end_price - start_price) / start_price,
        "days_observed": int(len(window)),
    }


def summarise_analogues(analogues: list[dict]) -> str:
    """One-line human summary, used by the explanation agent."""
    if not analogues:
        return "No historical analogue was found outside the exclusion window."
    parts = []
    for item in analogues:
        outcome = item["what_happened_next_30d"]
        if outcome.get("available"):
            parts.append(
                f"{item['date']} ({item['asset']}, similarity {item['similarity']:.2f}) "
                f"was followed by a {outcome['max_drawdown']:.1%} drawdown"
            )
        else:
            parts.append(
                f"{item['date']} ({item['asset']}, similarity {item['similarity']:.2f}); "
                f"outcome unavailable"
            )
    return "; ".join(parts)
