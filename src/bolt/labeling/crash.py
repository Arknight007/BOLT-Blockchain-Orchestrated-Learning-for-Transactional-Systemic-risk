"""Crash labelling - G1 (BOLT_SPEC.md Section 5).

A label is 1 when the minimum close over ``(t, t+horizon]`` falls at least
``threshold`` below ``close[t]``. The last ``horizon`` dates are NaN and MUST be
dropped, not filled: there is not enough forward data to know the answer, and
filling them with 0 would teach the model that the end of every series is calm.

This is the substantive change from the base paper. Ke et al. predict a pin-bar
reversal - a candlestick microstructure pattern. A pin-bar is a trading signal;
a 20% drawdown is a systemic event. Different targets, different economic
meaning, different research question.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bolt.logging_setup import get_logger

log = get_logger(__name__)


def label_crashes(prices: pd.Series, threshold: float, horizon: int) -> pd.Series:
    """Binary crash label per date.

    Args:
        prices: close prices indexed by date, ascending.
        threshold: minimum decline as a positive fraction (0.20 = 20%).
        horizon: forward window length in days.

    Returns:
        Int8-backed Series aligned to ``prices.index`` with NaN for the final
        ``horizon`` dates. Stored as ``Float64`` so NaN survives; callers drop
        those rows before training.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be a fraction in (0,1), got {threshold}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices must be sorted ascending by date")

    # Minimum close over the FORWARD window, excluding today. Reversing the
    # series turns a forward-looking min into a trailing one, which is the only
    # place in this codebase a forward window is legitimate - it defines the
    # target, not a feature.
    reversed_prices = prices.iloc[::-1]
    forward_min = (
        reversed_prices.shift(1).rolling(horizon, min_periods=horizon).min().iloc[::-1]
    )

    decline = (prices - forward_min) / prices
    labels = (decline >= threshold).astype("float64")
    labels[forward_min.isna()] = np.nan
    return labels.rename("label")


def label_panel(panel: pd.DataFrame, threshold: float, horizon: int,
                price_col: str = "close") -> pd.Series:
    """Apply :func:`label_crashes` per asset across a (date, asset) panel."""
    parts = []
    for asset, group in panel.groupby(level="asset", sort=False):
        ordered = group.sort_index(level="date")
        series = label_crashes(
            ordered[price_col].droplevel("asset"), threshold, horizon
        )
        series.index = ordered.index
        parts.append(series)
    labels = pd.concat(parts).reindex(panel.index)
    rate = labels.mean()
    log.info(
        "labels: %d/%d positive (%.2f%%), %d dropped for insufficient forward data",
        int(labels.sum(skipna=True)), int(labels.notna().sum()),
        100.0 * rate if pd.notna(rate) else float("nan"), int(labels.isna().sum()),
    )
    return labels


def label_sensitivity_table(
    panel: pd.DataFrame,
    grid: dict,
    episodes: list,
    price_col: str = "close",
) -> pd.DataFrame:
    """Positive-class rate and episode coverage for every (threshold, horizon) pair.

    This table is what makes the crash definition defensible rather than
    arbitrary. A panel that asks "why 20% and 14 days?" gets a table showing what
    every other choice would have produced, not an assertion.
    """
    rows = []
    for threshold in grid["thresholds"]:
        for horizon in grid["horizons"]:
            labels = label_panel(panel, threshold, horizon, price_col)
            valid = labels.dropna()
            covered = episode_coverage(labels, episodes)
            rows.append({
                "threshold": threshold,
                "horizon_days": horizon,
                "positive_rate": float(valid.mean()) if len(valid) else np.nan,
                "n_positive": int(valid.sum()),
                "n_labelled": int(len(valid)),
                "episodes_covered": sum(covered.values()),
                "episodes_total": len(episodes),
                "uncovered": ", ".join(n for n, ok in covered.items() if not ok) or "-",
            })
    table = pd.DataFrame(rows)
    log.info("sensitivity table: %d (threshold, horizon) combinations", len(table))
    return table


def episode_coverage(labels: pd.Series, episodes: list) -> dict[str, bool]:
    """Whether each crisis episode contains at least one positive label.

    Validation requirement (Section 5): the default parameters must produce
    positives overlapping every configured episode. If they do not, the crash
    definition is wrong - surface that, do not tune around it.
    """
    dates = labels.index.get_level_values("date")
    result: dict[str, bool] = {}
    for episode in episodes:
        start = pd.Timestamp(episode.start, tz="UTC")
        end = pd.Timestamp(episode.end, tz="UTC")
        # A warning is useful BEFORE the episode, so look back one horizon:
        # a label set on day t fires because of what happens after t.
        window = (dates >= start - pd.Timedelta(days=45)) & (dates <= end)
        block = labels[window].dropna()
        result[episode.name] = bool(len(block) and block.sum() > 0)
    return result


def assert_episodes_covered(labels: pd.Series, episodes: list) -> None:
    """Raise when the configured labelling misses a known crisis."""
    covered = episode_coverage(labels, episodes)
    missed = [name for name, ok in covered.items() if not ok]
    if missed:
        raise ValueError(
            f"the default labelling produces no positive labels for {missed}. "
            f"These are known severe declines, so a definition that misses them is "
            f"wrong (BOLT_SPEC.md Section 5). Surface this; do not tune around it."
        )
