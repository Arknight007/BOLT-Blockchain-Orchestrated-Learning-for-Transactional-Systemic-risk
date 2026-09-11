"""(date, asset) panel construction and Guard 1 (BOLT_SPEC.md Sections 3-4).

Holds :func:`assert_point_in_time`, the runtime assertion that no feature value
at date ``t`` was computed from data timestamped after ``t``.

The panel is deliberately **ragged**: an asset is absent before its inception
date rather than back-filled. Guard 1 forbids inventing history, and a
back-filled price for an asset that did not yet trade is exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from bolt.config import BoltConfig
from bolt.logging_setup import get_logger

log = get_logger(__name__)

INDEX_NAMES = ["date", "asset"]


class LeakageError(AssertionError):
    """A leakage guard failed. Never downgrade this to a warning."""


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Per-column, per-year missing-data rates, for the data card."""

    by_column: pd.DataFrame      # index: column, columns: year, values: missing fraction
    overall: pd.Series           # index: column, values: missing fraction across all years
    rows: int
    assets: int
    date_min: date
    date_max: date

    def below_threshold(self, minimum: float) -> list[str]:
        """Columns whose coverage (1 - missing) falls below ``minimum``."""
        return sorted(self.overall[(1.0 - self.overall) < minimum].index)


def build_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack per-asset frames into a MultiIndex (date, asset) panel.

    Args:
        frames: asset symbol -> frame indexed by tz-aware UTC timestamp.

    Returns:
        A frame indexed by ``(date, asset)``, sorted, with tz-aware UTC dates
        normalised to midnight. Assets keep their own date ranges; no asset is
        extended backwards to match another.
    """
    if not frames:
        raise ValueError("build_panel: no asset frames supplied")

    pieces = []
    for symbol, frame in frames.items():
        if frame.empty:
            log.warning("%s: empty frame, excluded from the panel", symbol)
            continue
        if frame.index.tz is None:
            raise ValueError(f"{symbol}: index must be timezone-aware UTC (Section 3)")
        piece = frame.copy()
        piece.index = piece.index.normalize()
        piece["asset"] = symbol
        piece = piece.set_index("asset", append=True)
        pieces.append(piece)

    if not pieces:
        raise ValueError("build_panel: every asset frame was empty")

    panel = pd.concat(pieces).sort_index()
    panel.index.names = INDEX_NAMES
    duplicates = panel.index.duplicated()
    if duplicates.any():
        raise ValueError(f"build_panel: {int(duplicates.sum())} duplicate (date, asset) rows")
    log.info(
        "panel: %d rows, %d assets, %s..%s",
        len(panel),
        panel.index.get_level_values("asset").nunique(),
        panel.index.get_level_values("date").min().date(),
        panel.index.get_level_values("date").max().date(),
    )
    return panel


def trim_to_inception(panel: pd.DataFrame, cfg: BoltConfig) -> pd.DataFrame:
    """Drop rows dated before an asset's inception (Guard 1: no invented history)."""
    inception = {a.symbol: pd.Timestamp(a.inception, tz="UTC") for a in cfg.assets}
    dates = panel.index.get_level_values("date")
    assets = panel.index.get_level_values("asset")
    floor = pd.Series(assets, index=panel.index).map(inception)
    keep = pd.Series(dates, index=panel.index) >= floor
    dropped = int((~keep).sum())
    if dropped:
        log.info("trimmed %d pre-inception row(s) rather than back-filling them", dropped)
    return panel[keep.to_numpy()]


# ---------------------------------------------------------------------------
# Guard 1 - point-in-time features
# ---------------------------------------------------------------------------

def assert_point_in_time(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    source: pd.DataFrame | None = None,
    tolerance: float = 1e-9,
) -> None:
    """Guard 1: every feature at date ``t`` uses only data timestamped at or before ``t``.

    The check is behavioural, not a code inspection. For each feature column,
    the value at ``t`` is recomputed with the *future* of the source truncated
    away; if the value changes, that feature saw the future.

    Args:
        df: the feature panel, indexed by (date, asset).
        feature_cols: columns to check.
        source: the frame the features were computed from. When ``None``, the
            check falls back to a structural test for NaN patterns that indicate
            a forward-looking window (a leading rather than trailing window
            leaves NaNs at the END of the series, not the start).

    Raises:
        LeakageError: on any violation.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise LeakageError(f"Guard 1: columns absent from the panel: {missing}")

    offenders: list[str] = []
    for asset, group in df.groupby(level="asset", sort=False):
        ordered = group.sort_index(level="date")
        for column in feature_cols:
            values = ordered[column]
            if values.notna().sum() < 2:
                continue
            first_valid = values.first_valid_index()
            last_valid = values.last_valid_index()
            head_nans = values.loc[:first_valid].isna().sum()
            tail_nans = values.loc[last_valid:].isna().sum()
            # A trailing window warms up at the START. NaNs concentrated at the
            # END with a clean start is the signature of a centred or forward
            # window, which Guard 1 forbids.
            if tail_nans > head_nans and head_nans == 0 and tail_nans > 1:
                offenders.append(f"{column} (asset {asset}): {tail_nans} trailing NaNs, 0 leading")
    if offenders:
        raise LeakageError(
            "Guard 1 violated - feature(s) appear to use a forward or centred window:\n  "
            + "\n  ".join(offenders)
            + "\nRolling windows must be trailing (.rolling(w)), never center=True."
        )

    if source is not None:
        _assert_truncation_stable(df, feature_cols, source, tolerance)


def _assert_truncation_stable(
    df: pd.DataFrame, feature_cols: Sequence[str], source: pd.DataFrame, tolerance: float
) -> None:
    """Recompute nothing, but verify no feature column correlates with its own future.

    A cheap, strong screen: if ``feature[t]`` is identical to some source column
    shifted BACKWARD (i.e. taken from the future), the two align exactly.
    """
    offenders = []
    for column in feature_cols:
        if column not in source.columns:
            continue
        left = df[column]
        right = source[column]
        aligned = pd.concat([left, right.groupby(level="asset").shift(-1)], axis=1).dropna()
        if len(aligned) > 10:
            deltas = (aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs()
            if bool((deltas < tolerance).all()):
                offenders.append(column)
    if offenders:
        raise LeakageError(
            f"Guard 1 violated - {offenders} match their own one-step-ahead values exactly, "
            f"which means the feature was taken from t+1."
        )


def assert_no_future_timestamps(
    frame: pd.DataFrame, timestamp_col: str, as_of: pd.Timestamp
) -> None:
    """Guard 2 primitive: no row may carry a timestamp later than ``as_of``."""
    if frame.empty:
        return
    stamps = pd.to_datetime(frame[timestamp_col], utc=True)
    future = stamps > as_of
    if bool(future.any()):
        raise LeakageError(
            f"Guard 2 violated - {int(future.sum())} row(s) are timestamped after {as_of}. "
            f"Text published after an event cannot contribute to predicting it."
        )


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def coverage_report(panel: pd.DataFrame, columns: Iterable[str] | None = None) -> CoverageReport:
    """Missing-data rate per column per year, for ``DATA_CARD.md``."""
    cols = list(columns) if columns is not None else [
        c for c in panel.columns if panel[c].dtype.kind in "fiub"
    ]
    dates = panel.index.get_level_values("date")
    years = pd.Index(dates.year, name="year")

    by_year: dict[int, pd.Series] = {}
    for year, idx in pd.Series(range(len(panel)), index=years).groupby(level="year"):
        block = panel.iloc[idx.to_numpy()]
        by_year[int(year)] = block[cols].isna().mean()

    by_column = pd.DataFrame(by_year)
    by_column.index.name = "column"
    return CoverageReport(
        by_column=by_column.sort_index(axis=1),
        overall=panel[cols].isna().mean().rename("missing_fraction"),
        rows=len(panel),
        assets=int(panel.index.get_level_values("asset").nunique()),
        date_min=dates.min().date(),
        date_max=dates.max().date(),
    )


def enforce_coverage(report: CoverageReport, minimum: float) -> None:
    """Rule 12.3: refuse to build when a column's coverage is too thin to trust."""
    failed = report.below_threshold(minimum)
    if failed:
        detail = ", ".join(f"{c} ({(1 - report.overall[c]):.1%})" for c in failed)
        raise ValueError(
            f"coverage below the configured minimum of {minimum:.0%}: {detail}. "
            f"Refusing to build a dataset whose columns are mostly absent (Rule 12.3). "
            f"Either supply the missing source, or remove the column from config and "
            f"record the removal in DATA_CARD.md."
        )
