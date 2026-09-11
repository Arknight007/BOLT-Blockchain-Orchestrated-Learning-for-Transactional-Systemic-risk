"""Walk-forward, event-aware, embargoed splits - Guard 4 (BOLT_SPEC.md Section 7).

Because windows are ``lookback_days`` long and labels look ``horizon_days``
forward, a training sample whose window ends near the split boundary can have a
label determined by prices inside the test period. Dropping such samples is the
whole of Guard 4.

The assertion after construction is not negotiable:

    meta.loc[train_idx, "label_end"].max() < meta.loc[test_idx, "window_start"].min()

If it fails the split is leaking. Fix the split; never weaken the assertion.

Guard 5 lives here too: :class:`TrainOnlyScaler` refuses to transform before it
has been fitted, and records the sample count it was fitted on so a test can
assert it saw the training fold and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bolt.config import BoltConfig, Fold
from bolt.logging_setup import get_logger

log = get_logger(__name__)


class LeakageError(AssertionError):
    """A leakage guard failed. Never downgrade this to a warning."""


@dataclass(frozen=True, slots=True)
class SplitResult:
    """One walk-forward fold's row indices, plus what the embargo removed."""

    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    embargoed: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __len__(self) -> int:
        return len(self.train_idx) + len(self.test_idx)


def walk_forward_splits(
    meta: pd.DataFrame, folds: list[Fold], embargo_days: int
) -> list[SplitResult]:
    """Build embargoed expanding-window splits from window metadata.

    Args:
        meta: the frame returned by ``build_windows``, with ``window_start``,
            ``window_end`` and ``label_end``.
        folds: configured folds.
        embargo_days: must be >= lookback + horizon; the config loader already
            enforces that, and this function re-checks against the data.

    Returns:
        One :class:`SplitResult` per fold that has both training and test rows.

    Raises:
        LeakageError: if any constructed split overlaps.
    """
    required = {"window_start", "window_end", "label_end"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"walk_forward_splits: meta is missing {sorted(missing)}")

    window_start = pd.to_datetime(meta["window_start"], utc=True)
    label_end = pd.to_datetime(meta["label_end"], utc=True)
    results: list[SplitResult] = []

    for number, fold in enumerate(folds, start=1):
        train_end = pd.Timestamp(fold.train_end, tz="UTC")
        test_start = pd.Timestamp(fold.test_start, tz="UTC")
        test_end = pd.Timestamp(fold.test_end, tz="UTC")

        in_test = (window_start >= test_start) & (window_start <= test_end)

        # A training sample qualifies only if its ENTIRE span -- window and the
        # forward label period -- finishes before the embargo boundary. Using
        # window_end here instead of label_end is the classic silent leak.
        boundary = test_start - pd.Timedelta(embargo_days, "D")
        candidate = window_start <= train_end
        qualifies = candidate & (label_end < boundary)
        embargoed = int((candidate & ~qualifies).sum())

        train_idx = np.flatnonzero(qualifies.to_numpy())
        test_idx = np.flatnonzero(in_test.to_numpy())
        if len(train_idx) == 0 or len(test_idx) == 0:
            log.warning("fold %d (%s..%s): %d train / %d test rows; skipped",
                        number, fold.test_start, fold.test_end, len(train_idx), len(test_idx))
            continue

        result = SplitResult(
            name=f"fold{number}_{fold.test_start.year}",
            train_idx=train_idx,
            test_idx=test_idx,
            embargoed=embargoed,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        assert_no_overlap(meta, result)
        results.append(result)
        log.info(
            "%s: train=%d test=%d, %d sample(s) embargoed, gap %s -> %s",
            result.name, len(train_idx), len(test_idx), embargoed,
            label_end.iloc[train_idx].max().date(), window_start.iloc[test_idx].min().date(),
        )

    if not results:
        raise ValueError("walk_forward_splits produced no usable folds")
    return results


def assert_no_overlap(meta: pd.DataFrame, split: SplitResult) -> None:
    """Guard 4: the last training label must resolve before the first test window."""
    label_end = pd.to_datetime(meta["label_end"], utc=True)
    window_start = pd.to_datetime(meta["window_start"], utc=True)

    last_train_label = label_end.iloc[split.train_idx].max()
    first_test_window = window_start.iloc[split.test_idx].min()

    if not last_train_label < first_test_window:
        raise LeakageError(
            f"Guard 4 violated in {split.name}: the latest training label resolves on "
            f"{last_train_label.date()}, which is not before the earliest test window "
            f"starting {first_test_window.date()}. A training sample's outcome is "
            f"determined by prices the model is about to be tested on. Increase "
            f"split.embargo_days; do not weaken this assertion."
        )

    overlap = np.intersect1d(split.train_idx, split.test_idx)
    if overlap.size:
        raise LeakageError(
            f"Guard 4 violated in {split.name}: {overlap.size} row(s) appear in both "
            f"the training and test sets."
        )


# ---------------------------------------------------------------------------
# Guard 5 - scaler fitted on train only
# ---------------------------------------------------------------------------

class TrainOnlyScaler:
    """Standardise features using training-fold statistics only.

    Fitting on the full dataset leaks the test period's mean and variance into
    training. This class makes that mistake visible: it records
    ``n_samples_seen_`` so a test can assert it equals the training-fold size,
    and it refuses to transform before being fitted.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_samples_seen_: int = 0

    def fit(self, X: np.ndarray) -> "TrainOnlyScaler":
        if X.ndim == 3:
            flat = X.reshape(-1, X.shape[-1])
            self.n_samples_seen_ = X.shape[0]
        elif X.ndim == 2:
            flat = X
            self.n_samples_seen_ = X.shape[0]
        else:
            raise ValueError(f"expected 2D or 3D input, got shape {X.shape}")

        self.mean_ = np.nanmean(flat, axis=0)
        scale = np.nanstd(flat, axis=0)
        # A constant column has zero variance; dividing by it produces inf.
        self.scale_ = np.where(scale > 1e-12, scale, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError(
                "Guard 5: TrainOnlyScaler.transform() called before fit(). The scaler "
                "must be fitted on the training fold and only then applied to test."
            )
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
