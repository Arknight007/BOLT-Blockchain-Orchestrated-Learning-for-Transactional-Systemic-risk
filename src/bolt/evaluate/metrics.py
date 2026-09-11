"""Rare-event metrics and the accuracy ban (BOLT_SPEC.md Section 7).

Accuracy is banned in code, not by convention. On this dataset the positive rate
is roughly 11%, so a model that predicts "no crash" every single day scores 89%
accuracy while being worthless. The base paper reports F1 rather than accuracy
for exactly this reason, and quoting accuracy would forfeit the credibility that
choice earns.

``lead_time_days`` is the metric that matters operationally: a warning that
arrives after the crash has started is not a warning.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from bolt.logging_setup import get_logger

log = get_logger(__name__)

FORBIDDEN = {"accuracy", "accuracy_score", "acc", "balanced_accuracy"}

METRIC_NAMES = [
    "pr_auc", "roc_auc", "f1", "precision", "recall",
    "false_alarm_rate", "brier", "positive_rate", "n_positive", "n_samples",
]


def guard_metrics(names: Iterable[str]) -> None:
    """Refuse to report accuracy. Raises :class:`ValueError` on any banned name.

    This is not decoration. It is the single sentence that earns credibility
    with an examiner, and it has to be enforced rather than promised.
    """
    bad = FORBIDDEN & {str(n).lower() for n in names}
    if bad:
        raise ValueError(
            f"{sorted(bad)} is banned: with a ~11% positive rate, predicting 'no crash' "
            f"always scores ~89%. Use PR-AUC, F1 and lead time."
        )


def false_alarm_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of calm days on which the system cried wolf (FP / (FP + TN))."""
    negatives = y_true == 0
    total = int(negatives.sum())
    if total == 0:
        return float("nan")
    return float((y_pred[negatives] == 1).sum() / total)


def compute_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Full rare-event metric suite at one operating point.

    Deliberately does not compute accuracy; :func:`guard_metrics` is called on
    the produced keys so the ban cannot be bypassed by adding a key later.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    single_class = len(np.unique(y_true)) < 2
    if single_class:
        log.warning("metrics: the evaluation set contains one class only; "
                    "ranking metrics are undefined and reported as NaN")

    metrics = {
        "pr_auc": float("nan") if single_class else float(average_precision_score(y_true, y_score)),
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "false_alarm_rate": false_alarm_rate(y_true, y_pred),
        "brier": float(brier_score_loss(y_true, y_score)),
        "positive_rate": float(y_true.mean()),
        "n_positive": int(y_true.sum()),
        "n_samples": int(len(y_true)),
    }
    guard_metrics(metrics.keys())
    return metrics


def baseline_pr_auc(y_true: np.ndarray) -> float:
    """PR-AUC of a random classifier: the positive rate.

    Every reported PR-AUC must be read against this. A PR-AUC of 0.15 on a 11%
    base rate is a real but modest improvement; the same number on a 20% base
    rate is worse than guessing.
    """
    return float(np.asarray(y_true).mean())


def find_best_threshold(
    y_true: np.ndarray, y_score: np.ndarray, metric: str = "f1"
) -> tuple[float, float]:
    """Operating point maximising ``metric`` on the given data.

    Must be called on TRAINING-fold scores only. Tuning the threshold on test
    scores is leakage of exactly the kind Section 4 exists to prevent.
    """
    guard_metrics([metric])
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5, float("nan")

    candidates = np.unique(np.quantile(y_score, np.linspace(0.50, 0.999, 60)))
    best_threshold, best_value = 0.5, -1.0
    for threshold in candidates:
        prediction = (y_score >= threshold).astype(int)
        value = f1_score(y_true, prediction, zero_division=0)
        if value > best_value:
            best_threshold, best_value = float(threshold), float(value)
    return best_threshold, best_value


def lead_time_days(
    y_score: pd.Series,
    threshold: float,
    episodes: Sequence,
    sustained_days: int = 2,
) -> pd.DataFrame:
    """Days of warning before each crisis episode began.

    For each episode, finds the FIRST sustained threshold crossing (at least
    ``sustained_days`` consecutive days above threshold, so a single noisy day
    does not count as a warning) and reports how many days before the episode
    start it occurred.

    Negative means the warning came too late. One row per episode: failures are
    reported individually and never averaged away.

    Args:
        y_score: predicted probabilities indexed by date.
        threshold: the operating point.
        episodes: configured crisis episodes.
        sustained_days: consecutive days required to count as a warning.
    """
    scores = y_score.sort_index()
    dates = pd.DatetimeIndex(scores.index)
    above = (scores >= threshold).to_numpy()

    # Mark the first day of every run of `sustained_days` consecutive crossings.
    sustained = np.zeros(len(above), dtype=bool)
    if len(above) >= sustained_days:
        rolling = np.convolve(above.astype(int), np.ones(sustained_days, dtype=int), mode="valid")
        sustained[: len(rolling)] = rolling == sustained_days

    rows = []
    for episode in episodes:
        start = pd.Timestamp(episode.start, tz="UTC")
        # Look back at most 90 days: a "warning" six months early is not a warning.
        window = (dates >= start - pd.Timedelta(90, "D")) & (dates < start)
        candidates = np.flatnonzero(window & sustained)
        if candidates.size:
            first = dates[candidates[0]]
            lead = int((start - first).days)
            fired = True
        else:
            # No pre-episode warning. Did it fire late, during the episode?
            end = pd.Timestamp(episode.end, tz="UTC")
            during = np.flatnonzero((dates >= start) & (dates <= end) & sustained)
            if during.size:
                lead = -int((dates[during[0]] - start).days)
                fired = True
            else:
                lead, fired = np.nan, False
        rows.append({
            "episode": episode.name,
            "start": start.date(),
            "lead_time_days": lead,
            "warned": fired,
        })
    return pd.DataFrame(rows)


def aggregate_folds(per_fold: Mapping[str, Mapping[str, float]]) -> pd.Series:
    """Mean and standard deviation of each metric across folds."""
    frame = pd.DataFrame(per_fold).T
    guard_metrics(frame.columns)
    out: dict[str, float] = {}
    for column in frame.columns:
        if column in {"n_positive", "n_samples"}:
            out[column] = float(frame[column].sum())
            continue
        out[f"{column}_mean"] = float(frame[column].mean())
        out[f"{column}_std"] = float(frame[column].std(ddof=0))
    return pd.Series(out)
