"""Common model interface (BOLT_SPEC.md Section 6).

Every model sees the SAME inputs. Sequence models take ``X`` of shape
``(N, T, F)``; tabular models take the same tensor and flatten it internally via
:func:`bolt.windows.flatten_windows`. If the tabular models received different
features the comparison would not be fair, and a panel would be right to say so.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BoltModel(Protocol):
    """The interface every benchmarked model implements."""

    name: str

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(crash) with shape ``(N,)``."""
        ...


def class_weights(y: np.ndarray) -> dict[int, float]:
    """Balanced class weights: ``n / (2 * n_class)``.

    This is BOLT's primary imbalance strategy. SMOTE is compared against it as an
    ablation, because synthetic oversampling of overlapping time-series windows
    creates near-duplicate samples that leak across folds.
    """
    y = np.asarray(y).astype(int)
    total = len(y)
    positives = int(y.sum())
    negatives = total - positives
    if positives == 0 or negatives == 0:
        return {0: 1.0, 1: 1.0}
    return {0: total / (2.0 * negatives), 1: total / (2.0 * positives)}


def sample_weights_from_classes(y: np.ndarray) -> np.ndarray:
    """Per-sample weights derived from :func:`class_weights`."""
    weights = class_weights(y)
    y = np.asarray(y).astype(int)
    return np.where(y == 1, weights[1], weights[0]).astype(np.float64)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def weighted_bce(
    y_true: np.ndarray, y_prob: np.ndarray, weights: np.ndarray | None = None, eps: float = 1e-12
) -> float:
    """Class-weighted binary cross-entropy."""
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    losses = -(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    if weights is None:
        return float(np.mean(losses))
    return float(np.sum(losses * weights) / np.sum(weights))
