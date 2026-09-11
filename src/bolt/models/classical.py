"""Logistic Regression, Random Forest and MLP baselines (BOLT_SPEC.md Section 6).

Logistic Regression is retained specifically because the base paper used it:
keeping it preserves comparability with the published work and establishes a
performance floor. If a linear model on identical features performs close to the
deep models, we are obliged to report that (Rule 12.7).

All three take the same window tensor and flatten it internally, so every model
in the bench sees identical inputs.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from bolt.windows import flatten_windows


class LogRegModel:
    """L2-regularised logistic regression with balanced class weights."""

    name = "logreg"

    def __init__(self, C: float = 1.0, max_iter: int = 2000, seed: int = 42) -> None:
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.model: LogisticRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        self.model = LogisticRegression(
            C=self.C, max_iter=self.max_iter, random_state=self.seed,
            class_weight="balanced", solver="lbfgs",
        )
        self.model.fit(flatten_windows(X), y, sample_weight=sample_weight)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LogRegModel.predict_proba called before fit()")
        return self.model.predict_proba(flatten_windows(X))[:, 1]


class RandomForestModel:
    """Random Forest, the strongest model in the base paper's comparison set."""

    name = "rf"

    def __init__(self, n_estimators: int = 400, max_depth: int = 8, seed: int = 42) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.seed = int(seed)
        self.model: RandomForestClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            random_state=self.seed, class_weight="balanced", n_jobs=4,
        )
        self.model.fit(flatten_windows(X), y, sample_weight=sample_weight)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("RandomForestModel.predict_proba called before fit()")
        return self.model.predict_proba(flatten_windows(X))[:, 1]


class MLPModel:
    """Feed-forward network on the flattened window.

    Included as the non-recurrent neural baseline: it has access to the same
    30-day window as the LSTM but no architectural notion of sequence order, so
    the gap between them isolates what recurrence actually buys.
    """

    name = "mlp"

    def __init__(self, hidden_layer_sizes=(64, 32), max_iter: int = 500, seed: int = 42) -> None:
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.model: MLPClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        # MLPClassifier has no sample_weight support, so the imbalance is handled
        # by resampling the minority class up to the configured ratio instead.
        flat = flatten_windows(X)
        if sample_weight is not None:
            flat, y = _weighted_resample(flat, y, sample_weight, self.seed)
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter,
            random_state=self.seed, early_stopping=True, n_iter_no_change=15,
        )
        self.model.fit(flat, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MLPModel.predict_proba called before fit()")
        return self.model.predict_proba(flatten_windows(X))[:, 1]


def _weighted_resample(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample rows with replacement in proportion to weights.

    This is resampling of REAL rows, not synthesis. It duplicates existing
    windows rather than interpolating between them, so it cannot create a sample
    that never occurred -- which is the specific risk SMOTE carries on
    overlapping time-series windows.
    """
    rng = np.random.default_rng(seed)
    probabilities = weights / weights.sum()
    idx = rng.choice(len(X), size=len(X), replace=True, p=probabilities)
    return X[idx], y[idx]
