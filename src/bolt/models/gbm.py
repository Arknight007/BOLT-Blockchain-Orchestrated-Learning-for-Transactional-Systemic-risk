"""XGBoost baseline - G3 (BOLT_SPEC.md Section 6).

The base paper's comparison set covered Logistic Regression, SVM and Random
Forest. Gradient boosting - the strongest classical family for structured
financial data - was omitted. Including it makes our comparison harder to win
rather than easier, which is the point: a panel respects a team that competes
against the strongest available alternative.

XGBoost also pairs cleanly with SHAP's TreeExplainer, on which the G4
consistency analysis depends.
"""

from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

from bolt.models.base import class_weights
from bolt.windows import flatten_windows


class XGBModel:
    """XGBoost on the flattened window tensor."""

    name = "xgb"

    def __init__(self, n_estimators: int = 400, max_depth: int = 4,
                 learning_rate: float = 0.05, seed: int = 42) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.model: XGBClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        weights = class_weights(y)
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.seed,
            # Class imbalance handled by the positive-class weight ratio, which
            # is BOLT's primary strategy; SMOTE is compared as an ablation.
            scale_pos_weight=weights[1] / weights[0],
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=4,
            verbosity=0,
        )
        self.model.fit(flatten_windows(X), y, sample_weight=sample_weight)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("XGBModel.predict_proba called before fit()")
        return self.model.predict_proba(flatten_windows(X))[:, 1]
