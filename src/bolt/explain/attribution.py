"""SHAP and gradient-times-input attribution (BOLT_SPEC.md Section 8).

- ``TreeExplainer`` for XGBoost and Random Forest.
- ``LinearExplainer`` for Logistic Regression.
- Gradient x input for the NumPy LSTM and GRU, using the exact input gradients
  the from-scratch implementation exposes (and which
  ``tests/test_lstm_gradcheck.py`` verifies against numerical differentiation).

Attributions arrive per (sample, timestep, feature) and are aggregated to
per-feature by summing over the window. BOTH views are recorded: the per-feature
view answers "which driver", the per-timestep view answers "how far in advance".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bolt.logging_setup import get_logger
from bolt.windows import flatten_windows

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Attribution:
    """Attributions for a set of samples.

    Attributes:
        per_feature: ``(N, F)`` contribution summed over the window.
        per_timestep: ``(N, T, F)`` raw contributions.
        feature_names: length-F names matching the F axis.
        method: which explainer produced this.
    """

    per_feature: np.ndarray
    per_timestep: np.ndarray
    feature_names: list[str]
    method: str

    def ranked(self, index: int | None = None) -> pd.DataFrame:
        """Ranked (feature, contribution, share_of_total) for one sample or the mean."""
        values = (
            self.per_feature[index] if index is not None
            else self.per_feature.mean(axis=0)
        )
        magnitude = np.abs(values)
        total = magnitude.sum()
        frame = pd.DataFrame({
            "feature": self.feature_names,
            "contribution": values,
            "abs_contribution": magnitude,
            "share_of_total": magnitude / total if total > 0 else np.zeros_like(magnitude),
        })
        return frame.sort_values("abs_contribution", ascending=False).reset_index(drop=True)

    def mean_absolute(self) -> pd.Series:
        """Mean |attribution| per feature - the input to the G4 consistency analysis."""
        return pd.Series(
            np.abs(self.per_feature).mean(axis=0), index=self.feature_names
        ).sort_values(ascending=False)


def _unflatten(values: np.ndarray, lookback: int, n_features: int) -> np.ndarray:
    """Reshape flattened ``(N, T*F)`` attributions back to ``(N, T, F)``."""
    return values.reshape(values.shape[0], lookback, n_features)


def attribute_tree(model, X: np.ndarray, feature_names: list[str],
                   max_samples: int = 500) -> Attribution:
    """SHAP TreeExplainer for XGBoost and Random Forest."""
    import shap

    inner = getattr(model, "model", model)
    sample = X[:max_samples]
    flat = flatten_windows(sample)

    explainer = shap.TreeExplainer(inner)
    values = explainer.shap_values(flat, check_additivity=False)
    if isinstance(values, list):          # older API returns one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:                  # (N, features, classes)
        values = values[:, :, -1]

    per_timestep = _unflatten(values, sample.shape[1], sample.shape[2])
    return Attribution(per_timestep.sum(axis=1), per_timestep, feature_names, "shap_tree")


def attribute_linear(model, X: np.ndarray, feature_names: list[str],
                     background: np.ndarray | None = None,
                     max_samples: int = 500) -> Attribution:
    """SHAP LinearExplainer for Logistic Regression."""
    import shap

    inner = getattr(model, "model", model)
    sample = X[:max_samples]
    flat = flatten_windows(sample)
    reference = flatten_windows(background if background is not None else X[:max_samples])

    explainer = shap.LinearExplainer(inner, reference)
    values = np.asarray(explainer.shap_values(flat))
    if values.ndim == 3:
        values = values[:, :, -1]

    per_timestep = _unflatten(values, sample.shape[1], sample.shape[2])
    return Attribution(per_timestep.sum(axis=1), per_timestep, feature_names, "shap_linear")


def attribute_gradient(model, X: np.ndarray, feature_names: list[str],
                       max_samples: int = 500) -> Attribution:
    """Gradient x input for the from-scratch sequence models.

    ``d(logit)/d(x) * x`` is a first-order attribution: it answers "how much did
    this input value move the logit, given where the model currently is". It is
    the natural counterpart to SHAP for a model we differentiate ourselves, and
    it is only trustworthy because the input gradients are gradient-checked.
    """
    if not hasattr(model, "input_gradients"):
        raise TypeError(
            f"{type(model).__name__} exposes no input_gradients(); gradient attribution "
            f"is only available for the from-scratch NumPy models."
        )
    sample = X[:max_samples]
    gradients = model.input_gradients(sample)
    per_timestep = gradients * sample
    return Attribution(per_timestep.sum(axis=1), per_timestep, feature_names, "grad_x_input")


def attribute(model, X: np.ndarray, feature_names: list[str],
              background: np.ndarray | None = None, max_samples: int = 500) -> Attribution:
    """Dispatch to the right explainer for the model type."""
    name = getattr(model, "name", type(model).__name__)
    if name in {"lstm", "gru"}:
        return attribute_gradient(model, X, feature_names, max_samples)
    if name in {"xgb", "rf"}:
        return attribute_tree(model, X, feature_names, max_samples)
    if name == "logreg":
        return attribute_linear(model, X, feature_names, background, max_samples)
    raise TypeError(
        f"no attribution method for model {name!r}. The volatility rule and MLP are "
        f"deliberately unexplained: the rule has one input by construction, and the "
        f"MLP has no gradient-checked input-gradient path in this build."
    )


def timestep_profile(attribution: Attribution) -> pd.Series:
    """Mean |attribution| by position in the window, earliest day first.

    Answers how far ahead of the decision date the model's evidence sits. Mass
    concentrated at the final timestep means the model is reacting, not warning.
    """
    magnitude = np.abs(attribution.per_timestep).mean(axis=(0, 2))
    lookback = len(magnitude)
    index = [f"t-{lookback - 1 - i}" for i in range(lookback)]
    return pd.Series(magnitude, index=index)
