"""Volatility-threshold rule, the non-ML baseline (BOLT_SPEC.md Section 6).

Fires when trailing realised volatility exceeds its own rolling 90th percentile.
No training, no parameters learned from labels, no gradient.

This baseline exists to prove the ML models are earning their complexity. If it
wins, that is the result and it gets reported (Rule 12.7). A seven-model
comparison in which nobody checked whether a one-line rule does just as well is
not a comparison a panel should believe.
"""

from __future__ import annotations

import numpy as np

from bolt.logging_setup import get_logger

log = get_logger(__name__)


class VolatilityRule:
    """Non-learned baseline: high trailing volatility means elevated crash risk.

    ``predict_proba`` returns the empirical quantile of the window's final
    realised-volatility reading against the training distribution, so the output
    is a genuine score in [0, 1] that PR-AUC can rank rather than a bare 0/1.
    """

    name = "rule"

    def __init__(self, vol_quantile: float = 0.90, vol_window: int = 14,
                 quantile_window: int = 252, feature_index: int | None = None) -> None:
        self.vol_quantile = float(vol_quantile)
        self.vol_window = int(vol_window)
        self.quantile_window = int(quantile_window)
        self.feature_index = feature_index
        self.reference_: np.ndarray | None = None
        self.threshold_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        """Learn only the volatility DISTRIBUTION from training windows.

        The labels are deliberately unused. This is what makes it a rule rather
        than a model, and it is why its score is an honest floor.
        """
        signal = self._signal(X)
        self.reference_ = np.sort(signal)
        self.threshold_ = float(np.quantile(signal, self.vol_quantile))
        log.info("rule: volatility threshold at the %.0fth percentile = %.4f",
                 100 * self.vol_quantile, self.threshold_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.reference_ is None:
            raise RuntimeError("VolatilityRule.predict_proba called before fit()")
        signal = self._signal(X)
        # Empirical CDF position against the training distribution.
        positions = np.searchsorted(self.reference_, signal, side="right")
        return positions / max(len(self.reference_), 1)

    def _signal(self, X: np.ndarray) -> np.ndarray:
        """Final-timestep realised volatility for each window.

        Defaults to feature 0, which is ``realized_vol_7`` in the configured
        feature order. Passing ``feature_index`` makes the choice explicit.
        """
        index = 0 if self.feature_index is None else self.feature_index
        return np.asarray(X[:, -1, index], dtype=np.float64)
