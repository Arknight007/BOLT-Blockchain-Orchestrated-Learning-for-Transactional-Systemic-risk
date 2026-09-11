"""Quantitative / Prediction Agent (ChainGuard architecture).

This is where the neural network lives. The agent runs the trained models over
the current window and reconciles their outputs.

Intelligence type: **LSTM + XGBoost**, the two halves of the problem:

* The LSTM reads the SEQUENCE. A crash is not signalled by one variable crossing
  a threshold; it is signalled by a pattern developing across days, in which
  volatility rises while selling pressure accelerates and sentiment deteriorates.
  A recurrent model reads the window in order and carries earlier context.
* XGBoost reads the ENGINEERED TABULAR view. Gradient boosting is consistently
  strong on structured financial data, and the base paper omitted it.

When the models disagree materially the agent says so rather than averaging the
disagreement away. Two models 30 points apart is itself evidence - it is the
signal the skeptic agent is looking for.
"""

from __future__ import annotations

import numpy as np

from bolt.agents.base import (
    AgentContext,
    AgentReport,
    AgentStatus,
    Evidence,
    RiskLevel,
)
from bolt.logging_setup import get_logger

log = get_logger(__name__)

#: Probability-point gap above which the models are treated as disagreeing.
DISAGREEMENT_THRESHOLD = 0.20


class QuantitativeAgent:
    """Runs the trained models and reconciles their probabilities."""

    name = "Quantitative Agent"

    def __init__(self, models: dict[str, object] | None = None,
                 scaler: object | None = None) -> None:
        """
        Args:
            models: fitted models keyed by name, e.g. ``{"lstm": ..., "xgb": ...}``.
            scaler: the TRAIN-FITTED scaler. Guard 5 applies here too: scoring a
                live window with a scaler fitted on that window would leak.
        """
        self.models = models or {}
        self.scaler = scaler

    def analyse(self, context: AgentContext) -> AgentReport:
        if not self.models:
            return AgentReport(
                agent=self.name, risk=RiskLevel.INSUFFICIENT_EVIDENCE, score=float("nan"),
                confidence=0.0, status=AgentStatus.UNAVAILABLE,
                notes=("no trained models supplied; run `bolt train` first",),
            )
        if context.X_window is None:
            return AgentReport(
                agent=self.name, risk=RiskLevel.INSUFFICIENT_EVIDENCE, score=float("nan"),
                confidence=0.0, status=AgentStatus.UNAVAILABLE,
                notes=("no complete feature window at this date; the lookback is "
                       "incomplete or a feature is missing",),
            )

        X = np.asarray(context.X_window, dtype=np.float64)
        if self.scaler is not None:
            X = self.scaler.transform(X)

        probabilities: dict[str, float] = {}
        failures: list[str] = []
        for name, model in self.models.items():
            try:
                probabilities[name] = float(np.asarray(model.predict_proba(X)).ravel()[0])
            except Exception as exc:                       # noqa: BLE001 - reported, not hidden
                failures.append(f"{name}: {type(exc).__name__}")
                log.warning("model %s failed to score: %s", name, exc)

        if not probabilities:
            return AgentReport(
                agent=self.name, risk=RiskLevel.INSUFFICIENT_EVIDENCE, score=float("nan"),
                confidence=0.0, status=AgentStatus.UNAVAILABLE,
                notes=tuple(["every model failed to score"] + failures),
            )

        values = np.asarray(list(probabilities.values()))
        combined = float(np.mean(values))
        spread = float(values.max() - values.min()) if len(values) > 1 else 0.0
        disagreement = spread >= DISAGREEMENT_THRESHOLD

        evidence = [
            Evidence(
                name=f"model:{name}", value=probability, direction=+1,
                weight=probability,
                detail=f"{name.upper()} estimates {probability:.0%} crash probability",
            )
            for name, probability in sorted(probabilities.items())
        ]

        notes: list[str] = []
        if disagreement:
            low = min(probabilities, key=probabilities.get)
            high = max(probabilities, key=probabilities.get)
            notes.append(
                f"models DISAGREE by {spread:.0%} ({low}={probabilities[low]:.0%}, "
                f"{high}={probabilities[high]:.0%}); the combined estimate is less "
                f"reliable than its value suggests"
            )
        if failures:
            notes.append(f"failed to score: {', '.join(failures)}")

        return AgentReport(
            agent=self.name,
            risk=RiskLevel.from_score(100.0 * combined, _bands(context)),
            score=100.0 * combined,
            # Agreement between independent model families is the confidence.
            confidence=float(np.clip(1.0 - spread, 0.0, 1.0)),
            status=AgentStatus.DEGRADED if (failures or disagreement) else AgentStatus.OK,
            evidence=tuple(evidence),
            notes=tuple(notes),
            extra={
                "probabilities": {k: round(v, 6) for k, v in sorted(probabilities.items())},
                "spread": round(spread, 6),
                "models_scored": len(probabilities),
            },
        )

    def expected_severity(self, context: AgentContext, probability: float) -> dict:
        """Expected decline range, derived from realised history - never invented.

        The models predict a BINARY event (a drawdown beyond the threshold within
        the horizon), so they carry no magnitude estimate. Rather than fabricate
        one, this reports the empirical distribution of drawdowns that actually
        followed comparable conditions, and says plainly when there is no basis.
        """
        prices = context.series("close")
        if prices is None or len(prices) < 60:
            return {"available": False, "reason": "insufficient price history"}

        returns = prices.pct_change(context.config.horizon_days if context.config else 14)
        declines = -returns[returns < 0].dropna()
        if declines.empty:
            return {"available": False, "reason": "no historical declines to calibrate against"}

        return {
            "available": True,
            "expected_decline_low": float(declines.quantile(0.50)),
            "expected_decline_high": float(declines.quantile(0.90)),
            "horizon_days": int(context.config.horizon_days) if context.config else 14,
            "basis": (
                f"empirical distribution of {len(declines)} historical "
                f"{context.config.horizon_days if context.config else 14}-day declines "
                f"for {context.asset}"
            ),
        }


def _bands(context: AgentContext) -> dict[str, float]:
    if context.config is not None:
        try:
            return dict(context.config.section("agents")["risk_bands"])
        except Exception:
            pass
    return {"moderate": 40.0, "high": 65.0, "critical": 80.0}
