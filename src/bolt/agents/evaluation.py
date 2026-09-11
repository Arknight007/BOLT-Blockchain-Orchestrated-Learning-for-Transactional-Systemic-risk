"""Evaluation Agent (ChainGuard architecture).

After a prediction's horizon closes, this agent scores it against what actually
happened and records the outcome.

Intelligence type: **analytics plus rules**, deterministic.

This is the agent that turns the on-chain commitments into an auditable TRACK
RECORD rather than a pile of hashes. A commitment proves a prediction existed at
a time; this agent proves whether it was right, and because the prediction was
committed before the outcome, the resulting record cannot be curated after the
fact.

Note what it does NOT compute: accuracy. ``guard_metrics`` raises on it, here as
everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from bolt.agents.base import AgentReport, AgentStatus, Evidence, RiskLevel
from bolt.evaluate.metrics import compute_metrics, guard_metrics
from bolt.logging_setup import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Outcome:
    """What actually happened after a prediction's as-of date."""

    prediction_id: str
    asset: str
    as_of_date: str
    horizon_days: int
    predicted_band: str
    predicted_probability: float
    actual_max_drawdown: float
    crash_occurred: bool
    classification: str          # TRUE_POSITIVE / FALSE_POSITIVE / TRUE_NEGATIVE / FALSE_NEGATIVE
    resolved: bool
    detail: str

    @classmethod
    def from_dict(cls, record: dict) -> "Outcome":
        """Rebuild from a ledger line, ignoring fields the store added."""
        fields = {
            "prediction_id", "asset", "as_of_date", "horizon_days", "predicted_band",
            "predicted_probability", "actual_max_drawdown", "crash_occurred",
            "classification", "resolved", "detail",
        }
        return cls(**{k: v for k, v in record.items() if k in fields})

    def to_dict(self) -> dict:
        return {
            "prediction_id": self.prediction_id, "asset": self.asset,
            "as_of_date": self.as_of_date, "horizon_days": self.horizon_days,
            "predicted_band": self.predicted_band,
            "predicted_probability": round(self.predicted_probability, 6),
            "actual_max_drawdown": round(self.actual_max_drawdown, 6),
            "crash_occurred": self.crash_occurred,
            "classification": self.classification, "resolved": self.resolved,
            "detail": self.detail,
        }


class EvaluationAgent:
    """Resolves committed predictions against realised outcomes."""

    name = "Evaluation Agent"

    def __init__(self, threshold: float, alert_bands=("HIGH", "CRITICAL")) -> None:
        self.threshold = float(threshold)
        self.alert_bands = set(alert_bands)

    def resolve(self, prediction: dict, prices: pd.Series) -> Outcome:
        """Score one prediction against the price series that followed it."""
        as_of = pd.Timestamp(prediction["as_of_date"], tz="UTC")
        horizon = int(prediction["horizon_days"])
        band = str(prediction["severity_band"])
        probability = float(prediction.get("probability", 0.0))

        series = prices.sort_index()
        window = series.loc[as_of: as_of + pd.Timedelta(horizon, "D")]

        if len(window) < 2:
            return Outcome(
                prediction_id=prediction["prediction_id"], asset=prediction["asset"],
                as_of_date=prediction["as_of_date"], horizon_days=horizon,
                predicted_band=band, predicted_probability=probability,
                actual_max_drawdown=float("nan"), crash_occurred=False,
                classification="UNRESOLVED", resolved=False,
                detail=(
                    f"the {horizon}-day horizon has not closed yet, or prices are "
                    f"unavailable. The prediction stands; it is not scored either way."
                ),
            )

        start = float(window.iloc[0])
        drawdown = (start - float(window.min())) / start
        crashed = bool(drawdown >= self.threshold)
        warned = band in self.alert_bands

        classification = (
            "TRUE_POSITIVE" if (warned and crashed) else
            "FALSE_POSITIVE" if (warned and not crashed) else
            "FALSE_NEGATIVE" if (not warned and crashed) else
            "TRUE_NEGATIVE"
        )
        verdict = {"TRUE_POSITIVE": "CORRECT", "TRUE_NEGATIVE": "CORRECT"}.get(
            classification, "INCORRECT"
        )
        return Outcome(
            prediction_id=prediction["prediction_id"], asset=prediction["asset"],
            as_of_date=prediction["as_of_date"], horizon_days=horizon,
            predicted_band=band, predicted_probability=probability,
            actual_max_drawdown=drawdown, crash_occurred=crashed,
            classification=classification, resolved=True,
            detail=(
                f"predicted {band} ({probability:.0%}); actual maximum drawdown over "
                f"{horizon} days was {drawdown:.1%} against a {self.threshold:.0%} "
                f"threshold. {verdict}."
            ),
        )

    def analyse(self, outcomes: list[Outcome]) -> AgentReport:
        """Aggregate resolved outcomes into a track-record report."""
        resolved = [o for o in outcomes if o.resolved]
        if not resolved:
            return AgentReport(
                agent=self.name, risk=RiskLevel.INSUFFICIENT_EVIDENCE, score=float("nan"),
                confidence=0.0, status=AgentStatus.UNAVAILABLE,
                notes=(
                    f"{len(outcomes)} prediction(s) recorded, none resolved yet. A track "
                    f"record needs closed horizons; nothing is claimed until then.",
                ),
            )

        y_true = np.array([o.crash_occurred for o in resolved], dtype=int)
        y_score = np.array([o.predicted_probability for o in resolved], dtype=float)
        metrics = (
            compute_metrics(y_true, y_score, 0.5) if len(np.unique(y_true)) > 1
            else {"note": "all resolved outcomes share one class; ranking metrics undefined"}
        )
        guard_metrics(metrics.keys())

        counts = pd.Series([o.classification for o in resolved]).value_counts().to_dict()
        correct = counts.get("TRUE_POSITIVE", 0) + counts.get("TRUE_NEGATIVE", 0)

        evidence = [
            Evidence(name=k, value=float(v), direction=0, weight=float(v) / len(resolved),
                     detail=f"{k}: {v} of {len(resolved)} resolved predictions")
            for k, v in sorted(counts.items())
        ]
        return AgentReport(
            agent=self.name, risk=RiskLevel.LOW,
            score=100.0 * correct / len(resolved), confidence=1.0, status=AgentStatus.OK,
            evidence=tuple(evidence),
            notes=(
                f"{len(resolved)} of {len(outcomes)} prediction(s) resolved; "
                f"{counts.get('TRUE_POSITIVE', 0)} true positive(s), "
                f"{counts.get('FALSE_POSITIVE', 0)} false alarm(s), "
                f"{counts.get('FALSE_NEGATIVE', 0)} missed crash(es)",
            ),
            extra={"counts": counts, "metrics": {
                k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()
            }},
        )
