"""Agent / Model Health Agent (ChainGuard architecture).

Continuous monitoring: is the system still working, or has the world moved away
from what it was trained on?

Intelligence type: **ML monitoring**, deterministic statistics.

Three things it watches:

* **Feature drift.** Population Stability Index between the training distribution
  and recent data. A model scoring inputs unlike anything it was trained on is
  extrapolating, and its probabilities stop meaning what they meant.
* **Performance decay.** Per-fold metrics trending downward across time.
* **Agent reliability.** How often each agent produced a usable report at all.

This is the agent that eventually says "retraining required" - and says it from
measurements, not from a feeling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bolt.agents.base import AgentReport, AgentStatus, Evidence, RiskLevel
from bolt.logging_setup import get_logger

log = get_logger(__name__)

#: Conventional PSI cut-points, fixed in advance so a result cannot be reframed.
PSI_MINOR = 0.10
PSI_MAJOR = 0.25


@dataclass(frozen=True, slots=True)
class DriftReading:
    feature: str
    psi: float
    severity: str

    @property
    def drifted(self) -> bool:
        return self.psi >= PSI_MINOR


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """PSI between two samples of one feature.

    PSI = sum (current% - reference%) * ln(current% / reference%), over quantile
    bins of the reference distribution. Standard in credit risk monitoring, where
    the same question - has the population moved? - has been asked for decades.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < bins * 5 or len(current) < bins:
        return float("nan")

    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf

    reference_share = np.histogram(reference, bins=edges)[0] / len(reference)
    current_share = np.histogram(current, bins=edges)[0] / len(current)
    # Floor empty bins so the logarithm stays finite.
    floor = 1e-6
    reference_share = np.maximum(reference_share, floor)
    current_share = np.maximum(current_share, floor)
    return float(np.sum((current_share - reference_share) *
                        np.log(current_share / reference_share)))


class HealthAgent:
    """Monitors drift, decay and agent reliability."""

    name = "Health Agent"

    def feature_drift(
        self, reference: np.ndarray, current: np.ndarray, feature_names: list[str]
    ) -> list[DriftReading]:
        """PSI per feature between a training window and a recent window."""
        if reference.ndim == 3:
            reference = reference.reshape(-1, reference.shape[-1])
        if current.ndim == 3:
            current = current.reshape(-1, current.shape[-1])

        readings = []
        for index, name in enumerate(feature_names):
            psi = population_stability_index(reference[:, index], current[:, index])
            severity = (
                "unknown" if not np.isfinite(psi) else
                "major" if psi >= PSI_MAJOR else
                "minor" if psi >= PSI_MINOR else "stable"
            )
            readings.append(DriftReading(name, psi, severity))
        return sorted(readings, key=lambda r: (-r.psi if np.isfinite(r.psi) else 0))

    def performance_trend(self, per_fold: pd.DataFrame, metric: str = "pr_auc") -> dict:
        """Is the primary metric declining across chronologically ordered folds?"""
        if per_fold.empty or metric not in per_fold.columns:
            return {"available": False, "reason": f"no {metric} column"}

        trends = {}
        for model, block in per_fold.groupby("model"):
            ordered = block.sort_values("fold")
            values = ordered[metric].to_numpy(dtype=float)
            if len(values) < 3 or not np.isfinite(values).all():
                continue
            slope = float(np.polyfit(np.arange(len(values)), values, 1)[0])
            trends[model] = {
                "slope_per_fold": round(slope, 6),
                "first": round(float(values[0]), 6),
                "last": round(float(values[-1]), 6),
                "declining": slope < -0.01,
            }
        return {"available": bool(trends), "metric": metric, "trends": trends}

    def analyse(
        self,
        drift: list[DriftReading] | None = None,
        performance: dict | None = None,
        agent_history: list[list[AgentReport]] | None = None,
    ) -> AgentReport:
        evidence: list[Evidence] = []
        notes: list[str] = []
        concerns = 0

        if drift:
            major = [d for d in drift if d.severity == "major"]
            minor = [d for d in drift if d.severity == "minor"]
            concerns += 2 * len(major) + len(minor)
            for reading in (major + minor)[:5]:
                evidence.append(Evidence(
                    name=reading.feature, value=reading.psi, direction=+1,
                    weight=min(reading.psi, 1.0),
                    detail=f"{reading.feature}: PSI {reading.psi:.3f} ({reading.severity} drift)",
                ))
            notes.append(
                f"feature drift: {len(major)} major, {len(minor)} minor of {len(drift)} features"
                if (major or minor) else
                f"no material feature drift across {len(drift)} features"
            )

        if performance and performance.get("available"):
            declining = [m for m, t in performance["trends"].items() if t["declining"]]
            concerns += len(declining)
            if declining:
                notes.append(
                    f"{performance['metric']} is declining across folds for: "
                    f"{', '.join(declining)}"
                )
                for model in declining[:3]:
                    trend = performance["trends"][model]
                    evidence.append(Evidence(
                        name=f"decay:{model}", value=trend["slope_per_fold"], direction=+1,
                        weight=min(abs(trend["slope_per_fold"]) * 10, 1.0),
                        detail=(
                            f"{model} {performance['metric']} fell from {trend['first']:.3f} "
                            f"to {trend['last']:.3f} across folds"
                        ),
                    ))
            else:
                notes.append(f"no systematic {performance['metric']} decay across folds")

        if agent_history:
            reliability = self._agent_reliability(agent_history)
            for agent, rate in sorted(reliability.items(), key=lambda kv: kv[1]):
                if rate < 0.80:
                    concerns += 1
                    evidence.append(Evidence(
                        name=f"reliability:{agent}", value=rate, direction=+1,
                        weight=1.0 - rate,
                        detail=f"{agent} produced a usable report {rate:.0%} of the time",
                    ))
            notes.append(f"agent reliability measured over {len(agent_history)} run(s)")

        if not evidence and not notes:
            return AgentReport(
                agent=self.name, risk=RiskLevel.INSUFFICIENT_EVIDENCE, score=float("nan"),
                confidence=0.0, status=AgentStatus.UNAVAILABLE,
                notes=("no monitoring inputs supplied",),
            )

        recommendation = (
            "RETRAINING RECOMMENDED: multiple health indicators have degraded."
            if concerns >= 4 else
            "Monitor: some indicators have moved but none is individually decisive."
            if concerns >= 2 else
            "Healthy: no action required."
        )
        notes.append(recommendation)

        return AgentReport(
            agent=self.name,
            risk=RiskLevel.MODERATE if concerns >= 4 else RiskLevel.LOW,
            score=float(min(100.0, 12.5 * concerns)), confidence=1.0, status=AgentStatus.OK,
            evidence=tuple(evidence), notes=tuple(notes),
            extra={"concerns": concerns, "recommendation": recommendation},
        )

    @staticmethod
    def _agent_reliability(history: list[list[AgentReport]]) -> dict[str, float]:
        totals: dict[str, int] = {}
        usable: dict[str, int] = {}
        for run in history:
            for report in run:
                totals[report.agent] = totals.get(report.agent, 0) + 1
                usable[report.agent] = usable.get(report.agent, 0) + int(report.usable)
        return {agent: usable[agent] / totals[agent] for agent in totals}
