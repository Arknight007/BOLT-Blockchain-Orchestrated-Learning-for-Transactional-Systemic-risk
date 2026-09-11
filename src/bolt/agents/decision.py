"""Decision Agent (ChainGuard architecture).

Makes the final call after weighing the preliminary assessment against the
skeptic's counter-evidence.

Intelligence type: **thresholds plus explicit reasoning rules**, deterministic.

The important design property is that ``INSUFFICIENT_EVIDENCE`` is a real
outcome. A warning system that is forced to produce a number for every day will
produce noise on the days it knows nothing, and those are exactly the days a
reader would most like to be told nothing is known. Refusing to answer is the
honest output when the evidence base has collapsed.
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

DEFAULT_BANDS = {"moderate": 40.0, "high": 65.0, "critical": 80.0}

#: Below this confidence the system declines to issue a level at all.
MIN_CONFIDENCE = 0.25


class DecisionAgent:
    """Produces the final risk band, or declines to."""

    name = "Decision Agent"

    def analyse(
        self,
        context: AgentContext,
        preliminary: AgentReport,
        skeptic: AgentReport,
        reports: list[AgentReport],
    ) -> AgentReport:
        bands = _bands(context)

        if preliminary.status is AgentStatus.UNAVAILABLE or not np.isfinite(preliminary.score):
            return AgentReport(
                agent=self.name, risk=RiskLevel.INSUFFICIENT_EVIDENCE, score=float("nan"),
                confidence=0.0, status=AgentStatus.UNAVAILABLE,
                notes=("the orchestrator could not form an assessment; no warning is "
                       "issued and none is implied",),
                extra={"decision": "ABSTAIN", "reason": "no usable agent evidence"},
            )

        penalty = float(skeptic.extra.get("penalty", 0.0))
        # The skeptic reduces CONFIDENCE, and only moves the score as a
        # consequence. Counter-evidence makes a warning less certain; it does not
        # make the underlying conditions milder.
        final_confidence = float(np.clip(preliminary.confidence * (1.0 - penalty), 0.0, 1.0))
        final_score = float(preliminary.score * (1.0 - 0.5 * penalty))

        if final_confidence < MIN_CONFIDENCE:
            return AgentReport(
                agent=self.name, risk=RiskLevel.INSUFFICIENT_EVIDENCE, score=final_score,
                confidence=final_confidence, status=AgentStatus.DEGRADED,
                evidence=tuple(skeptic.top_evidence(3)),
                notes=(
                    f"confidence fell to {final_confidence:.0%}, below the {MIN_CONFIDENCE:.0%} "
                    f"floor. The system declines to state a risk level rather than issue one "
                    f"it cannot support.",
                ),
                extra={"decision": "ABSTAIN", "reason": "confidence below floor",
                       "raw_score": round(preliminary.score, 6)},
            )

        level = RiskLevel.from_score(final_score, bands)
        action = {
            RiskLevel.LOW: "No warning.",
            RiskLevel.MODERATE: "Monitor.",
            RiskLevel.HIGH: "Warning issued.",
            RiskLevel.CRITICAL: "Severe crash warning issued.",
        }.get(level, "No warning.")

        evidence: list[Evidence] = []
        for report in reports:
            if report.usable and np.isfinite(report.score):
                evidence.extend(report.top_evidence(2))
        evidence.extend(skeptic.top_evidence(2))

        notes = [
            f"{action} Preliminary {preliminary.score:.0f}/100 adjusted to "
            f"{final_score:.0f}/100 after {skeptic.extra.get('challenges', 0)} challenge(s)."
        ]
        if penalty > 0:
            notes.append(
                f"confidence reduced from {preliminary.confidence:.0%} to "
                f"{final_confidence:.0%} by the skeptic"
            )

        return AgentReport(
            agent=self.name, risk=level, score=final_score, confidence=final_confidence,
            status=AgentStatus.OK, evidence=tuple(evidence), notes=tuple(notes),
            extra={
                "decision": "ALERT" if level.rank >= RiskLevel.HIGH.rank else "NO_ALERT",
                "action": action,
                "raw_score": round(preliminary.score, 6),
                "skeptic_penalty": round(penalty, 6),
                "bands": bands,
            },
        )


def _bands(context: AgentContext) -> dict[str, float]:
    if context.config is not None:
        try:
            return dict(context.config.section("agents")["risk_bands"])
        except Exception:
            pass
    return dict(DEFAULT_BANDS)
