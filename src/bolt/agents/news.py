"""News & Sentiment Agent (ChainGuard architecture).

Status: **PARTIAL**. Deliberately so, and the limitation is stated in every
report this agent emits.

What works now: the deterministic sentiment path. The agent reads the
point-in-time sentiment features and the Fear and Greed index and reports
whether market sentiment is stressed. This is real, reproducible, and enters the
hashed on-chain payload.

What does NOT work yet: **event identification**. The architecture calls for an
agent that identifies the EVENT explaining a sentiment move ("a regulatory
announcement caused a sharp increase in negative sentiment") rather than
reducing news to a score. That needs headline text, which needs a CryptoPanic
API key that is not configured in this build, plus an NLP layer over it.

The honest position is that this agent currently reports sentiment LEVEL, not
sentiment CAUSE, and says so. Reporting a fabricated causal event would be worse
than reporting no event at all.
"""

from __future__ import annotations

import numpy as np

from bolt.agents.base import (
    AgentContext,
    AgentReport,
    AgentStatus,
    Evidence,
    RiskLevel,
    percentile_of,
)
from bolt.logging_setup import get_logger

log = get_logger(__name__)

NOT_IMPLEMENTED_NOTE = (
    "EVENT IDENTIFICATION IS NOT IMPLEMENTED. This agent reports sentiment LEVEL only. "
    "Naming the event that caused a sentiment move requires headline text (a CryptoPanic "
    "API key, not configured) and an NLP layer over it - scheduled for Phase 7. No causal "
    "claim is made here, and none should be read into the score."
)


class NewsEventAgent:
    """Reports sentiment level. Event identification is not yet built."""

    name = "News & Sentiment Agent"

    SIGNALS = (
        ("headline_polarity_1d", -1, "same-day sentiment polarity"),
        ("headline_polarity_7d", -1, "7-day sentiment polarity"),
        ("negative_ratio_7d", +1, "share of negative days over 7 days"),
    )

    def analyse(self, context: AgentContext) -> AgentReport:
        contributions: list[float] = []
        evidence: list[Evidence] = []
        missing: list[str] = []

        for feature, direction, label in self.SIGNALS:
            value = context.value(feature)
            if not np.isfinite(value):
                missing.append(feature)
                continue
            percentile = percentile_of(context.series(feature), value)
            if not np.isfinite(percentile):
                missing.append(feature)
                continue

            risk_percentile = percentile if direction > 0 else 1.0 - percentile
            contributions.append(risk_percentile)

            if risk_percentile >= 0.80:
                evidence.append(Evidence(
                    name=feature, value=value, direction=+1, weight=risk_percentile,
                    detail=(
                        f"{label} is stressed: {value:.3f}, "
                        f"{risk_percentile:.0%} of its trailing history"
                    ),
                ))
            elif risk_percentile <= 0.20:
                evidence.append(Evidence(
                    name=feature, value=value, direction=-1, weight=1.0 - risk_percentile,
                    detail=f"{label} is constructive: {value:.3f}",
                ))

        if not contributions:
            return AgentReport(
                agent=self.name, risk=RiskLevel.INSUFFICIENT_EVIDENCE, score=float("nan"),
                confidence=0.0, status=AgentStatus.UNAVAILABLE,
                notes=(NOT_IMPLEMENTED_NOTE, f"no sentiment features available: {missing}"),
            )

        score = 100.0 * float(np.mean(contributions))
        return AgentReport(
            agent=self.name,
            risk=RiskLevel.from_score(score, _bands(context)),
            score=score,
            # Capped: a sentiment read with no event attribution and no headline
            # text behind it should never be the most confident voice in the room.
            confidence=float(min(0.55, len(contributions) / len(self.SIGNALS))),
            status=AgentStatus.DEGRADED,
            evidence=tuple(evidence),
            notes=(
                NOT_IMPLEMENTED_NOTE,
                "sentiment is derived from the alternative.me Fear and Greed index, "
                "a real published daily series, rather than from headline polarity",
            ),
            extra={
                "event_identification": "NOT_IMPLEMENTED",
                "major_event": None,
                "signals_used": len(contributions),
            },
        )


def _bands(context: AgentContext) -> dict[str, float]:
    if context.config is not None:
        try:
            return dict(context.config.section("agents")["risk_bands"])
        except Exception:
            pass
    return {"moderate": 40.0, "high": 65.0, "critical": 80.0}
