"""Risk Orchestrator Agent (ChainGuard architecture).

The central coordinator. Receives reports from the market, on-chain, news and
quantitative agents and determines whether the evidence is CONSISTENT, not
merely whether it is alarming.

Intelligence type: **weighted evidence aggregation**, deterministic.

The design choice that matters: three independent sources agreeing at 70 is a
stronger signal than one source at 95 with the others silent. Agreement across
methodologically independent evidence is the thing a single model cannot give
you, and it is what this layer exists to measure. So the aggregate is weighted
by each agent's confidence AND adjusted by how much the agents concur.
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

#: Relative weight of each agent in the aggregate. The quantitative agent
#: carries the most because it is the only component trained against outcomes;
#: the others carry real weight because they are independent of it.
DEFAULT_WEIGHTS = {
    "Quantitative Agent": 0.40,
    "Market Intelligence Agent": 0.25,
    "On-Chain Intelligence Agent": 0.25,
    "News & Sentiment Agent": 0.10,
}

MIN_AGENTS_FOR_DECISION = 2


class RiskOrchestratorAgent:
    """Aggregates agent reports into a preliminary risk score."""

    name = "Risk Orchestrator"

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or dict(DEFAULT_WEIGHTS)

    def analyse(self, context: AgentContext, reports: list[AgentReport]) -> AgentReport:
        usable = [r for r in reports if r.usable and np.isfinite(r.score)]
        if len(usable) < MIN_AGENTS_FOR_DECISION:
            return AgentReport(
                agent=self.name, risk=RiskLevel.INSUFFICIENT_EVIDENCE, score=float("nan"),
                confidence=0.0, status=AgentStatus.UNAVAILABLE,
                notes=(
                    f"only {len(usable)} agent(s) produced a usable assessment; at least "
                    f"{MIN_AGENTS_FOR_DECISION} independent sources are required before "
                    f"this system will state a risk level",
                ),
            )

        scores = np.array([r.score for r in usable])
        weights = np.array([
            self.weights.get(r.agent, 0.1) * max(r.confidence, 0.05) for r in usable
        ])
        weights = weights / weights.sum()
        weighted = float(np.sum(scores * weights))

        # Consensus, measured as CONFIDENCE-WEIGHTED dispersion around the
        # weighted mean rather than as the raw min-max range.
        #
        # The principle, stated before looking at any result: a dissent should
        # count against the conclusion in proportion to how confident the
        # dissenter is. A tentative disagreement from an agent operating at 43%
        # confidence is weaker evidence against than a firm one at 96%. Raw
        # min-max treats them identically, so a single hedged outlier could
        # collapse the consensus of everyone else.
        spread = float(scores.max() - scores.min())
        deviation = float(np.sqrt(np.sum(weights * (scores - weighted) ** 2)))
        # Map dispersion onto [0.5, 1.0]: total disagreement HALVES confidence
        # rather than destroying it, because the weighted score still carries
        # information even when the agents differ. Multiplying a dispersion
        # factor by the mean confidence collapsed both together and made the
        # system abstain on essentially every prediction.
        consensus = float(np.clip(1.0 - deviation / 50.0, 0.0, 1.0))
        consensus_factor = 0.5 + 0.5 * consensus

        evidence = [
            Evidence(
                name=r.agent, value=r.score, direction=+1 if r.score >= 50 else -1,
                weight=float(w),
                detail=f"{r.agent}: {r.risk.value} at {r.score:.0f}/100 "
                       f"(confidence {r.confidence:.0%}, weight {w:.0%})",
            )
            for r, w in zip(usable, weights)
        ]

        elevated = [r for r in usable if r.score >= 65]
        notes = [self._summarise(usable, elevated, spread)]
        degraded = [r.agent for r in usable if r.status is AgentStatus.DEGRADED]
        if degraded:
            notes.append(f"operating on degraded input from: {', '.join(degraded)}")

        return AgentReport(
            agent=self.name,
            risk=RiskLevel.from_score(weighted, _bands(context)),
            score=weighted,
            # Confidence-weighted mean, scaled by how much the agents concur.
            confidence=float(
                consensus_factor * np.sum(weights * np.array([r.confidence for r in usable]))
            ),
            status=AgentStatus.OK if len(usable) >= 3 else AgentStatus.DEGRADED,
            evidence=tuple(evidence),
            notes=tuple(notes),
            extra={
                "consensus": round(consensus, 6),
                "weighted_dispersion": round(deviation, 6),
                "spread": round(spread, 6),
                "agents_used": len(usable),
                "agents_total": len(reports),
            },
        )

    @staticmethod
    def _summarise(usable: list[AgentReport], elevated: list[AgentReport], spread: float) -> str:
        """The orchestrator's plain-language reading of the evidence."""
        quant = next((r for r in usable if r.agent == "Quantitative Agent"), None)
        independent = [r for r in elevated if r.agent != "Quantitative Agent"]

        if not elevated:
            return (
                f"No source reports elevated risk; the {len(usable)} agents agree the "
                f"market is not in a stressed state (spread {spread:.0f} points)"
            )
        if quant is not None and quant.score >= 65 and independent:
            return (
                f"{len(independent)} independent evidence source(s) indicate elevated "
                f"downside risk, and the quantitative models support the warning at "
                f"{quant.score:.0f}/100"
            )
        if quant is not None and quant.score >= 65 and not independent:
            return (
                f"the quantitative models warn at {quant.score:.0f}/100 but NO independent "
                f"evidence source corroborates them; the models may be responding to a "
                f"pattern the observable drivers do not explain"
            )
        return (
            f"{len(elevated)} source(s) report elevated risk while the quantitative "
            f"models do not corroborate; evidence is mixed (spread {spread:.0f} points)"
        )


def _bands(context: AgentContext) -> dict[str, float]:
    if context.config is not None:
        try:
            return dict(context.config.section("agents")["risk_bands"])
        except Exception:
            pass
    return {"moderate": 40.0, "high": 65.0, "critical": 80.0}
