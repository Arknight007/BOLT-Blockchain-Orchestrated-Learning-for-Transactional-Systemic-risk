"""Skeptic / Red-Team Agent (ChainGuard architecture).

The most important differentiator in the architecture. Its job is to CHALLENGE
the prediction: "what evidence suggests this might be wrong?"

Intelligence type: **rules**, deterministic. Every challenge below is a
falsifiable check against observable state, not a language model's opinion about
plausibility. That matters because the skeptic's output moves the final decision,
and a component that can move a decision must be reproducible by a verifier.

The system is therefore not designed to find reasons to predict a crash. It
actively tries to disprove its own warning, and the counter-evidence is recorded
in the on-chain payload alongside the prediction - so the committed record shows
what the system knew might be wrong at the time it committed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bolt.agents.base import (
    AgentContext,
    AgentReport,
    AgentStatus,
    Evidence,
    RiskLevel,
)
from bolt.logging_setup import get_logger

log = get_logger(__name__)

#: Maximum total confidence reduction the skeptic may apply, as a fraction.
MAX_PENALTY = 0.60


class SkepticAgent:
    """Red-teams the preliminary assessment and recommends a confidence adjustment."""

    name = "Skeptic Agent"

    def analyse(
        self,
        context: AgentContext,
        preliminary: AgentReport,
        reports: list[AgentReport],
        analogues: list[dict] | None = None,
    ) -> AgentReport:
        challenges: list[Evidence] = []

        for check in (
            self._check_long_term_trend,
            self._check_conflicting_flow,
            self._check_model_disagreement,
            self._check_data_quality,
            self._check_agent_divergence,
            self._check_regime_novelty,
        ):
            found = check(context, preliminary, reports)
            if found is not None:
                challenges.append(found)

        analogue_challenge = self._check_historical_similarity(analogues)
        if analogue_challenge is not None:
            challenges.append(analogue_challenge)

        penalty = float(min(sum(c.weight for c in challenges), MAX_PENALTY))
        recommendation = self._recommendation(penalty, challenges)

        return AgentReport(
            agent=self.name,
            # The skeptic's "score" is how much doubt it found, not crash risk.
            risk=RiskLevel.LOW if penalty < 0.2 else RiskLevel.MODERATE,
            score=100.0 * penalty,
            confidence=1.0,
            status=AgentStatus.OK,
            evidence=tuple(challenges),
            notes=(recommendation,),
            extra={"penalty": round(penalty, 6), "challenges": len(challenges)},
        )

    # -- individual challenges --------------------------------------------
    def _check_long_term_trend(self, context, preliminary, reports) -> Evidence | None:
        """A crash warning inside a strong uptrend deserves more scepticism."""
        prices = context.series("close")
        if prices is None or len(prices) < 120:
            return None
        recent = float(prices.iloc[-1])
        long_mean = float(prices.iloc[-120:].mean())
        if recent > long_mean * 1.10 and preliminary.score >= 60:
            return Evidence(
                name="long_term_trend", value=recent / long_mean - 1.0, direction=-1,
                weight=0.15,
                detail=(
                    f"price is {recent / long_mean - 1.0:.1%} above its 120-day mean; "
                    f"the long-term trend remains constructive and contradicts the warning"
                ),
            )
        return None

    def _check_conflicting_flow(self, context, preliminary, reports) -> Evidence | None:
        """Selling pressure and stablecoin supply telling opposite stories."""
        netflow = context.value("exchange_netflow_7")
        stable = context.value("stablecoin_netflow")
        if not (np.isfinite(netflow) and np.isfinite(stable)):
            return None
        if netflow > 0 and stable > 0.01 and preliminary.score >= 60:
            return Evidence(
                name="conflicting_flow", value=stable, direction=-1, weight=0.12,
                detail=(
                    f"stablecoin supply is still EXPANDING ({stable:.2%} over 7 days) while "
                    f"the warning rests on selling pressure; capital is entering the system, "
                    f"not leaving it"
                ),
            )
        return None

    def _check_model_disagreement(self, context, preliminary, reports) -> Evidence | None:
        quant = next((r for r in reports if r.agent == "Quantitative Agent"), None)
        if quant is None:
            return None
        spread = float(quant.extra.get("spread", 0.0))
        if spread >= 0.20:
            return Evidence(
                name="model_disagreement", value=spread, direction=-1,
                weight=float(min(0.05 + spread * 0.5, 0.20)),
                detail=(
                    f"the sequence and tabular models disagree by {spread:.0%}; they are "
                    f"reading the same window differently, so neither should be trusted "
                    f"at face value"
                ),
            )
        return None

    def _check_data_quality(self, context, preliminary, reports) -> Evidence | None:
        """Data MISSING for this prediction weakens the conclusion built on it.

        Only ``UNAVAILABLE`` counts. ``DEGRADED`` is deliberately excluded, and
        the distinction matters:

        * STRUCTURAL degradation is a known limitation of this build, constant
          across every prediction - the On-Chain agent runs on documented
          proxies, the News agent has no event-identification layer. Those
          agents ALREADY discount their own confidence at source (News caps at
          0.55, On-Chain multiplies by 0.75), so penalising them again here
          double-counts the same weakness.
        * SITUATIONAL absence is specific to this prediction: an agent that
          could see nothing at all today.

        Double-counting was not theoretical. Measured on 2022-11-05, with the
        LSTM at 100% and XGBoost at 96% on the eve of the FTX collapse, the
        permanent-degradation penalty pushed confidence to 16% and the system
        abstained on a warning it had genuinely made. A skeptic that vetoes
        every prediction is not a skeptic, it is an off switch.
        """
        unavailable = [r.agent for r in reports if r.status is AgentStatus.UNAVAILABLE]
        if not unavailable:
            return None
        return Evidence(
            name="data_quality", value=float(len(unavailable)), direction=-1,
            weight=float(min(0.10 * len(unavailable), 0.25)),
            detail=(
                f"{len(unavailable)} agent(s) could see no data at all for this "
                f"prediction ({', '.join(unavailable)}); the evidence base is thinner "
                f"than the headline score implies"
            ),
        )

    def _check_agent_divergence(self, context, preliminary, reports) -> Evidence | None:
        spread = float(preliminary.extra.get("spread", 0.0))
        if spread >= 40.0:
            return Evidence(
                name="agent_divergence", value=spread, direction=-1,
                weight=float(min(spread / 400.0, 0.15)),
                detail=(
                    f"the agents span {spread:.0f} points of disagreement; this is not a "
                    f"consensus warning, it is one or two sources pulling the average"
                ),
            )
        return None

    def _check_regime_novelty(self, context, preliminary, reports) -> Evidence | None:
        """Is the current volatility regime one the model has actually seen?"""
        vol = context.value("realized_vol_30")
        history = context.series("realized_vol_30")
        if history is None or not np.isfinite(vol) or len(history.dropna()) < 100:
            return None
        clean = history.dropna()
        if vol > float(clean.quantile(0.99)):
            return Evidence(
                name="regime_novelty", value=vol, direction=-1, weight=0.12,
                detail=(
                    f"30-day volatility ({vol:.3f}) exceeds the 99th percentile of "
                    f"everything the model was trained on; this is an out-of-distribution "
                    f"regime and the prediction is an extrapolation"
                ),
            )
        return None

    @staticmethod
    def _check_historical_similarity(analogues: list[dict] | None) -> Evidence | None:
        if not analogues:
            return None
        best = max(float(a.get("similarity", 0.0)) for a in analogues)
        if best < 0.60:
            return Evidence(
                name="weak_analogue", value=best, direction=-1, weight=0.10,
                detail=(
                    f"the closest historical analogue has similarity {best:.2f}; the "
                    f"present configuration has no strong precedent, so precedent-based "
                    f"reasoning carries little weight here"
                ),
            )
        return None

    @staticmethod
    def _recommendation(penalty: float, challenges: list[Evidence]) -> str:
        if not challenges:
            return "No material counter-evidence found; the assessment stands as issued."
        if penalty >= 0.40:
            return (
                f"Reduce confidence substantially: {len(challenges)} independent challenges "
                f"totalling a {penalty:.0%} reduction."
            )
        if penalty >= 0.20:
            return f"Reduce confidence: {len(challenges)} challenge(s), {penalty:.0%} reduction."
        return f"Minor caveats only: {len(challenges)} challenge(s), {penalty:.0%} reduction."
