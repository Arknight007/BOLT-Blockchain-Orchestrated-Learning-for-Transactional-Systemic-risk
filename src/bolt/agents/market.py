"""Market Intelligence Agent (ChainGuard architecture).

Analyses the financial behaviour of the asset: volatility, momentum, volume
anomaly, drawdown state and where each sits in its own trailing distribution.

Intelligence type: **rules plus analytics**, fully deterministic. No LLM. The
agent's job is to detect abnormal market behaviour and state what it saw, and
percentile thresholds do that reproducibly. Its output feeds the hashed on-chain
payload, which rules out anything non-reproducible.

This agent does NOT make the final crash decision. It reports evidence.
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

# Percentile above which a reading counts as elevated / extreme.
ELEVATED = 0.80
EXTREME = 0.95


class MarketIntelligenceAgent:
    """Detects abnormal market behaviour from price and volume features."""

    name = "Market Intelligence Agent"

    #: (feature, direction, label). direction +1 means a HIGH reading raises risk.
    SIGNALS = (
        ("realized_vol_7", +1, "short-horizon volatility"),
        ("realized_vol_30", +1, "30-day volatility"),
        ("volume_z_20", +1, "volume anomaly"),
        ("atr_14", +1, "average true range"),
        ("drawdown_depth", +1, "drawdown from the running peak"),
        ("momentum_10", -1, "10-day momentum"),
        ("rsi_14", -1, "relative strength"),
    )

    def analyse(self, context: AgentContext) -> AgentReport:
        evidence: list[Evidence] = []
        contributions: list[float] = []
        missing: list[str] = []

        for feature, direction, label in self.SIGNALS:
            value = context.value(feature)
            if not np.isfinite(value):
                missing.append(feature)
                continue

            history = context.series(feature)
            percentile = percentile_of(history, value)
            if not np.isfinite(percentile):
                missing.append(feature)
                continue

            # A -1 direction means a LOW reading is the risky one, so flip the
            # percentile before scoring: weak momentum is high risk.
            risk_percentile = percentile if direction > 0 else 1.0 - percentile
            contributions.append(risk_percentile)

            if risk_percentile >= ELEVATED:
                severity = "extreme" if risk_percentile >= EXTREME else "elevated"
                evidence.append(Evidence(
                    name=feature,
                    value=value,
                    direction=+1,
                    weight=risk_percentile,
                    detail=(
                        f"{label} is {severity}: {value:.4g}, "
                        f"{risk_percentile:.0%} of its trailing history"
                    ),
                ))
            elif risk_percentile <= 0.20:
                evidence.append(Evidence(
                    name=feature,
                    value=value,
                    direction=-1,
                    weight=1.0 - risk_percentile,
                    detail=(
                        f"{label} is calm: {value:.4g}, "
                        f"{risk_percentile:.0%} of its trailing history"
                    ),
                ))

        if not contributions:
            return AgentReport(
                agent=self.name, risk=RiskLevel.INSUFFICIENT_EVIDENCE, score=float("nan"),
                confidence=0.0, status=AgentStatus.UNAVAILABLE,
                notes=(f"no usable market features; missing {sorted(set(missing))}",),
            )

        score = 100.0 * float(np.mean(contributions))
        coverage = len(contributions) / len(self.SIGNALS)
        status = AgentStatus.OK if coverage >= 0.8 else AgentStatus.DEGRADED

        notes: list[str] = []
        if missing:
            notes.append(
                f"{len(missing)} of {len(self.SIGNALS)} signals unavailable "
                f"({', '.join(sorted(set(missing)))}); confidence reduced accordingly"
            )

        return AgentReport(
            agent=self.name,
            risk=RiskLevel.from_score(score, _bands(context)),
            score=score,
            # Confidence is coverage-weighted and rises when signals agree:
            # four indicators all screaming is worth more than one outlier.
            confidence=float(coverage * _agreement(contributions)),
            status=status,
            evidence=tuple(evidence),
            notes=tuple(notes),
            extra={"signals_used": len(contributions), "signals_total": len(self.SIGNALS)},
        )


def _agreement(values: list[float]) -> float:
    """How consistently the signals point the same way, in [0, 1]."""
    if len(values) < 2:
        return 0.5
    array = np.asarray(values)
    # Low dispersion around a common view means high agreement. The 0.5 divisor
    # maps the maximum possible std of a [0,1] variable onto the full range.
    return float(np.clip(1.0 - 2.0 * np.std(array), 0.0, 1.0))


def _bands(context: AgentContext) -> dict[str, float]:
    """Risk-band cut-points from config, with the architecture's defaults."""
    if context.config is not None:
        try:
            return dict(context.config.section("agents")["risk_bands"])
        except Exception:  # config predates the agents section
            pass
    return {"moderate": 40.0, "high": 65.0, "critical": 80.0}
