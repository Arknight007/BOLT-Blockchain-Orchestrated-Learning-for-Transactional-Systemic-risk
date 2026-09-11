"""On-Chain Intelligence Agent (ChainGuard architecture).

Analyses what is happening inside the blockchain ecosystem: exchange flow
pressure, whale activity, network activity, stablecoin supply and valuation
turnover.

Intelligence type: **on-chain analytics**, fully deterministic. No LLM.

Honesty constraint specific to this agent: with no Etherscan key configured,
four of its five inputs are documented proxies (see ``ingest/provenance.py``).
The agent states that in its notes on every run. An on-chain agent that presents
proxy-derived evidence as ledger truth would be the most misleading component in
the system, so it says what it is looking at.
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
from bolt.ingest.provenance import REGISTRY, Provenance
from bolt.logging_setup import get_logger

log = get_logger(__name__)

ELEVATED = 0.80
EXTREME = 0.95


class OnChainIntelligenceAgent:
    """Detects unusual on-chain (and on-chain-proxy) activity."""

    name = "On-Chain Intelligence Agent"

    SIGNALS = (
        ("exchange_netflow_7", +1, "net selling pressure toward exchanges"),
        ("whale_tx_count", +1, "large-holder activity"),
        ("active_addresses", +1, "network activity"),
        ("stablecoin_netflow", -1, "stablecoin supply growth"),
        ("nvt_ratio", +1, "valuation relative to turnover"),
    )

    def analyse(self, context: AgentContext) -> AgentReport:
        evidence: list[Evidence] = []
        contributions: list[float] = []
        missing: list[str] = []
        proxies: list[str] = []

        for feature, direction, label in self.SIGNALS:
            value = context.value(feature)
            source = REGISTRY.get(feature)
            if source is not None and source.provenance is Provenance.PROXY:
                proxies.append(feature)

            if not np.isfinite(value):
                missing.append(feature)
                continue
            percentile = percentile_of(context.series(feature), value)
            if not np.isfinite(percentile):
                missing.append(feature)
                continue

            risk_percentile = percentile if direction > 0 else 1.0 - percentile
            contributions.append(risk_percentile)

            if risk_percentile >= ELEVATED:
                severity = "sharply elevated" if risk_percentile >= EXTREME else "elevated"
                qualifier = " (proxy)" if feature in proxies else ""
                evidence.append(Evidence(
                    name=feature, value=value, direction=+1, weight=risk_percentile,
                    detail=(
                        f"{label}{qualifier} is {severity}: {value:.4g}, "
                        f"{risk_percentile:.0%} of its trailing history"
                    ),
                ))
            elif risk_percentile <= 0.20:
                evidence.append(Evidence(
                    name=feature, value=value, direction=-1, weight=1.0 - risk_percentile,
                    detail=f"{label} is benign: {value:.4g}, {risk_percentile:.0%} of history",
                ))

        if not contributions:
            return AgentReport(
                agent=self.name, risk=RiskLevel.INSUFFICIENT_EVIDENCE, score=float("nan"),
                confidence=0.0, status=AgentStatus.UNAVAILABLE,
                notes=(
                    f"no usable on-chain features for {context.asset}; "
                    f"missing {sorted(set(missing))}",
                ),
            )

        score = 100.0 * float(np.mean(contributions))
        coverage = len(contributions) / len(self.SIGNALS)

        notes = []
        if proxies:
            notes.append(
                f"{len(proxies)} of {len(self.SIGNALS)} inputs are DOCUMENTED PROXIES, not "
                f"direct ledger reads ({', '.join(sorted(set(proxies)))}). No Etherscan key "
                f"is configured; see data/processed/DATA_CARD.md for what each stands in for."
            )
        if missing:
            notes.append(f"unavailable: {', '.join(sorted(set(missing)))}")

        # A report built mostly on proxies is DEGRADED, not OK. The distinction
        # is carried into the on-chain commitment so a verifier can see it.
        status = AgentStatus.OK
        if coverage < 0.8:
            status = AgentStatus.DEGRADED
        elif len(proxies) >= 3:
            status = AgentStatus.DEGRADED

        return AgentReport(
            agent=self.name,
            risk=RiskLevel.from_score(score, _bands(context)),
            score=score,
            # Proxy-derived evidence is discounted: it is correlated with the
            # true metric, not equal to it, and the confidence should say so.
            confidence=float(coverage * (0.75 if len(proxies) >= 3 else 1.0)),
            status=status,
            evidence=tuple(evidence),
            notes=tuple(notes),
            extra={"proxy_inputs": len(proxies), "signals_used": len(contributions)},
        )


def _bands(context: AgentContext) -> dict[str, float]:
    if context.config is not None:
        try:
            return dict(context.config.section("agents")["risk_bands"])
        except Exception:
            pass
    return {"moderate": 40.0, "high": 65.0, "critical": 80.0}
