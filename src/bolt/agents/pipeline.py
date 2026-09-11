"""Agent orchestration pipeline (ChainGuard architecture).

Runs the full chain for one (asset, as_of) decision:

    Market ─┐
    OnChain ─┼─> Orchestrator ─> Skeptic ─> Decision ─> Blockchain ─> commitment
    News   ─┤
    Quant  ─┘                        └─> Explanation (narrative, not committed)

Point-in-time is enforced by CONSTRUCTION here, not by convention: the context is
built by slicing the panel at ``as_of`` before any agent runs, so no agent can
reach data it should not have seen. That is Guard 1 applied at the agent layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from bolt.agents.base import AgentContext, AgentReport, RiskLevel
from bolt.agents.blockchain import BlockchainAuditAgent, CommitmentRecord
from bolt.agents.decision import DecisionAgent
from bolt.agents.explanation import ExplanationAgent
from bolt.agents.market import MarketIntelligenceAgent
from bolt.agents.news import NewsEventAgent
from bolt.agents.onchain import OnChainIntelligenceAgent
from bolt.agents.orchestrator import RiskOrchestratorAgent
from bolt.agents.quant import QuantitativeAgent
from bolt.agents.skeptic import SkepticAgent
from bolt.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class PipelineResult:
    """Everything one agent run produced."""

    context: AgentContext
    reports: list[AgentReport]          # the four evidence agents only
    preliminary: AgentReport
    skeptic: AgentReport
    decision: AgentReport
    explanation: AgentReport | None = None
    audit: AgentReport | None = None
    commitment: CommitmentRecord | None = None
    analogues: list[dict] = field(default_factory=list)

    @property
    def all_reports(self) -> list[AgentReport]:
        """Every report produced, in chain order, each exactly once."""
        chain = list(self.reports) + [self.preliminary, self.skeptic, self.decision]
        if self.explanation is not None:
            chain.append(self.explanation)
        if self.audit is not None:
            chain.append(self.audit)
        return chain

    def render(self) -> str:
        """The full chain, formatted the way the architecture document shows it."""
        blocks = [r.render() for r in self.all_reports]
        if self.commitment is not None:
            state = "COMMITTED" if self.commitment.committed else "NOT COMMITTED"
            blocks.append(
                f"Blockchain Audit Agent\n"
                f"  prediction id : {self.commitment.prediction.prediction_id[:26]}...\n"
                f"  digest        : {self.commitment.digest[:26]}...\n"
                f"  payload       : {self.commitment.payload_path}\n"
                f"  status        : {state}\n"
                f"  {self.commitment.reason}"
            )
        return "\n\n".join(blocks)


def build_context(
    cfg,
    panel: pd.DataFrame,
    asset: str,
    as_of: pd.Timestamp,
    feature_names: list[str],
) -> AgentContext:
    """Slice the panel at ``as_of``. Nothing later than that date is visible."""
    as_of = pd.Timestamp(as_of)
    as_of = (as_of.tz_localize("UTC") if as_of.tzinfo is None
             else as_of.tz_convert("UTC")).normalize()
    try:
        block = panel.xs(asset, level="asset")
    except KeyError as exc:
        raise ValueError(f"{asset} is not in the panel") from exc

    history = block.loc[:as_of]
    if history.empty:
        raise ValueError(f"no data for {asset} on or before {as_of.date()}")

    lookback = cfg.lookback_days
    window = history.tail(lookback)
    latest = history.iloc[-1]
    features = {name: float(latest.get(name, np.nan)) for name in feature_names}

    X_window = None
    if len(window) == lookback:
        matrix = window[feature_names].to_numpy(dtype=np.float64)
        if np.isfinite(matrix).all():
            X_window = matrix.reshape(1, lookback, len(feature_names))
        else:
            log.warning("%s at %s: window has missing features; the quantitative agent "
                        "will report UNAVAILABLE rather than score an imputed window",
                        asset, as_of.date())

    return AgentContext(
        asset=asset, as_of=as_of, window=window, panel=history, features=features,
        X_window=X_window, feature_names=tuple(feature_names), config=cfg,
    )


class ChainGuardPipeline:
    """The full multi-agent chain."""

    def __init__(self, cfg, models: dict | None = None, scaler=None,
                 narrative_backend=None) -> None:
        self.cfg = cfg
        self.market = MarketIntelligenceAgent()
        self.onchain = OnChainIntelligenceAgent()
        self.news = NewsEventAgent()
        self.quant = QuantitativeAgent(models or {}, scaler)
        weights = None
        try:
            weights = dict(cfg.section("agents")["weights"])
        except Exception:  # config predates the agents section
            pass
        self.orchestrator = RiskOrchestratorAgent(weights)
        self.skeptic = SkepticAgent()
        self.decision = DecisionAgent()
        self.explanation = ExplanationAgent(narrative_backend)
        self.blockchain = BlockchainAuditAgent(cfg)

    def run(
        self,
        context: AgentContext,
        commit: bool = False,
        analogues: list[dict] | None = None,
        attribution: pd.DataFrame | None = None,
    ) -> PipelineResult:
        """Run every agent in order and return the complete result."""
        reports = [
            self.market.analyse(context),
            self.onchain.analyse(context),
            self.news.analyse(context),
            self.quant.analyse(context),
        ]

        preliminary = self.orchestrator.analyse(context, reports)
        skeptic = self.skeptic.analyse(context, preliminary, reports, analogues)
        decision = self.decision.analyse(context, preliminary, skeptic, reports)

        with_skeptic = reports + [skeptic]
        explanation = self.explanation.analyse(
            context, decision, with_skeptic, attribution, analogues
        )

        severity = {}
        if decision.risk.rank >= RiskLevel.MODERATE.rank:
            severity = self.quant.expected_severity(context, decision.score / 100.0)

        audit_report, commitment = self.blockchain.analyse(
            context, decision, with_skeptic, commit=commit, expected_severity=severity
        )

        return PipelineResult(
            context=context, reports=reports,
            preliminary=preliminary, skeptic=skeptic, decision=decision,
            explanation=explanation, audit=audit_report, commitment=commitment,
            analogues=analogues or [],
        )
