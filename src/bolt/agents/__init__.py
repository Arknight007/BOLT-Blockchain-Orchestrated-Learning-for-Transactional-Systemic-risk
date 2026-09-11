"""Multi-agent evidence layer (ChainGuard architecture).

Specialised agents independently analyse market, on-chain and external evidence;
the quantitative agent runs the neural and gradient-boosted models; an
orchestrator aggregates; a skeptic challenges; a decision agent rules; and the
blockchain agent commits the result before the outcome is known.

**Design principle: not every agent is an LLM agent.** That is the common failure
mode of this architecture - eleven language-model calls wearing different hats,
none of which can be reproduced, tested, or defended to an examiner. Here each
agent uses the *simplest intelligence that does its job*:

===========================  ==================  ==========================
Agent                        Intelligence        Status in this build
===========================  ==================  ==========================
MarketIntelligenceAgent      rules + analytics   IMPLEMENTED, deterministic
OnChainIntelligenceAgent     on-chain analytics  IMPLEMENTED, deterministic
QuantitativeAgent            LSTM + XGBoost      IMPLEMENTED, the models
RiskOrchestratorAgent        weighted evidence   IMPLEMENTED, deterministic
SkepticAgent                 rules               IMPLEMENTED, deterministic
DecisionAgent                thresholds          IMPLEMENTED, deterministic
BlockchainAuditAgent         deterministic code  IMPLEMENTED (G5)
EvaluationAgent              analytics           IMPLEMENTED
HealthAgent                  drift monitoring    IMPLEMENTED
---------------------------  ------------------  --------------------------
NewsEventAgent               NLP / LLM           PARTIAL - lexicon only
ExplanationAgent             LLM narrative       INTERFACE ONLY
===========================  ==================  ==========================

The split is principled rather than arbitrary: every agent whose output feeds
the on-chain commitment is deterministic and reproducible, because a committed
prediction that cannot be recomputed from the same inputs proves nothing. The
two agents that remain interfaces are the two whose only job is to produce
*prose*, which is the one thing a cryptographic commitment does not need.
"""

from bolt.agents.base import (
    Agent,
    AgentReport,
    AgentStatus,
    Evidence,
    RiskLevel,
)

__all__ = ["Agent", "AgentReport", "AgentStatus", "Evidence", "RiskLevel"]
