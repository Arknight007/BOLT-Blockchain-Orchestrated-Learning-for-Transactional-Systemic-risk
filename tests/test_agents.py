"""ChainGuard agent behaviour.

The properties tested here are the ones that make the agent layer defensible
rather than decorative:

* An agent that cannot see its data says so, instead of returning LOW risk.
  Silently reporting "no crash" because no evidence arrived would be the worst
  possible failure mode for a warning system.
* The skeptic actually finds counter-evidence, and that counter-evidence
  actually moves the decision.
* The system can decline to answer.
* Agent reports serialise deterministically, because they are hashed into the
  on-chain payload.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bolt.agents.base import AgentContext, AgentReport, AgentStatus, Evidence, RiskLevel
from bolt.agents.decision import DecisionAgent
from bolt.agents.evaluation import EvaluationAgent
from bolt.agents.health import HealthAgent, population_stability_index
from bolt.agents.market import MarketIntelligenceAgent
from bolt.agents.news import NewsEventAgent
from bolt.agents.onchain import OnChainIntelligenceAgent
from bolt.agents.orchestrator import RiskOrchestratorAgent
from bolt.agents.quant import QuantitativeAgent
from bolt.agents.skeptic import SkepticAgent

FEATURES = [
    "realized_vol_7", "realized_vol_30", "volume_z_20", "atr_14", "drawdown_depth",
    "momentum_10", "rsi_14", "exchange_netflow_7", "whale_tx_count", "active_addresses",
    "stablecoin_netflow", "nvt_ratio", "headline_polarity_1d", "headline_polarity_7d",
    "negative_ratio_7d", "close",
]


def _context(stressed: bool = True, n: int = 400, missing: list[str] | None = None):
    """A synthetic context: either a calm regime or a stressed one."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    panel = pd.DataFrame(index=dates)
    for name in FEATURES:
        panel[name] = rng.normal(0.0, 1.0, n)
    panel["close"] = 100.0 + np.arange(n) * 0.1

    features = {}
    for name in FEATURES:
        if missing and name in missing:
            features[name] = float("nan")
        elif stressed:
            # Push risk-raising features high and risk-lowering features low.
            features[name] = 5.0 if name not in ("momentum_10", "rsi_14",
                                                 "headline_polarity_1d",
                                                 "headline_polarity_7d",
                                                 "stablecoin_netflow") else -5.0
        else:
            features[name] = 0.0

    return AgentContext(
        asset="BTC", as_of=dates[-1], window=panel.tail(30), panel=panel,
        features=features, feature_names=tuple(FEATURES), config=None,
    )


# ---------------------------------------------------------------------------
# Evidence agents
# ---------------------------------------------------------------------------

def test_market_agent_flags_a_stressed_regime():
    report = MarketIntelligenceAgent().analyse(_context(stressed=True))
    assert report.status is AgentStatus.OK
    assert report.score > 65, f"stressed regime scored only {report.score}"
    assert report.risk.rank >= RiskLevel.HIGH.rank
    assert report.evidence, "a high score with no evidence is not an explanation"


def test_market_agent_stays_calm_in_a_calm_regime():
    report = MarketIntelligenceAgent().analyse(_context(stressed=False))
    assert report.score < 65


def test_agent_reports_unavailable_rather_than_low_when_blind():
    """The critical failure mode: missing data must never read as 'safe'."""
    context = _context(missing=FEATURES)
    for agent in (MarketIntelligenceAgent(), OnChainIntelligenceAgent(), NewsEventAgent()):
        report = agent.analyse(context)
        assert report.status is AgentStatus.UNAVAILABLE
        assert report.risk is RiskLevel.INSUFFICIENT_EVIDENCE
        assert report.risk is not RiskLevel.LOW, "blindness must not present as safety"


def test_onchain_agent_declares_its_proxies():
    """An on-chain agent presenting proxy data as ledger truth would mislead."""
    report = OnChainIntelligenceAgent().analyse(_context())
    assert report.extra["proxy_inputs"] >= 3
    assert any("PROX" in note.upper() for note in report.notes)
    assert report.status is AgentStatus.DEGRADED


def test_news_agent_declares_event_identification_unimplemented():
    report = NewsEventAgent().analyse(_context())
    assert report.extra["event_identification"] == "NOT_IMPLEMENTED"
    assert report.extra["major_event"] is None
    assert any("NOT IMPLEMENTED" in note for note in report.notes)
    assert report.confidence <= 0.55, "an agent with no event layer must not be confident"


def test_quant_agent_is_unavailable_without_models():
    report = QuantitativeAgent({}).analyse(_context())
    assert report.status is AgentStatus.UNAVAILABLE
    assert "bolt train" in " ".join(report.notes)


def test_quant_agent_reports_model_disagreement():
    class Stub:
        def __init__(self, value): self.value = value
        def predict_proba(self, X): return np.array([self.value])

    context = _context()
    context.X_window = np.zeros((1, 30, len(FEATURES)))
    report = QuantitativeAgent({"lstm": Stub(0.20), "xgb": Stub(0.85)}).analyse(context)
    assert report.extra["spread"] == pytest.approx(0.65, abs=1e-6)
    assert any("DISAGREE" in note for note in report.notes)
    assert report.confidence < 0.5, "wide disagreement must reduce confidence"


# ---------------------------------------------------------------------------
# Orchestrator, skeptic, decision
# ---------------------------------------------------------------------------

def _report(agent, score, confidence=0.9, status=AgentStatus.OK):
    return AgentReport(agent=agent, risk=RiskLevel.from_score(
        score, {"moderate": 40, "high": 65, "critical": 80}),
        score=score, confidence=confidence, status=status)


def test_orchestrator_needs_two_independent_sources():
    context = _context()
    reports = [_report("Market Intelligence Agent", 90.0)]
    result = RiskOrchestratorAgent().analyse(context, reports)
    assert result.risk is RiskLevel.INSUFFICIENT_EVIDENCE
    assert result.status is AgentStatus.UNAVAILABLE


def test_orchestrator_rewards_agreement_over_a_single_loud_voice():
    context = _context()
    agreeing = [
        _report("Market Intelligence Agent", 70.0),
        _report("On-Chain Intelligence Agent", 70.0),
        _report("Quantitative Agent", 70.0),
    ]
    divided = [
        _report("Market Intelligence Agent", 10.0),
        _report("On-Chain Intelligence Agent", 10.0),
        _report("Quantitative Agent", 95.0),
    ]
    orchestrator = RiskOrchestratorAgent()
    consensus = orchestrator.analyse(context, agreeing)
    conflict = orchestrator.analyse(context, divided)
    assert consensus.confidence > conflict.confidence
    assert consensus.extra["consensus"] > conflict.extra["consensus"]


def test_orchestrator_notes_when_models_lack_corroboration():
    context = _context()
    reports = [
        _report("Quantitative Agent", 85.0),
        _report("Market Intelligence Agent", 20.0),
        _report("On-Chain Intelligence Agent", 20.0),
    ]
    result = RiskOrchestratorAgent().analyse(context, reports)
    assert "NO independent evidence" in " ".join(result.notes)


def test_skeptic_finds_counter_evidence_and_it_lowers_confidence():
    context = _context()
    preliminary = _report("Risk Orchestrator", 80.0, confidence=0.9)
    reports = [
        _report("Quantitative Agent", 80.0, status=AgentStatus.DEGRADED),
        _report("On-Chain Intelligence Agent", 80.0, status=AgentStatus.DEGRADED),
    ]
    skeptic = SkepticAgent().analyse(context, preliminary, reports)
    assert skeptic.evidence, "the skeptic must actually challenge something"
    assert skeptic.extra["penalty"] > 0

    decided = DecisionAgent().analyse(context, preliminary, skeptic, reports)
    assert decided.confidence < preliminary.confidence, (
        "counter-evidence that does not move the decision is theatre"
    )


def test_skeptic_penalty_is_capped():
    """An unbounded penalty would let the skeptic veto every warning."""
    from bolt.agents.skeptic import MAX_PENALTY

    context = _context()
    preliminary = _report("Risk Orchestrator", 95.0, confidence=1.0)
    reports = [_report(f"Agent {i}", 95.0, status=AgentStatus.DEGRADED) for i in range(10)]
    skeptic = SkepticAgent().analyse(context, preliminary, reports)
    assert skeptic.extra["penalty"] <= MAX_PENALTY


def test_decision_agent_abstains_when_confidence_collapses():
    """INSUFFICIENT_EVIDENCE is a real outcome, not an error path."""
    context = _context()
    preliminary = _report("Risk Orchestrator", 70.0, confidence=0.20)
    skeptic = AgentReport(
        agent="Skeptic Agent", risk=RiskLevel.MODERATE, score=50.0, confidence=1.0,
        status=AgentStatus.OK, extra={"penalty": 0.5, "challenges": 4},
    )
    decided = DecisionAgent().analyse(context, preliminary, skeptic, [])
    assert decided.risk is RiskLevel.INSUFFICIENT_EVIDENCE
    assert decided.extra["decision"] == "ABSTAIN"


def test_decision_agent_issues_a_warning_on_strong_uncontested_evidence():
    context = _context()
    preliminary = _report("Risk Orchestrator", 85.0, confidence=0.9)
    skeptic = AgentReport(
        agent="Skeptic Agent", risk=RiskLevel.LOW, score=0.0, confidence=1.0,
        status=AgentStatus.OK, extra={"penalty": 0.0, "challenges": 0},
    )
    decided = DecisionAgent().analyse(context, preliminary, skeptic, [])
    assert decided.risk.rank >= RiskLevel.HIGH.rank
    assert decided.extra["decision"] == "ALERT"


def test_skeptic_lowers_confidence_but_does_not_invent_calm():
    """Counter-evidence makes a warning less certain, not the conditions milder."""
    context = _context()
    preliminary = _report("Risk Orchestrator", 90.0, confidence=0.9)
    skeptic = AgentReport(
        agent="Skeptic Agent", risk=RiskLevel.MODERATE, score=30.0, confidence=1.0,
        status=AgentStatus.OK, extra={"penalty": 0.3, "challenges": 2},
    )
    decided = DecisionAgent().analyse(context, preliminary, skeptic, [])
    assert decided.score > preliminary.score * 0.5, "the score must not collapse to calm"
    assert decided.confidence < preliminary.confidence


# ---------------------------------------------------------------------------
# Serialisation - these reports get hashed
# ---------------------------------------------------------------------------

def test_agent_report_serialises_deterministically():
    from bolt.chain.payload import canonical_payload

    report = AgentReport(
        agent="Market Intelligence Agent", risk=RiskLevel.HIGH, score=78.25,
        confidence=0.83, status=AgentStatus.OK,
        evidence=(Evidence("realized_vol_7", 1.23456789, 1, 0.9, "volatility elevated"),),
    )
    first, second = report.to_dict(), report.to_dict()
    assert first == second

    payload = {
        "prediction_id": "0x1", "asset": "BTC", "as_of_date": "2022-11-05",
        "horizon_days": 14, "risk_score": 78.25, "probability": 0.7825,
        "severity_band": "HIGH", "top_drivers": [], "model_name": "lstm",
        "model_version": "abc", "feature_hash": "0x1", "code_version": "abc",
        "agent_reports": [first],
    }
    assert canonical_payload(payload) == canonical_payload({**payload, "agent_reports": [second]})


def test_non_finite_values_do_not_leak_into_a_report():
    report = AgentReport(
        agent="X", risk=RiskLevel.LOW, score=float("nan"), confidence=float("inf"),
        status=AgentStatus.OK,
    )
    as_dict = report.to_dict()
    assert as_dict["score"] is None and as_dict["confidence"] is None


# ---------------------------------------------------------------------------
# Evaluation and health
# ---------------------------------------------------------------------------

def test_evaluation_agent_classifies_outcomes():
    dates = pd.date_range("2022-11-05", periods=20, freq="D", tz="UTC")
    crashed = pd.Series(np.linspace(100.0, 70.0, 20), index=dates)
    calm = pd.Series(np.linspace(100.0, 102.0, 20), index=dates)

    agent = EvaluationAgent(threshold=0.20)
    warning = {"prediction_id": "a", "asset": "BTC", "as_of_date": "2022-11-05",
               "horizon_days": 14, "severity_band": "HIGH", "probability": 0.8}
    quiet = {**warning, "prediction_id": "b", "severity_band": "LOW", "probability": 0.1}

    assert agent.resolve(warning, crashed).classification == "TRUE_POSITIVE"
    assert agent.resolve(warning, calm).classification == "FALSE_POSITIVE"
    assert agent.resolve(quiet, crashed).classification == "FALSE_NEGATIVE"
    assert agent.resolve(quiet, calm).classification == "TRUE_NEGATIVE"


def test_evaluation_agent_leaves_open_horizons_unresolved():
    dates = pd.date_range("2022-11-05", periods=1, freq="D", tz="UTC")
    agent = EvaluationAgent(threshold=0.20)
    outcome = agent.resolve(
        {"prediction_id": "a", "asset": "BTC", "as_of_date": "2022-11-05",
         "horizon_days": 14, "severity_band": "HIGH", "probability": 0.8},
        pd.Series([100.0], index=dates),
    )
    assert outcome.classification == "UNRESOLVED" and not outcome.resolved


def test_evaluation_agent_never_computes_accuracy():
    agent = EvaluationAgent(threshold=0.20)
    dates = pd.date_range("2022-11-05", periods=20, freq="D", tz="UTC")
    prices = pd.Series(np.linspace(100.0, 70.0, 20), index=dates)
    outcomes = [
        agent.resolve({"prediction_id": str(i), "asset": "BTC",
                       "as_of_date": "2022-11-05", "horizon_days": 14,
                       "severity_band": "HIGH" if i % 2 else "LOW",
                       "probability": 0.8 if i % 2 else 0.1}, prices)
        for i in range(6)
    ]
    metrics = agent.analyse(outcomes).extra["metrics"]
    assert not {"accuracy", "accuracy_score"} & set(metrics)


def test_psi_detects_a_shifted_distribution():
    rng = np.random.default_rng(0)
    reference = rng.normal(0.0, 1.0, 2000)
    assert population_stability_index(reference, rng.normal(0.0, 1.0, 2000)) < 0.10
    assert population_stability_index(reference, rng.normal(2.0, 1.0, 2000)) > 0.25


def test_health_agent_recommends_retraining_on_multiple_failures():
    from bolt.agents.health import DriftReading

    drift = [DriftReading(f"f{i}", 0.4, "major") for i in range(3)]
    report = HealthAgent().analyse(drift=drift)
    assert "RETRAINING RECOMMENDED" in " ".join(report.notes)


def test_health_agent_reports_healthy_when_nothing_moved():
    from bolt.agents.health import DriftReading

    drift = [DriftReading(f"f{i}", 0.01, "stable") for i in range(5)]
    report = HealthAgent().analyse(drift=drift)
    assert "Healthy" in " ".join(report.notes)
