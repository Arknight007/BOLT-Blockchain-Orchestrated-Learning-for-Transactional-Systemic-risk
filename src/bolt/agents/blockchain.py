"""Blockchain Audit Agent - G5 (ChainGuard architecture).

Intelligence type: **deterministic code**. No model, no heuristic, no judgement.
This agent's only job is to turn a finished decision into a canonical payload, a
SHA-256 digest, and an on-chain commitment - and that job must be perfectly
reproducible or the commitment is worthless.

What gets committed is the digest, not the data. The payload is published
alongside; a verifier recomputes the hash and compares. Storing raw market data
on-chain would cost a fortune and add no security whatsoever.

What makes this the strongest claim in the project: the chain timestamps the
commitment, and a block timestamp cannot be backdated. So the record shows the
prediction existed at that moment - before the outcome was known - which is
exactly what no backtest in the literature can demonstrate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bolt.agents.base import AgentContext, AgentReport, AgentStatus, Evidence, RiskLevel
from bolt.chain.payload import Prediction, feature_hash, prediction_id
from bolt.logging_setup import get_logger
from bolt.version import UNKNOWN, git_commit_sha

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CommitmentRecord:
    prediction: Prediction
    payload_path: Path
    digest: str
    committed: bool
    tx_hash: str | None = None
    block_time: int | None = None
    explorer_url: str | None = None
    reason: str = ""


class BlockchainAuditAgent:
    """Builds the canonical payload, hashes it, and optionally commits on-chain."""

    name = "Blockchain Audit Agent"

    def __init__(self, cfg, client=None) -> None:
        self.cfg = cfg
        self.client = client

    def build_prediction(
        self,
        context: AgentContext,
        decision: AgentReport,
        reports: list[AgentReport],
        model_name: str = "ensemble",
        expected_severity: dict | None = None,
    ) -> Prediction:
        """Assemble the committable payload from a finished decision."""
        code_version = git_commit_sha(self.cfg.repo_root)
        if code_version == UNKNOWN:
            raise ValueError(
                "cannot determine the git commit SHA. Rule 12.5 requires every committed "
                "prediction to be traceable to exact code; committing an untraceable "
                "prediction would defeat the purpose of the commitment."
            )

        as_of = context.as_of.date().isoformat()
        identifier = prediction_id(context.asset, as_of, model_name, code_version)

        drivers = [
            {
                "feature": item.name,
                "contribution": round(float(item.weight), 6),
                "direction": int(item.direction),
            }
            for item in decision.top_evidence(5)
        ]
        skeptic = next((r for r in reports if r.agent == "Skeptic Agent"), None)
        counter = (
            [{"name": e.name, "detail": e.detail, "weight": round(float(e.weight), 6)}
             for e in skeptic.top_evidence(3)]
            if skeptic else []
        )

        return Prediction(
            prediction_id=identifier,
            asset=context.asset,
            as_of_date=as_of,
            horizon_days=int(self.cfg.horizon_days),
            risk_score=round(float(decision.score), 6),
            probability=round(float(decision.score) / 100.0, 6),
            severity_band=decision.risk.value,
            top_drivers=drivers,
            model_name=model_name,
            model_version=code_version,
            feature_hash=feature_hash(context.features),
            code_version=code_version,
            agent_reports=[r.to_dict() for r in reports],
            counter_evidence=counter,
            expected_severity=expected_severity or {},
        )

    def analyse(
        self,
        context: AgentContext,
        decision: AgentReport,
        reports: list[AgentReport],
        commit: bool = False,
        model_name: str = "ensemble",
        expected_severity: dict | None = None,
    ) -> tuple[AgentReport, CommitmentRecord]:
        prediction = self.build_prediction(
            context, decision, reports, model_name, expected_severity
        )
        digest = prediction.digest_hex()
        payload_path = prediction.save(self.cfg.path("predictions"))

        record = CommitmentRecord(
            prediction=prediction, payload_path=Path(payload_path), digest=digest,
            committed=False, reason="payload written; not committed (--commit not set)",
        )

        if commit:
            record = self._commit(prediction, Path(payload_path), digest)

        evidence = [
            Evidence("digest", 0.0, 0, 1.0, f"SHA-256 {digest[:22]}..."),
            Evidence("payload", 0.0, 0, 1.0, f"payload written to {payload_path}"),
        ]
        if record.committed:
            evidence.append(Evidence(
                "commitment", 0.0, 0, 1.0,
                f"committed on-chain at block time {record.block_time}, tx {record.tx_hash}",
            ))

        return (
            AgentReport(
                agent=self.name, risk=decision.risk, score=decision.score,
                confidence=1.0,
                status=AgentStatus.OK if record.committed else AgentStatus.DEGRADED,
                evidence=tuple(evidence), notes=(record.reason,),
                extra={
                    "prediction_id": prediction.prediction_id,
                    "digest": digest,
                    "committed": record.committed,
                    "tx_hash": record.tx_hash,
                },
            ),
            record,
        )

    def _commit(self, prediction: Prediction, payload_path: Path, digest: str) -> CommitmentRecord:
        if self.client is None:
            try:
                from bolt.chain.client import RegistryClient

                self.client = RegistryClient(self.cfg)
            except Exception as exc:                      # noqa: BLE001 - reported
                return CommitmentRecord(
                    prediction, payload_path, digest, committed=False,
                    reason=f"on-chain commitment unavailable: {exc}",
                )
        try:
            receipt = self.client.commit(prediction.prediction_id, digest)
        except Exception as exc:                          # noqa: BLE001 - reported
            return CommitmentRecord(
                prediction, payload_path, digest, committed=False,
                reason=f"commit failed: {exc}",
            )
        return CommitmentRecord(
            prediction, payload_path, digest, committed=True,
            tx_hash=receipt.tx_hash, block_time=receipt.block_time,
            explorer_url=receipt.explorer_url,
            reason=(
                f"committed on-chain before the outcome is known; block timestamp "
                f"{receipt.block_time} cannot be backdated"
            ),
        )
