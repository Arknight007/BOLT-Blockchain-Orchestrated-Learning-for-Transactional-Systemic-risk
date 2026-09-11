"""Canonical prediction serialisation and SHA-256 digest - G5 (BOLT_SPEC.md Section 9).

Determinism is mandatory. A third party recomputing the hash must get a
bit-identical result regardless of dict insertion order, platform, or Python
version - otherwise the commitment proves nothing, because a mismatch would be
indistinguishable from tampering.

Three rules make that hold:

* ``sort_keys=True`` removes insertion-order dependence.
* ``separators=(',', ':')`` removes whitespace variation.
* ``ensure_ascii=True`` removes encoding variation.

Floats are rounded before serialisation, because a value that prints with 17
significant digits on one machine and 16 on another produces a different digest
for the same prediction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

FLOAT_PRECISION = 8

#: Fields every payload must carry (BOLT_SPEC.md Section 9).
REQUIRED_FIELDS = (
    "prediction_id", "asset", "as_of_date", "horizon_days", "risk_score",
    "probability", "severity_band", "top_drivers", "model_name", "model_version",
    "feature_hash", "code_version",
)


class PayloadError(ValueError):
    """The payload is malformed and must not be committed."""


def _canonicalise(value: Any) -> Any:
    """Recursively normalise a value for deterministic serialisation."""
    if isinstance(value, dict):
        return {str(k): _canonicalise(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonicalise(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None                       # NaN and inf have no JSON form
        return round(value, FLOAT_PRECISION)
    if isinstance(value, int):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None:
        return None
    return str(value)


def canonical_payload(prediction: dict) -> bytes:
    """Deterministic serialisation of a prediction.

    Raises:
        PayloadError: when a required field is missing. An incomplete payload
            must never be committed: the commitment would be unverifiable.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in prediction]
    if missing:
        raise PayloadError(
            f"payload is missing required field(s) {missing}. Every field in "
            f"REQUIRED_FIELDS must be present for the commitment to be verifiable."
        )
    normalised = _canonicalise(prediction)
    return json.dumps(
        normalised, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def digest(payload: bytes) -> bytes:
    """SHA-256 of the canonical payload."""
    return hashlib.sha256(payload).digest()


def digest_hex(payload: bytes) -> str:
    """SHA-256 as a 0x-prefixed hex string, ready for the contract."""
    return "0x" + hashlib.sha256(payload).hexdigest()


def prediction_id(asset: str, as_of: str, model: str, code_version: str) -> str:
    """Deterministic 32-byte id derived from what the prediction IS.

    Deriving rather than randomising means the same prediction cannot be
    committed twice under different ids to manufacture a better track record -
    the second attempt hits the contract's overwrite revert.
    """
    material = f"{asset}|{as_of}|{model}|{code_version}".encode("utf-8")
    return "0x" + hashlib.sha256(material).hexdigest()


def feature_hash(features: dict[str, float]) -> str:
    """SHA-256 over the feature vector that produced the prediction.

    Lets a verifier confirm the prediction was made from the stated inputs, not
    merely that some prediction with this text existed.
    """
    blob = json.dumps(
        _canonicalise(features), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "0x" + hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True, slots=True)
class Prediction:
    """A complete, committable prediction."""

    prediction_id: str
    asset: str
    as_of_date: str
    horizon_days: int
    risk_score: float
    probability: float
    severity_band: str
    top_drivers: list[dict]
    model_name: str
    model_version: str
    feature_hash: str
    code_version: str
    agent_reports: list[dict] = field(default_factory=list)
    counter_evidence: list[dict] = field(default_factory=list)
    expected_severity: dict = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict:
        payload = {
            "prediction_id": self.prediction_id,
            "asset": self.asset,
            "as_of_date": self.as_of_date,
            "horizon_days": self.horizon_days,
            "risk_score": self.risk_score,
            "probability": self.probability,
            "severity_band": self.severity_band,
            "top_drivers": self.top_drivers,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "feature_hash": self.feature_hash,
            "code_version": self.code_version,
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(),
        }
        # The agent evidence and the skeptic's counter-evidence are committed
        # too: the record should show what the system knew might be wrong at the
        # moment it committed, not just what it predicted.
        if self.agent_reports:
            payload["agent_reports"] = self.agent_reports
        if self.counter_evidence:
            payload["counter_evidence"] = self.counter_evidence
        if self.expected_severity:
            payload["expected_severity"] = self.expected_severity
        return payload

    def canonical(self) -> bytes:
        return canonical_payload(self.to_dict())

    def digest_hex(self) -> str:
        return digest_hex(self.canonical())

    def save(self, directory) -> "object":
        """Write the payload JSON that a verifier will hash."""
        from pathlib import Path

        path = Path(directory) / f"{self.prediction_id[:18]}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.canonical())
        return path
