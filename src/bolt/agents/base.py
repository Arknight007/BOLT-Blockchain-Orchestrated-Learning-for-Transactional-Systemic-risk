"""Agent contract and shared types (ChainGuard architecture).

Every agent produces an :class:`AgentReport`: a risk level, a confidence, and a
list of :class:`Evidence` items that justify it. Reports are plain dataclasses
that serialise deterministically, because they become part of the payload that
gets hashed and committed on-chain (G5).

That constraint is what shapes this layer. An agent whose output cannot be
recomputed byte-for-byte from the same inputs cannot participate in a
cryptographic commitment: the verifier would recompute a different digest and the
proof would fail. So the agents that feed the commitment are deterministic by
construction, and the ones that only narrate are kept out of the hashed payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np


class RiskLevel(str, Enum):
    """Ordered risk bands. String-valued so they serialise canonically."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

    @property
    def rank(self) -> int:
        order = {
            RiskLevel.INSUFFICIENT_EVIDENCE: -1,
            RiskLevel.LOW: 0,
            RiskLevel.MODERATE: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.CRITICAL: 3,
        }
        return order[self]

    @classmethod
    def from_score(cls, score: float, bands: dict[str, float]) -> "RiskLevel":
        """Map a 0-100 risk score onto a band using configured cut-points."""
        if not np.isfinite(score):
            return cls.INSUFFICIENT_EVIDENCE
        if score >= bands["critical"]:
            return cls.CRITICAL
        if score >= bands["high"]:
            return cls.HIGH
        if score >= bands["moderate"]:
            return cls.MODERATE
        return cls.LOW


class AgentStatus(str, Enum):
    """Whether an agent actually ran.

    ``DEGRADED`` and ``UNAVAILABLE`` are first-class outcomes. An agent that
    cannot see its data must say so; silently returning LOW risk because no
    evidence arrived would be the worst possible failure mode for a warning
    system (Rule 12.3).
    """

    OK = "OK"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


@dataclass(frozen=True, slots=True)
class Evidence:
    """One named observation supporting or opposing a risk assessment.

    Attributes:
        name: the driver, matching a feature name where one exists.
        value: the observed value.
        direction: ``+1`` raises risk, ``-1`` lowers it, ``0`` is neutral.
        weight: how much this observation moved the agent's conclusion, 0-1.
        detail: a human-readable statement of what was observed.
    """

    name: str
    value: float
    direction: int
    weight: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": _round(self.value),
            "direction": int(self.direction),
            "weight": _round(self.weight),
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class AgentReport:
    """What every agent returns."""

    agent: str
    risk: RiskLevel
    score: float                      # 0-100
    confidence: float                 # 0-1
    status: AgentStatus
    evidence: tuple[Evidence, ...] = ()
    notes: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.status in (AgentStatus.OK, AgentStatus.DEGRADED)

    def top_evidence(self, k: int = 5) -> list[Evidence]:
        return sorted(self.evidence, key=lambda e: abs(e.weight), reverse=True)[:k]

    def to_dict(self) -> dict[str, Any]:
        """Deterministic dict for the on-chain payload."""
        return {
            "agent": self.agent,
            "risk": self.risk.value,
            "score": _round(self.score),
            "confidence": _round(self.confidence),
            "status": self.status.value,
            "evidence": [e.to_dict() for e in self.top_evidence(5)],
            "notes": list(self.notes),
        }

    def render(self) -> str:
        """Terminal-friendly block, in the style of the architecture document."""
        lines = [f"{self.agent}", f"  Risk: {self.risk.value}  ({self.score:.0f}/100)"]
        if self.status is not AgentStatus.OK:
            lines.append(f"  Status: {self.status.value}")
        if self.evidence:
            lines.append("  Evidence:")
            for item in self.top_evidence():
                arrow = {1: "^", -1: "v", 0: "-"}[item.direction]
                lines.append(f"    {arrow} {item.detail}")
        for note in self.notes:
            lines.append(f"  Note: {note}")
        lines.append(f"  Confidence: {self.confidence:.0%}")
        return "\n".join(lines)


@runtime_checkable
class Agent(Protocol):
    """The interface every agent implements."""

    name: str

    def analyse(self, context: "AgentContext") -> AgentReport:
        ...


@dataclass
class AgentContext:
    """Everything the agents are allowed to see for one (asset, as_of) decision.

    Constructed once per prediction so that no agent can reach around it for
    data that would not have been available at ``as_of``. This is Guard 1
    enforced by construction at the agent layer.
    """

    asset: str
    as_of: Any                        # pd.Timestamp
    window: Any                       # pd.DataFrame, the trailing lookback rows
    panel: Any                        # pd.DataFrame, history up to as_of ONLY
    features: dict[str, float]        # latest feature values
    X_window: np.ndarray | None = None      # (1, T, F) tensor for the models
    feature_names: tuple[str, ...] = ()
    config: Any = None
    artefacts: dict[str, Any] = field(default_factory=dict)

    def value(self, feature: str) -> float:
        """Latest value of a feature, or NaN when unavailable."""
        return float(self.features.get(feature, np.nan))

    def series(self, column: str):
        """Trailing history of one column, up to and including ``as_of``."""
        if column not in self.panel.columns:
            return None
        return self.panel[column]


def _round(value: Any, places: int = 6) -> Any:
    """Round floats for canonical serialisation; pass everything else through.

    Determinism matters here: the payload is hashed, and a float that prints
    with 17 digits on one machine and 16 on another produces a different digest.
    """
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        return round(float(value), places)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def percentile_of(series, value: float) -> float:
    """Where ``value`` sits in the trailing distribution of ``series``, in [0, 1]."""
    if series is None or len(series) == 0 or not np.isfinite(value):
        return float("nan")
    clean = np.asarray(series.dropna(), dtype=float)
    if clean.size == 0:
        return float("nan")
    return float((clean <= value).mean())
