"""Volatility-threshold rule, the non-ML baseline (BOLT_SPEC.md Section 6).

Exists to prove the ML models earn their complexity. If it wins, that is the
result and it gets reported (Rule 12.7).

Status: not yet implemented (scheduled for Phase 3).
"""

from __future__ import annotations

_PHASE = "Phase 3"


def _not_yet(name: str) -> None:
    raise NotImplementedError(
        f"{name} is scheduled for Phase 3. BOLT never returns placeholder data: "
        "see BOLT_SPEC.md Rule 12.2 (never fabricate a result) and Rule 12.3 (fail loudly)."
    )
