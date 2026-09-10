"""Canonical prediction serialisation and SHA-256 digest - G5 (BOLT_SPEC.md Section 9).

Determinism is mandatory: a third party recomputing the hash must get a
bit-identical result regardless of dict insertion order.

Status: not yet implemented (scheduled for Phase 5).
"""

from __future__ import annotations

_PHASE = "Phase 5"


def _not_yet(name: str) -> None:
    raise NotImplementedError(
        f"{name} is scheduled for Phase 5. BOLT never returns placeholder data: "
        "see BOLT_SPEC.md Rule 12.2 (never fabricate a result) and Rule 12.3 (fail loudly)."
    )
