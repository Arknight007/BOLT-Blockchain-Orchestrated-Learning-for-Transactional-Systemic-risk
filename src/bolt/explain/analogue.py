"""Nearest historical regime retrieval (BOLT_SPEC.md Section 8).

Status: not yet implemented (scheduled for Phase 4).
"""

from __future__ import annotations

_PHASE = "Phase 4"


def _not_yet(name: str) -> None:
    raise NotImplementedError(
        f"{name} is scheduled for Phase 4. BOLT never returns placeholder data: "
        "see BOLT_SPEC.md Rule 12.2 (never fabricate a result) and Rule 12.3 (fail loudly)."
    )
