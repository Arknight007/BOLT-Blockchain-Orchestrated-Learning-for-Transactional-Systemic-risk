"""(date, asset) panel construction and Guard 1 (BOLT_SPEC.md Sections 3-4).

Holds assert_point_in_time(), the runtime assertion that no feature value at date
t was computed from data timestamped after t.

Status: not yet implemented (scheduled for Phase 1).
"""

from __future__ import annotations

_PHASE = "Phase 1"


def _not_yet(name: str) -> None:
    raise NotImplementedError(
        f"{name} is scheduled for Phase 1. BOLT never returns placeholder data: "
        "see BOLT_SPEC.md Rule 12.2 (never fabricate a result) and Rule 12.3 (fail loudly)."
    )
