"""Attribution consistency across crisis episodes - G4 (BOLT_SPEC.md Section 8).

Do the same drivers explain every crash, or does each crisis carry its own
signature? Both answers are reportable; only a fabricated one is not.

Status: not yet implemented (scheduled for Phase 4).
"""

from __future__ import annotations

_PHASE = "Phase 4"


def _not_yet(name: str) -> None:
    raise NotImplementedError(
        f"{name} is scheduled for Phase 4. BOLT never returns placeholder data: "
        "see BOLT_SPEC.md Rule 12.2 (never fabricate a result) and Rule 12.3 (fail loudly)."
    )
