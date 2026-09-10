"""Common model interface (BOLT_SPEC.md Section 6).

Every model sees the SAME inputs. Tabular models flatten the window tensor via
flatten_windows(); if the inputs differed the comparison would not be fair.

Status: not yet implemented (scheduled for Phase 3).
"""

from __future__ import annotations

_PHASE = "Phase 3"


def _not_yet(name: str) -> None:
    raise NotImplementedError(
        f"{name} is scheduled for Phase 3. BOLT never returns placeholder data: "
        "see BOLT_SPEC.md Rule 12.2 (never fabricate a result) and Rule 12.3 (fail loudly)."
    )
