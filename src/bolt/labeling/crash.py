"""Crash labelling - G1 (BOLT_SPEC.md Section 5).

A label is 1 when the minimum close over (t, t+horizon] falls at least `threshold`
below close[t]. The last `horizon` dates are NaN and MUST be dropped, not filled.

Status: not yet implemented (scheduled for Phase 2).
"""

from __future__ import annotations

_PHASE = "Phase 2"


def _not_yet(name: str) -> None:
    raise NotImplementedError(
        f"{name} is scheduled for Phase 2. BOLT never returns placeholder data: "
        "see BOLT_SPEC.md Rule 12.2 (never fabricate a result) and Rule 12.3 (fail loudly)."
    )
