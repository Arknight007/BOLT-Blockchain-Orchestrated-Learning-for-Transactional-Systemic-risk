"""Disk cache for raw API responses (BOLT_SPEC.md Section 3).

Cache key is (source, asset, start, end, frequency). Re-running the pipeline must
not re-hit any API; Phase 1 acceptance depends on a zero-call second run.

Status: not yet implemented (scheduled for Phase 1).
"""

from __future__ import annotations

_PHASE = "Phase 1"


def _not_yet(name: str) -> None:
    raise NotImplementedError(
        f"{name} is scheduled for Phase 1. BOLT never returns placeholder data: "
        "see BOLT_SPEC.md Rule 12.2 (never fabricate a result) and Rule 12.3 (fail loudly)."
    )
