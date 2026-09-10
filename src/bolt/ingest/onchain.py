"""On-chain metrics from Etherscan and public endpoints (BOLT_SPEC.md Section 3).

Where a metric has no free source, it is implemented as a DOCUMENTED proxy and
recorded in data/processed/DATA_CARD.md. Never a silent substitution.

Status: not yet implemented (scheduled for Phase 1).
"""

from __future__ import annotations

_PHASE = "Phase 1"


def _not_yet(name: str) -> None:
    raise NotImplementedError(
        f"{name} is scheduled for Phase 1. BOLT never returns placeholder data: "
        "see BOLT_SPEC.md Rule 12.2 (never fabricate a result) and Rule 12.3 (fail loudly)."
    )
