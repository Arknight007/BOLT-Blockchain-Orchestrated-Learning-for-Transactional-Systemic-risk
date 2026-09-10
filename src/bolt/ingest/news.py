"""Headline ingestion from CryptoPanic, RSS archives and Reddit (BOLT_SPEC.md Section 3).

The publication timestamp is load-bearing: Guard 2 rejects any headline whose
published_at is later than the feature date it would contribute to.

Status: not yet implemented (scheduled for Phase 1).
"""

from __future__ import annotations

_PHASE = "Phase 1"


def _not_yet(name: str) -> None:
    raise NotImplementedError(
        f"{name} is scheduled for Phase 1. BOLT never returns placeholder data: "
        "see BOLT_SPEC.md Rule 12.2 (never fabricate a result) and Rule 12.3 (fail loudly)."
    )
