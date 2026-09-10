"""Headline sentiment features (BOLT_SPEC.md Section 5).

Strictly point-in-time under Guard 2. Default backend is a deterministic finance
lexicon; the optional FinBERT backend runs locally at inference only.

Status: not yet implemented (scheduled for Phase 2).
"""

from __future__ import annotations

_PHASE = "Phase 2"


def _not_yet(name: str) -> None:
    raise NotImplementedError(
        f"{name} is scheduled for Phase 2. BOLT never returns placeholder data: "
        "see BOLT_SPEC.md Rule 12.2 (never fabricate a result) and Rule 12.3 (fail loudly)."
    )
