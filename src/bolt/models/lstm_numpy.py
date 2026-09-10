"""LSTM implemented from scratch in NumPy (BOLT_SPEC.md Section 6).

No PyTorch, no TensorFlow. The from-scratch implementation is part of the
contribution: it exposes gate activations for gradient attribution.

Status: not yet implemented (scheduled for Phase 3).
"""

from __future__ import annotations

_PHASE = "Phase 3"


def _not_yet(name: str) -> None:
    raise NotImplementedError(
        f"{name} is scheduled for Phase 3. BOLT never returns placeholder data: "
        "see BOLT_SPEC.md Rule 12.2 (never fabricate a result) and Rule 12.3 (fail loudly)."
    )
