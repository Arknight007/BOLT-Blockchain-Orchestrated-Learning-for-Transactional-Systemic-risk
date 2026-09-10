"""web3 submit and read against PredictionRegistry - G5 (BOLT_SPEC.md Section 9).

The private key is read from an environment variable only. Testnet only.

Status: not yet implemented (scheduled for Phase 5).
"""

from __future__ import annotations

_PHASE = "Phase 5"


def _not_yet(name: str) -> None:
    raise NotImplementedError(
        f"{name} is scheduled for Phase 5. BOLT never returns placeholder data: "
        "see BOLT_SPEC.md Rule 12.2 (never fabricate a result) and Rule 12.3 (fail loudly)."
    )
