"""CoinGecko and Binance market data (BOLT_SPEC.md Section 3).

Binance is the primary source for candles - it is the venue where the trades
actually happened. CoinGecko supplies market capitalisation and dominance.

Status: not yet implemented (scheduled for Phase 1).
"""

from __future__ import annotations

_PHASE = "Phase 1"


def _not_yet(name: str) -> None:
    raise NotImplementedError(
        f"{name} is scheduled for Phase 1. BOLT never returns placeholder data: "
        "see BOLT_SPEC.md Rule 12.2 (never fabricate a result) and Rule 12.3 (fail loudly)."
    )
