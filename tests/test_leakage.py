"""The five leakage guards of BOLT_SPEC.md Section 4.

Status: implemented in Phase 2.

Guard 1 point-in-time features; Guard 2 news timestamps; Guard 3 no target leakage; Guard 4 train/test embargo; Guard 5 scaler fitted on train only

This module is a placeholder so the test map matches the build order. It SKIPS
rather than passes: a green tick here before the code exists would be a lie.
"""

import pytest

pytest.skip(
    "scheduled for Phase 2 - see BOLT_SPEC.md Section 11",
    allow_module_level=True,
)
