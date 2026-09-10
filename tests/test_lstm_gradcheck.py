"""Numerical gradient check for the from-scratch LSTM and GRU (Section 6).

Status: implemented in Phase 3.

Analytic gradients must match numerical ones to a relative error below 1e-5. A silently wrong backward pass invalidates every downstream result.

This module is a placeholder so the test map matches the build order. It SKIPS
rather than passes: a green tick here before the code exists would be a lie.
"""

import pytest

pytest.skip(
    "scheduled for Phase 3 - see BOLT_SPEC.md Section 11",
    allow_module_level=True,
)
