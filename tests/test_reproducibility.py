"""Determinism (BOLT_SPEC.md Rule 12.4).

Status: implemented in Phase 3.

Same config plus same seed must produce byte-identical predictions across two runs.

This module is a placeholder so the test map matches the build order. It SKIPS
rather than passes: a green tick here before the code exists would be a lie.
"""

import pytest

pytest.skip(
    "scheduled for Phase 3 - see BOLT_SPEC.md Section 11",
    allow_module_level=True,
)
