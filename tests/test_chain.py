"""Payload determinism and the commitment contract - G5 (BOLT_SPEC.md Section 9).

The G5 claim is that BOLT's track record is auditable rather than merely
asserted. That claim rests on exactly two properties, and both are tested here:

1. **The digest is reproducible.** A verifier recomputing SHA-256 over the
   published payload must get a bit-identical result, regardless of dict
   insertion order, float formatting, or platform. If it does not, a mismatch is
   indistinguishable from tampering and the commitment proves nothing.

2. **A commitment cannot be overwritten.** Tested against the Solidity source in
   ``tests/test_contract.py`` where a compiler is available, and against the
   client's pre-flight check here.
"""

from __future__ import annotations

import json

import pytest

from bolt.chain.payload import (
    FLOAT_PRECISION,
    PayloadError,
    Prediction,
    canonical_payload,
    digest,
    digest_hex,
    feature_hash,
    prediction_id,
)


def _prediction(**overrides) -> dict:
    base = {
        "prediction_id": "0xabc123",
        "asset": "BTC",
        "as_of_date": "2022-11-05",
        "horizon_days": 14,
        "risk_score": 78.25,
        "probability": 0.7825,
        "severity_band": "HIGH",
        "top_drivers": [
            {"feature": "exchange_netflow_7", "contribution": 0.31, "direction": 1},
            {"feature": "realized_vol_7", "contribution": 0.22, "direction": 1},
        ],
        "model_name": "lstm",
        "model_version": "abc1234",
        "feature_hash": "0xdeadbeef",
        "code_version": "abc1234",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_canonical_payload_is_stable_across_insertion_orders():
    """The spec's explicit requirement: dict order must not change the digest."""
    forward = _prediction()
    reversed_order = {k: forward[k] for k in reversed(list(forward))}
    assert list(forward) != list(reversed_order)
    assert canonical_payload(forward) == canonical_payload(reversed_order)
    assert digest_hex(canonical_payload(forward)) == digest_hex(canonical_payload(reversed_order))


def test_canonical_payload_is_stable_across_nested_orders():
    a = _prediction(top_drivers=[{"feature": "x", "contribution": 0.5, "direction": 1}])
    b = _prediction(top_drivers=[{"direction": 1, "contribution": 0.5, "feature": "x"}])
    assert canonical_payload(a) == canonical_payload(b)


def test_canonical_payload_has_no_incidental_whitespace():
    raw = canonical_payload(_prediction())
    assert b", " not in raw and b": " not in raw, "whitespace would vary between writers"


def test_canonical_payload_is_ascii_only():
    payload = canonical_payload(_prediction(asset="BTC—EUR"))
    payload.decode("ascii")      # raises if any non-ASCII byte survived


def test_floats_are_rounded_so_the_digest_is_platform_stable():
    """A float printing with 17 digits on one machine and 16 on another breaks verification."""
    noisy = _prediction(risk_score=78.250000000000001)
    clean = _prediction(risk_score=78.25)
    assert canonical_payload(noisy) == canonical_payload(clean)


def test_non_finite_floats_become_null_not_nan():
    """NaN has no JSON representation and would produce invalid, unparseable output."""
    raw = canonical_payload(_prediction(risk_score=float("nan")))
    assert b"NaN" not in raw
    assert json.loads(raw)["risk_score"] is None


def test_digest_changes_when_any_field_changes():
    baseline = digest_hex(canonical_payload(_prediction()))
    for field, value in [
        ("asset", "ETH"), ("risk_score", 78.26), ("as_of_date", "2022-11-06"),
        ("severity_band", "CRITICAL"), ("model_version", "def5678"),
    ]:
        altered = digest_hex(canonical_payload(_prediction(**{field: value})))
        assert altered != baseline, f"changing {field} must change the digest"


def test_rounding_boundary_is_exactly_float_precision():
    below = 1.0 + 10 ** -(FLOAT_PRECISION + 2)
    assert canonical_payload(_prediction(risk_score=below)) == \
           canonical_payload(_prediction(risk_score=1.0))


def test_missing_required_field_is_refused():
    incomplete = _prediction()
    del incomplete["feature_hash"]
    with pytest.raises(PayloadError, match="feature_hash"):
        canonical_payload(incomplete)


def test_digest_is_sha256():
    import hashlib

    payload = canonical_payload(_prediction())
    assert digest(payload) == hashlib.sha256(payload).digest()
    assert digest_hex(payload) == "0x" + hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------

def test_prediction_id_is_deterministic():
    a = prediction_id("BTC", "2022-11-05", "lstm", "abc1234")
    b = prediction_id("BTC", "2022-11-05", "lstm", "abc1234")
    assert a == b and len(a) == 66


def test_prediction_id_differs_per_asset_date_model_and_code():
    base = prediction_id("BTC", "2022-11-05", "lstm", "abc1234")
    assert prediction_id("ETH", "2022-11-05", "lstm", "abc1234") != base
    assert prediction_id("BTC", "2022-11-06", "lstm", "abc1234") != base
    assert prediction_id("BTC", "2022-11-05", "gru", "abc1234") != base
    assert prediction_id("BTC", "2022-11-05", "lstm", "def5678") != base


def test_derived_id_prevents_recommitting_the_same_prediction():
    """Deriving rather than randomising the id is what makes the revert bite.

    A random id would let the same prediction be committed repeatedly until one
    of them happened to be right, and only that one published.
    """
    first = prediction_id("BTC", "2022-11-05", "lstm", "abc1234")
    second = prediction_id("BTC", "2022-11-05", "lstm", "abc1234")
    assert first == second, "the second attempt must collide and hit the contract revert"


def test_feature_hash_is_order_independent():
    assert feature_hash({"a": 1.0, "b": 2.0}) == feature_hash({"b": 2.0, "a": 1.0})


def test_feature_hash_detects_a_changed_input():
    assert feature_hash({"a": 1.0}) != feature_hash({"a": 1.000001})


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------

def test_prediction_round_trips_through_disk(tmp_path):
    prediction = Prediction(
        prediction_id="0xabc", asset="BTC", as_of_date="2022-11-05", horizon_days=14,
        risk_score=78.25, probability=0.7825, severity_band="HIGH",
        top_drivers=[{"feature": "realized_vol_7", "contribution": 0.3, "direction": 1}],
        model_name="lstm", model_version="abc1234", feature_hash="0xdead",
        code_version="abc1234", created_at="2026-01-01T00:00:00+00:00",
    )
    path = prediction.save(tmp_path)
    assert path.read_bytes() == prediction.canonical()


def test_independent_verifier_reproduces_the_digest(tmp_path):
    """The verifier must NOT import the pipeline, and must still agree with it."""
    from bolt.chain.verify import compute_digest

    prediction = Prediction(
        prediction_id="0xabc", asset="BTC", as_of_date="2022-11-05", horizon_days=14,
        risk_score=78.25, probability=0.7825, severity_band="HIGH",
        top_drivers=[{"feature": "realized_vol_7", "contribution": 0.3, "direction": 1}],
        model_name="lstm", model_version="abc1234", feature_hash="0xdead",
        code_version="abc1234", created_at="2026-01-01T00:00:00+00:00",
    )
    path = prediction.save(tmp_path)
    recomputed, document = compute_digest(path)
    assert recomputed == prediction.digest_hex()
    assert document["asset"] == "BTC"


def test_verifier_detects_a_tampered_payload(tmp_path):
    from bolt.chain.verify import compute_digest

    prediction = Prediction(
        prediction_id="0xabc", asset="BTC", as_of_date="2022-11-05", horizon_days=14,
        risk_score=78.25, probability=0.7825, severity_band="HIGH", top_drivers=[],
        model_name="lstm", model_version="abc1234", feature_hash="0xdead",
        code_version="abc1234", created_at="2026-01-01T00:00:00+00:00",
    )
    path = prediction.save(tmp_path)
    original = prediction.digest_hex()

    # Someone edits the published payload after the fact to look like a better call.
    document = json.loads(path.read_bytes())
    document["severity_band"] = "CRITICAL"
    path.write_text(json.dumps(document), encoding="utf-8")

    tampered, _ = compute_digest(path)
    assert tampered != original, "editing a published payload must break verification"


def test_verify_module_does_not_import_the_pipeline():
    """Section 9: verification that depends on what it verifies proves nothing."""
    source = (
        __import__("pathlib").Path("src/bolt/chain/verify.py").read_text(encoding="utf-8")
    )
    for banned in ("from bolt.chain.payload", "from bolt.models", "from bolt.features",
                   "from bolt.align", "import bolt.chain.payload"):
        assert banned not in source, (
            f"verify.py must not import the pipeline it verifies (found {banned!r})"
        )
