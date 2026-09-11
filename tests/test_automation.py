"""Prediction ledger and the automated monitoring cycle.

The ledger is what turns a pile of on-chain digests into a track record. Two
properties make it worth anything, and both are tested here:

* **Append-only.** A prediction is never edited after the fact, and a resolution
  is a separate record rather than a patch to the prediction it scores. A ledger
  that could be rewritten would be worth exactly as much as the private
  backtests G5 exists to replace.
* **Resolve before predict.** A cycle scores its own previous calls before
  making new ones, so today's health signal reflects yesterday's failures rather
  than lagging a full cycle behind them.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bolt.agents.evaluation import Outcome
from bolt.store import PredictionStore


def _payload(prediction_id: str = "0xaa", asset: str = "BTC",
             as_of: str = "2022-11-05", band: str = "HIGH") -> dict:
    return {
        "prediction_id": prediction_id, "asset": asset, "as_of_date": as_of,
        "horizon_days": 14, "risk_score": 78.0, "probability": 0.78,
        "severity_band": band, "top_drivers": [], "model_name": "ensemble",
        "model_version": "abc1234", "feature_hash": "0xfeed", "code_version": "abc1234",
    }


@pytest.fixture
def store(tmp_path) -> PredictionStore:
    return PredictionStore(tmp_path / "ledger")


# ---------------------------------------------------------------------------
# Append-only behaviour
# ---------------------------------------------------------------------------

def test_prediction_is_recorded_and_read_back(store):
    store.record_prediction(_payload(), digest="0xdead", committed=False)
    records = store.predictions()
    assert len(records) == 1
    assert records[0]["payload"]["asset"] == "BTC"
    assert records[0]["digest"] == "0xdead"


def test_the_same_prediction_id_is_never_written_twice(store):
    """The registry contract refuses overwrites; the ledger mirrors that rule."""
    store.record_prediction(_payload(), digest="0xdead", committed=False)
    store.record_prediction(_payload(), digest="0xbeef", committed=True)

    records = store.predictions()
    assert len(records) == 1, "a duplicate id must not append a second line"
    assert records[0]["digest"] == "0xdead", "the original record must survive"


def test_resolution_does_not_edit_the_prediction(store):
    """An outcome is a separate record. The prediction line is never touched."""
    store.record_prediction(_payload(), digest="0xdead", committed=False)
    before = store.paths.predictions.read_text(encoding="utf-8")

    store.record_outcome({"prediction_id": "0xaa", "classification": "TRUE_POSITIVE"})

    assert store.paths.predictions.read_text(encoding="utf-8") == before
    assert len(store.outcomes()) == 1


def test_unresolved_excludes_scored_predictions(store):
    store.record_prediction(_payload("0xaa"), digest="0x1", committed=False)
    store.record_prediction(_payload("0xbb"), digest="0x2", committed=False)
    assert len(store.unresolved()) == 2

    store.record_outcome({"prediction_id": "0xaa", "classification": "TRUE_NEGATIVE"})
    remaining = store.unresolved()
    assert len(remaining) == 1
    assert remaining[0]["prediction_id"] == "0xbb"


def test_every_line_is_valid_json(store):
    for i in range(5):
        store.record_prediction(_payload(f"0x{i:02x}"), digest=f"0x{i}", committed=False)
    for line in store.paths.predictions.read_text(encoding="utf-8").splitlines():
        json.loads(line)


def test_a_corrupt_line_is_skipped_not_fatal(store):
    """One bad line must not destroy the whole track record."""
    store.record_prediction(_payload("0xaa"), digest="0x1", committed=False)
    with store.paths.predictions.open("a", encoding="utf-8") as handle:
        handle.write("{ this is not json\n")
    store.record_prediction(_payload("0xbb"), digest="0x2", committed=False)

    assert len(store.predictions()) == 2


def test_track_record_never_reports_accuracy(store):
    store.record_prediction(_payload("0xaa"), digest="0x1", committed=False)
    store.record_outcome({"prediction_id": "0xaa", "classification": "TRUE_POSITIVE"})
    record = store.track_record()
    assert "accuracy" not in record
    assert record["counts"]["TRUE_POSITIVE"] == 1


def test_track_record_precision_and_recall(store):
    rows = [
        ("0x1", "TRUE_POSITIVE"), ("0x2", "TRUE_POSITIVE"),
        ("0x3", "FALSE_POSITIVE"), ("0x4", "FALSE_NEGATIVE"),
    ]
    for identifier, classification in rows:
        store.record_prediction(_payload(identifier), digest=identifier, committed=False)
        store.record_outcome({"prediction_id": identifier, "classification": classification})

    record = store.track_record()
    assert record["precision"] == pytest.approx(2 / 3)
    assert record["recall"] == pytest.approx(2 / 3)


def test_track_record_is_undefined_rather_than_zero_when_empty(store):
    record = store.track_record()
    assert record["precision"] is None and record["recall"] is None, (
        "no predictions means precision is undefined, not 0.0"
    )


# ---------------------------------------------------------------------------
# Outcome round-trip
# ---------------------------------------------------------------------------

def test_outcome_round_trips_through_the_ledger(store):
    outcome = Outcome(
        prediction_id="0xaa", asset="BTC", as_of_date="2022-11-05", horizon_days=14,
        predicted_band="HIGH", predicted_probability=0.78, actual_max_drawdown=0.25,
        crash_occurred=True, classification="TRUE_POSITIVE", resolved=True, detail="x",
    )
    store.record_outcome(outcome.to_dict())
    restored = Outcome.from_dict(store.outcomes()[0])
    assert restored == outcome, "the ledger added fields must not break reconstruction"


# ---------------------------------------------------------------------------
# The cycle
# ---------------------------------------------------------------------------

def test_resolve_runs_before_predict_in_the_cycle():
    """Ordering is load-bearing: a cycle must know its own past failures."""
    import inspect

    from bolt import automation

    source = inspect.getsource(automation.run_cycle)
    resolve_at = source.index("resolve_matured")
    predict_at = source.index("predict_all")
    assert resolve_at < predict_at, (
        "predicting before resolving would make the health signal lag a full cycle"
    )


def test_only_matured_predictions_are_resolved(tmp_path):
    from bolt.automation import resolve_matured
    from bolt.config import load_config

    cfg = load_config("config/default.yaml")
    store = PredictionStore(tmp_path / "ledger")
    store.record_prediction(_payload("0xaa", as_of="2022-11-05"), "0x1", False)

    # A real 30% collapse INSIDE the 14-day horizon. A linear 100 -> 70 across
    # 60 days would only be a ~7% move within the window, which the labeller
    # would rightly score as a false alarm.
    dates = pd.date_range("2022-11-01", periods=60, freq="D", tz="UTC")
    close = np.concatenate([
        np.full(4, 100.0),                      # calm up to the as-of date
        np.linspace(100.0, 68.0, 15),           # the collapse, inside the horizon
        np.full(41, 68.0),                      # flat afterwards
    ])
    prices = pd.DataFrame({"close": close}, index=dates)
    prices["asset"] = "BTC"
    panel = prices.set_index("asset", append=True)
    panel.index.names = ["date", "asset"]

    # The horizon closes on 2022-11-19; as of the 10th it is still open.
    resolved, _ = resolve_matured(cfg, store, panel, pd.Timestamp("2022-11-10", tz="UTC"))
    assert resolved == [], "a prediction whose horizon is open must not be scored"

    resolved, report = resolve_matured(cfg, store, panel, pd.Timestamp("2022-12-01", tz="UTC"))
    assert len(resolved) == 1
    assert resolved[0].classification == "TRUE_POSITIVE"
    assert report is not None


def test_cycle_summary_names_the_alerting_assets(store):
    from bolt.automation import CycleResult

    result = CycleResult(as_of="2022-11-05")
    result.predictions = [
        {"payload": _payload("0x1", "BTC", band="HIGH"), "committed": False},
        {"payload": _payload("0x2", "ETH", band="LOW"), "committed": False},
        {"payload": _payload("0x3", "XRP", band="CRITICAL"), "committed": True},
    ]
    summary = result.summary()
    assert summary["predictions"] == 3
    assert summary["alerts"] == 2
    assert set(summary["alert_assets"]) == {"BTC", "XRP"}
    assert summary["committed"] == 1
