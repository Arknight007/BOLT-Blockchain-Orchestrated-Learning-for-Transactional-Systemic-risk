"""Determinism (BOLT_SPEC.md Rule 12.4).

Same config plus same seed must produce identical predictions across two runs.

This matters more here than in an ordinary project. G5 commits a digest of a
prediction to a public ledger, and a verifier re-running the pipeline must be
able to reproduce that prediction exactly. If the pipeline is non-deterministic,
the commitment proves only that *some* prediction was made, not *this* one - and
the whole auditability claim weakens to nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bolt.evaluate.splits import TrainOnlyScaler, walk_forward_splits
from bolt.models.classical import LogRegModel, RandomForestModel
from bolt.models.gbm import XGBModel
from bolt.models.gru_numpy import NumpyGRU
from bolt.models.lstm_numpy import NumpyLSTM
from bolt.models.rule import VolatilityRule


@pytest.fixture(scope="module")
def dataset():
    """A small deterministic dataset with a learnable signal."""
    rng = np.random.default_rng(20260911)
    n, T, F = 300, 12, 6
    y = (rng.random(n) < 0.2).astype(np.int8)
    X = rng.normal(0.0, 1.0, (n, T, F))
    X[y == 1, -4:, 0] += 1.4        # signal in the recent part of the window
    return X, y


MODELS = [
    ("lstm", lambda: NumpyLSTM(hidden=6, epochs=8, seed=42, dropout=0.1)),
    ("gru", lambda: NumpyGRU(hidden=6, epochs=8, seed=42, dropout=0.1)),
    ("xgb", lambda: XGBModel(n_estimators=40, max_depth=3, seed=42)),
    ("rf", lambda: RandomForestModel(n_estimators=40, max_depth=5, seed=42)),
    ("logreg", lambda: LogRegModel(seed=42)),
    ("rule", lambda: VolatilityRule()),
]


@pytest.mark.parametrize("name, factory", MODELS, ids=[m[0] for m in MODELS])
def test_two_runs_produce_identical_predictions(dataset, name, factory):
    """Rule 12.4: identical config and seed, bit-identical output."""
    X, y = dataset

    first = factory()
    first.fit(X, y)
    a = first.predict_proba(X)

    second = factory()
    second.fit(X, y)
    b = second.predict_proba(X)

    np.testing.assert_array_equal(
        a, b, err_msg=f"{name} is not reproducible under a fixed seed"
    )


@pytest.mark.parametrize("name, factory", MODELS[:2], ids=["lstm", "gru"])
def test_different_seeds_give_different_models(dataset, name, factory):
    """The seed must actually do something, or the test above proves nothing."""
    X, y = dataset
    cls = type(factory())

    a = cls(hidden=6, epochs=8, seed=1, dropout=0.1)
    b = cls(hidden=6, epochs=8, seed=2, dropout=0.1)
    a.fit(X, y)
    b.fit(X, y)
    assert not np.array_equal(a.predict_proba(X), b.predict_proba(X))


def test_scaler_is_deterministic(dataset):
    X, _ = dataset
    a = TrainOnlyScaler().fit(X).transform(X)
    b = TrainOnlyScaler().fit(X).transform(X)
    np.testing.assert_array_equal(a, b)


def test_splits_are_deterministic():
    from bolt.config import Fold

    dates = pd.date_range("2021-01-01", periods=600, freq="D", tz="UTC")
    meta = pd.DataFrame({
        "asset": "BTC",
        "window_start": dates,
        "window_end": dates + pd.Timedelta(29, "D"),
        "label_end": dates + pd.Timedelta(43, "D"),
    })
    folds = [Fold(pd.Timestamp("2021-12-31").date(),
                  pd.Timestamp("2022-01-01").date(),
                  pd.Timestamp("2022-06-30").date())]

    a = walk_forward_splits(meta, folds, 45)
    b = walk_forward_splits(meta, folds, 45)
    assert len(a) == len(b)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left.train_idx, right.train_idx)
        np.testing.assert_array_equal(left.test_idx, right.test_idx)


def test_window_construction_is_deterministic():
    from bolt.windows import build_windows

    rng = np.random.default_rng(7)
    dates = pd.date_range("2021-01-01", periods=200, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {"f1": rng.normal(size=200), "f2": rng.normal(size=200),
         "label": (rng.random(200) < 0.2).astype(float)},
        index=dates,
    )
    frame["asset"] = "BTC"
    panel = frame.set_index("asset", append=True)
    panel.index.names = ["date", "asset"]

    X1, y1, m1 = build_windows(panel, 20, ["f1", "f2"], "label", 14)
    X2, y2, m2 = build_windows(panel, 20, ["f1", "f2"], "label", 14)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    pd.testing.assert_frame_equal(m1, m2)


def test_the_committed_digest_is_reproducible(dataset):
    """The property G5 depends on: the same prediction hashes the same way twice."""
    from bolt.chain.payload import canonical_payload, digest_hex

    X, y = dataset
    scores = []
    for _ in range(2):
        model = NumpyLSTM(hidden=6, epochs=8, seed=42, dropout=0.1)
        model.fit(X, y)
        scores.append(float(model.predict_proba(X[:1])[0]))

    assert scores[0] == scores[1], "a non-reproducible score cannot be committed"

    payloads = [
        canonical_payload({
            "prediction_id": "0x1", "asset": "BTC", "as_of_date": "2025-11-01",
            "horizon_days": 14, "risk_score": score * 100, "probability": score,
            "severity_band": "HIGH", "top_drivers": [], "model_name": "lstm",
            "model_version": "abc1234", "feature_hash": "0xfeed",
            "code_version": "abc1234",
        })
        for score in scores
    ]
    assert digest_hex(payloads[0]) == digest_hex(payloads[1])


def test_content_digest_is_stable_across_container_rewrites():
    """The reproducibility contract must survive a rebuild.

    Parquet embeds writer metadata, so writing identical data twice yields
    different file bytes - measured on the real panel: f8532919... and
    925ac6c8... for content assert_frame_equal confirmed identical. Hashing the
    FILE therefore reports a mismatch to anyone who rebuilds, which is exactly
    backwards for a claim that says "rebuild and check the hash".
    """
    import tempfile
    from pathlib import Path

    from bolt.align.build import content_digest

    rng = np.random.default_rng(3)
    dates = pd.date_range("2021-01-01", periods=200, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {"close": rng.normal(100, 5, 200), "vol": rng.normal(0.3, 0.05, 200)},
        index=dates,
    )
    frame["asset"] = "BTC"
    panel = frame.set_index("asset", append=True)
    panel.index.names = ["date", "asset"]

    with tempfile.TemporaryDirectory() as tmp:
        first = Path(tmp) / "a.parquet"
        second = Path(tmp) / "b.parquet"
        panel.to_parquet(first, engine="pyarrow", compression="snappy")
        panel.to_parquet(second, engine="pyarrow", compression="snappy")

        reloaded_a = pd.read_parquet(first)
        reloaded_b = pd.read_parquet(second)

        assert content_digest(reloaded_a) == content_digest(panel), (
            "a parquet round trip must not change the content digest"
        )
        assert content_digest(reloaded_a) == content_digest(reloaded_b)


def test_content_digest_detects_a_real_change():
    """Stability is only useful if the digest still catches actual edits."""
    from bolt.align.build import content_digest

    dates = pd.date_range("2021-01-01", periods=50, freq="D", tz="UTC")
    frame = pd.DataFrame({"close": np.linspace(100, 120, 50)}, index=dates)
    frame["asset"] = "BTC"
    panel = frame.set_index("asset", append=True)
    panel.index.names = ["date", "asset"]

    baseline = content_digest(panel)
    altered = panel.copy()
    altered.iloc[0, 0] += 0.001
    assert content_digest(altered) != baseline


def test_content_digest_ignores_column_and_row_order():
    from bolt.align.build import content_digest

    dates = pd.date_range("2021-01-01", periods=30, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {"b": np.arange(30.0), "a": np.arange(30.0) * 2}, index=dates
    )
    frame["asset"] = "BTC"
    panel = frame.set_index("asset", append=True)
    panel.index.names = ["date", "asset"]

    shuffled = panel[["b", "a"]].sample(frac=1.0, random_state=0)
    assert content_digest(shuffled) == content_digest(panel)
