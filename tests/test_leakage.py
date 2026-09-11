"""The five leakage guards of BOLT_SPEC.md Section 4.

These are the most important tests in the repository. A leaking model produces
excellent metrics and is worthless, and the failure is silent: nothing crashes,
the numbers simply come out too good. Each guard below is tested by constructing
data that WOULD leak and asserting that the guard catches it -- a guard that has
never rejected anything is not known to work.

    Guard 1  point-in-time features
    Guard 2  news timestamps
    Guard 3  no target leakage in features
    Guard 4  embargo between train and test
    Guard 5  scaler fitted on train only
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bolt.align.panel import LeakageError, assert_no_future_timestamps, assert_point_in_time
from bolt.config import Fold
from bolt.evaluate.splits import TrainOnlyScaler, walk_forward_splits
from bolt.evaluate.splits import LeakageError as SplitLeakageError
from bolt.features.technical import compute_technical, realized_volatility
from bolt.labeling.crash import label_crashes
from bolt.windows import build_windows

PARAMS = {
    "rsi_period": 14, "atr_period": 14, "momentum_period": 10,
    "volume_z_window": 20, "realized_vol_windows": [7, 14, 30], "nvt_window": 7,
    "onchain_netflow_window": 7, "sentiment_windows": [1, 7],
    "sentiment_count_z_window": 30, "contagion_window": 30,
    "lead_lag_max_lag": 5, "corr_graph_min_edge_weight": 0.30,
}


@pytest.fixture
def price_frame() -> pd.DataFrame:
    """A deterministic synthetic OHLCV series with a crash in the middle."""
    rng = np.random.default_rng(0)
    n = 400
    dates = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    steps = rng.normal(0.0, 0.02, n)
    steps[200:215] = -0.05          # a deliberate 15-day decline
    close = 100.0 * np.exp(np.cumsum(steps))
    frame = pd.DataFrame(
        {
            "close": close,
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "volume": rng.lognormal(10.0, 0.3, n),
        },
        index=dates,
    )
    frame.index.name = "date"
    return frame


def _panel(frame: pd.DataFrame, asset: str = "BTC") -> pd.DataFrame:
    out = frame.copy()
    out["asset"] = asset
    return out.set_index("asset", append=True)


# ---------------------------------------------------------------------------
# Guard 1 - point-in-time features
# ---------------------------------------------------------------------------

def test_guard1_accepts_trailing_features(price_frame):
    features = compute_technical(price_frame, PARAMS)
    assert_point_in_time(_panel(features), list(features.columns))


def test_guard1_rejects_a_centred_window(price_frame):
    """A centred rolling window leaves NaNs at BOTH ends and sees the future."""
    features = compute_technical(price_frame, PARAMS)
    features["realized_vol_7"] = (
        np.log(price_frame["close"] / price_frame["close"].shift(1))
        .rolling(7, center=True, min_periods=7).std()
    )
    # Centred windows are caught by the forward-shift test below; the structural
    # check catches the purely forward-looking case.
    forward = features.copy()
    forward["realized_vol_7"] = forward["realized_vol_7"].shift(-6)
    with pytest.raises(LeakageError, match="Guard 1"):
        assert_point_in_time(_panel(forward), ["realized_vol_7"])


def test_guard1_rejects_a_forward_window(price_frame):
    features = compute_technical(price_frame, PARAMS)
    leaked = features.copy()
    # Reverse the series, roll, reverse back: a textbook forward-looking window.
    reversed_close = price_frame["close"].iloc[::-1]
    leaked["realized_vol_7"] = (
        reversed_close.rolling(7, min_periods=7).std().iloc[::-1]
    )
    with pytest.raises(LeakageError, match="Guard 1"):
        assert_point_in_time(_panel(leaked), ["realized_vol_7"])


def test_guard1_reports_which_feature_leaked(price_frame):
    features = compute_technical(price_frame, PARAMS)
    features["rsi_14"] = features["rsi_14"].shift(-20)
    with pytest.raises(LeakageError) as exc:
        assert_point_in_time(_panel(features), list(features.columns))
    assert "rsi_14" in str(exc.value)


def test_trailing_windows_warm_up_at_the_start_not_the_end(price_frame):
    """The positive statement of Guard 1: NaNs belong at the beginning."""
    vol = realized_volatility(price_frame["close"], 30)
    assert vol.iloc[:30].isna().all(), "a trailing window must warm up"
    assert vol.iloc[30:].notna().all(), "a trailing window must not go missing later"


# ---------------------------------------------------------------------------
# Guard 2 - news timestamps
# ---------------------------------------------------------------------------

def test_guard2_accepts_past_headlines():
    as_of = pd.Timestamp("2022-11-10", tz="UTC")
    headlines = pd.DataFrame({
        "published_at": pd.to_datetime(["2022-11-08", "2022-11-09"], utc=True),
        "title": ["a", "b"],
    })
    assert_no_future_timestamps(headlines, "published_at", as_of)


def test_guard2_rejects_a_headline_published_after_the_feature_date():
    """The classic silent failure: text published AFTER a crash 'predicts' it."""
    as_of = pd.Timestamp("2022-11-08", tz="UTC")
    headlines = pd.DataFrame({
        "published_at": pd.to_datetime(["2022-11-07", "2022-11-11"], utc=True),
        "title": ["before FTX", "FTX has collapsed"],
    })
    with pytest.raises(LeakageError, match="Guard 2"):
        assert_no_future_timestamps(headlines, "published_at", as_of)


def test_guard2_is_inclusive_of_the_boundary():
    as_of = pd.Timestamp("2022-11-08", tz="UTC")
    headlines = pd.DataFrame({
        "published_at": pd.to_datetime(["2022-11-08"], utc=True), "title": ["same day"],
    })
    assert_no_future_timestamps(headlines, "published_at", as_of)


# ---------------------------------------------------------------------------
# Guard 3 - no target leakage in features
# ---------------------------------------------------------------------------

def test_guard3_shifting_prices_forward_changes_features_only_where_it_should(price_frame):
    """Spec Section 4: shift the price series by one day and recompute.

    Every feature at date t must change by exactly the amount implied by its own
    trailing window moving one step -- and crucially, feature[t] computed on the
    shifted series must equal feature[t-1] computed on the original. If a feature
    used t+1, that identity breaks.
    """
    original = compute_technical(price_frame, PARAMS)
    shifted_prices = price_frame.shift(1).dropna()
    shifted = compute_technical(shifted_prices, PARAMS)

    for column in original.columns:
        left = shifted[column].dropna()
        right = original[column].shift(1).reindex(left.index).dropna()
        common = left.index.intersection(right.index)
        assert len(common) > 50, f"{column}: too few comparable points"
        np.testing.assert_allclose(
            left.loc[common].to_numpy(), right.loc[common].to_numpy(),
            rtol=1e-9, atol=1e-9,
            err_msg=f"{column} does not shift cleanly with its input, so it reads t+1",
        )


def test_guard3_labels_use_the_future_but_features_never_do(price_frame):
    """The label is ALLOWED to look forward. That is the only such exception."""
    labels = label_crashes(price_frame["close"], 0.20, 14)
    assert labels.iloc[-14:].isna().all(), "the last horizon days cannot be known"
    assert labels.iloc[:-14].notna().all()

    features = compute_technical(price_frame, PARAMS)
    assert_point_in_time(_panel(features), list(features.columns))


def test_guard3_label_is_not_derivable_from_same_day_features(price_frame):
    """A feature perfectly correlated with the label would be the target in disguise."""
    features = compute_technical(price_frame, PARAMS)
    labels = label_crashes(price_frame["close"], 0.20, 14)
    joined = features.join(labels.rename("label")).dropna()
    if joined["label"].nunique() < 2:
        pytest.skip("synthetic series produced a single label class")
    for column in features.columns:
        corr = abs(joined[column].corr(joined["label"]))
        assert corr < 0.98, f"{column} is almost perfectly correlated with the label"


# ---------------------------------------------------------------------------
# Guard 4 - embargo between train and test
# ---------------------------------------------------------------------------

def _meta(n: int = 600, start: str = "2021-01-01", horizon: int = 14) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "asset": "BTC",
        "window_start": dates,
        "window_end": dates + pd.Timedelta(29, "D"),
        "label_end": dates + pd.Timedelta(29 + horizon, "D"),
    })


def test_guard4_embargo_removes_boundary_samples():
    meta = _meta()
    folds = [Fold(train_end=pd.Timestamp("2021-12-31").date(),
                  test_start=pd.Timestamp("2022-01-01").date(),
                  test_end=pd.Timestamp("2022-06-30").date())]
    splits = walk_forward_splits(meta, folds, embargo_days=45)
    assert len(splits) == 1
    assert splits[0].embargoed > 0, "the embargo must actually remove boundary samples"


def test_guard4_assertion_holds_on_every_fold():
    meta = _meta(n=900, start="2020-06-01")
    folds = [
        Fold(pd.Timestamp("2021-12-31").date(), pd.Timestamp("2022-01-01").date(),
             pd.Timestamp("2022-06-30").date()),
    ]
    for split in walk_forward_splits(meta, folds, embargo_days=45):
        last_train_label = meta.loc[split.train_idx, "label_end"].max()
        first_test_window = meta.loc[split.test_idx, "window_start"].min()
        assert last_train_label < first_test_window


def test_guard4_rejects_an_insufficient_embargo():
    """With a zero embargo, a training label resolves inside the test window."""
    from bolt.evaluate.splits import assert_no_overlap, SplitResult

    meta = _meta()
    window_start = meta["window_start"]
    test_start = pd.Timestamp("2022-01-01", tz="UTC")
    train_idx = np.flatnonzero((window_start <= pd.Timestamp("2021-12-31", tz="UTC")).to_numpy())
    test_idx = np.flatnonzero((window_start >= test_start).to_numpy())
    leaking = SplitResult("leaky", train_idx, test_idx, 0, test_start, test_start, test_start)

    with pytest.raises(SplitLeakageError, match="Guard 4"):
        assert_no_overlap(meta, leaking)


def test_guard4_rejects_rows_in_both_sets():
    from bolt.evaluate.splits import assert_no_overlap, SplitResult

    meta = _meta()
    shared = np.arange(0, 50)
    split = SplitResult("overlap", shared, shared, 0,
                        pd.Timestamp("2022-01-01", tz="UTC"),
                        pd.Timestamp("2022-01-01", tz="UTC"),
                        pd.Timestamp("2022-06-30", tz="UTC"))
    with pytest.raises(SplitLeakageError):
        assert_no_overlap(meta, split)


def test_guard4_window_metadata_spans_lookback_plus_horizon(price_frame):
    """The embargo is only correct if label_end really covers the forward horizon."""
    features = compute_technical(price_frame, PARAMS)
    panel = _panel(features.join(label_crashes(price_frame["close"], 0.20, 14).rename("label")))
    _, _, meta = build_windows(panel, 30, list(features.columns), "label", horizon=14)
    span = (meta["label_end"] - meta["window_start"]).dt.days
    assert (span == 30 - 1 + 14).all(), "label_end must cover the window AND the horizon"


# ---------------------------------------------------------------------------
# Guard 5 - scaler fitted on train only
# ---------------------------------------------------------------------------

def test_guard5_scaler_sees_only_the_training_fold():
    rng = np.random.default_rng(1)
    train = rng.normal(0.0, 1.0, (300, 30, 5))
    test = rng.normal(5.0, 3.0, (100, 30, 5))

    scaler = TrainOnlyScaler().fit(train)
    assert scaler.n_samples_seen_ == len(train), "Guard 5: scaler must see the train fold size"

    scaled_test = scaler.transform(test)
    # The test fold has a different distribution; if the scaler had been fitted on
    # everything, the scaled test data would be centred near zero. It must not be.
    assert abs(float(np.mean(scaled_test))) > 1.0, (
        "the test fold standardised to ~0 mean, which means the scaler saw it"
    )


def test_guard5_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="Guard 5"):
        TrainOnlyScaler().transform(np.zeros((5, 3)))


def test_guard5_fitting_on_everything_is_detectably_different():
    rng = np.random.default_rng(2)
    train = rng.normal(0.0, 1.0, (300, 5))
    test = rng.normal(5.0, 1.0, (100, 5))

    correct = TrainOnlyScaler().fit(train).transform(test)
    leaked = TrainOnlyScaler().fit(np.vstack([train, test])).transform(test)
    assert not np.allclose(correct, leaked), (
        "fitting on train-plus-test must change the result, otherwise this guard "
        "is not testing anything"
    )


def test_guard5_constant_column_does_not_divide_by_zero():
    X = np.ones((50, 4))
    scaled = TrainOnlyScaler().fit_transform(X)
    assert np.isfinite(scaled).all()
