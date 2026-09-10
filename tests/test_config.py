"""Configuration contract (BOLT_SPEC.md Sections 2 and 4).

The invariants asserted here are not style checks. Each one, if violated, produces
a run that looks fine and is wrong: a short embargo leaks the test period into
training, a reinstated accuracy metric produces a 98% headline on a 2% base rate.
"""

from __future__ import annotations

from datetime import date

import pytest

from bolt.config import ConfigError, load_config


def test_real_config_loads(config_path):
    cfg = load_config(config_path)
    assert cfg.seed == 42
    assert cfg.start_date == date(2020, 1, 1)
    assert cfg.primary_asset == "BTC"
    assert len(cfg.folds) == 4
    assert len(cfg.episodes) == 4


def test_feature_columns_cover_all_four_families(config_path):
    cfg = load_config(config_path)
    cols = cfg.feature_columns
    assert len(cols) == len(set(cols)), "duplicate feature name across families"
    for expected in ("realized_vol_7", "exchange_netflow_7", "headline_polarity_7d",
                     "mean_pairwise_corr_30"):
        assert expected in cols


def test_universe_splits_into_targets_and_context(config_path):
    cfg = load_config(config_path)
    symbols = {a.symbol for a in cfg.assets}
    assert {"BTC", "ETH", "BNB", "XRP", "LINK", "UNI", "HYPE", "ASTER", "USDT", "DAI"} == symbols
    # Stablecoins are never labelled: a 20% drawdown in a dollar peg is a de-peg,
    # and labelling them would silently dilute the positive-class rate.
    context = {a.symbol for a in cfg.assets if not a.is_target}
    assert context == {"USDT", "DAI"}


def test_every_asset_has_an_inception_date(config_path):
    cfg = load_config(config_path)
    for asset in cfg.assets:
        assert isinstance(asset.inception, date)


# --- Guard 4: the embargo invariant ---------------------------------------

def test_embargo_at_least_lookback_plus_horizon(config_path):
    cfg = load_config(config_path)
    assert cfg.embargo_days >= cfg.lookback_days + cfg.horizon_days


def test_short_embargo_is_rejected(raw_config, write_config):
    raw_config["split"]["embargo_days"] = 10
    with pytest.raises(ConfigError, match="Guard 4"):
        load_config(write_config(raw_config))


# --- Rule 12.1: the accuracy ban cannot be switched off -------------------

def test_disabling_the_accuracy_ban_is_rejected(raw_config, write_config):
    raw_config["evaluation"]["forbid_accuracy"] = False
    with pytest.raises(ConfigError, match="forbid_accuracy"):
        load_config(write_config(raw_config))


def test_accuracy_in_report_list_is_rejected(raw_config, write_config):
    raw_config["evaluation"]["report"].append("accuracy")
    with pytest.raises(ConfigError, match="banned metric"):
        load_config(write_config(raw_config))


# --- structural validation -------------------------------------------------

@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c["data"].update(end_date="2019-01-01"), "must be after"),
        (lambda c: c["labeling"].update(drawdown_threshold=1.5), "fraction in"),
        (lambda c: c["labeling"].update(horizon_days=0), "horizon_days"),
        (lambda c: c["assets"].update(primary="DOGE"), "not in the frozen universe"),
        (lambda c: c["assets"].update(primary="USDT"), "never labelled"),
        (lambda c: c["imbalance"].update(method="magic"), "unknown method"),
        (lambda c: c["explain"]["analogue"].update(exclude_days=5), "exclude_days"),
        (lambda c: c.pop("chain"), "missing required section"),
    ],
)
def test_invalid_configs_are_rejected(raw_config, write_config, mutate, message):
    mutate(raw_config)
    with pytest.raises(ConfigError, match=message):
        load_config(write_config(raw_config))


def test_overlapping_folds_are_rejected(raw_config, write_config):
    # Stretch fold 0's test range so it swallows the start of fold 1.
    raw_config["split"]["folds"][0]["test"][1] = "2023-06-30"
    with pytest.raises(ConfigError, match="chronological"):
        load_config(write_config(raw_config))


def test_test_range_before_train_end_is_rejected(raw_config, write_config):
    raw_config["split"]["folds"][1]["test"][0] = "2022-06-01"
    with pytest.raises(ConfigError, match="not after train_end"):
        load_config(write_config(raw_config))


def test_fold_outside_data_range_is_rejected(raw_config, write_config):
    raw_config["split"]["folds"][-1]["test"][1] = "2030-12-31"
    with pytest.raises(ConfigError, match="outside"):
        load_config(write_config(raw_config))


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_config_is_immutable(config_path):
    cfg = load_config(config_path)
    with pytest.raises((AttributeError, TypeError)):
        cfg.assets = ()
