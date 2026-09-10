"""Shared pytest fixtures."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"
ASSETS_PATH = REPO_ROOT / "config" / "assets.yaml"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def config_path() -> Path:
    return CONFIG_PATH


@pytest.fixture
def raw_config() -> dict:
    """A fresh mutable copy of the real config, for building invalid variants."""
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return copy.deepcopy(yaml.safe_load(handle))


@pytest.fixture
def write_config(tmp_path: Path):
    """Write a modified config (plus the real asset universe) into a temp repo root.

    Returns a callable taking the mutated config dict and returning the path to
    the written YAML, along with a temp root whose ``config/assets.yaml`` is a
    verbatim copy of the frozen universe.
    """

    def _write(config: dict, assets: dict | None = None) -> Path:
        (tmp_path / "config").mkdir(exist_ok=True)
        cfg_file = tmp_path / "config" / "default.yaml"
        with cfg_file.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        assets_file = tmp_path / "config" / "assets.yaml"
        if assets is None:
            assets_file.write_text(ASSETS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            with assets_file.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(assets, handle, sort_keys=False)
        return cfg_file

    return _write
