"""Repository scaffold and safety invariants (BOLT_SPEC.md Sections 1 and 9)."""

from __future__ import annotations

import importlib

import pytest

EXPECTED_DIRS = [
    "config", "data/raw", "data/interim", "data/processed",
    "src/bolt/ingest", "src/bolt/align", "src/bolt/features", "src/bolt/labeling",
    "src/bolt/models", "src/bolt/evaluate", "src/bolt/explain", "src/bolt/chain",
    "contracts", "notebooks", "tests",
    "outputs/figures", "outputs/tables", "outputs/predictions", "scripts",
]

EXPECTED_MODULES = [
    "bolt.config", "bolt.cli", "bolt.logging_setup", "bolt.version", "bolt.windows",
    "bolt.ingest.market", "bolt.ingest.onchain", "bolt.ingest.news", "bolt.ingest.cache",
    "bolt.align.panel",
    "bolt.features.technical", "bolt.features.onchain", "bolt.features.sentiment",
    "bolt.features.contagion",
    "bolt.labeling.crash",
    "bolt.models.base", "bolt.models.lstm_numpy", "bolt.models.gru_numpy",
    "bolt.models.gbm", "bolt.models.classical", "bolt.models.rule",
    "bolt.evaluate.splits", "bolt.evaluate.metrics", "bolt.evaluate.report",
    "bolt.explain.attribution", "bolt.explain.consistency", "bolt.explain.analogue",
    "bolt.chain.payload", "bolt.chain.client", "bolt.chain.verify",
]


@pytest.mark.parametrize("relative", EXPECTED_DIRS)
def test_directory_exists(repo_root, relative):
    assert (repo_root / relative).is_dir(), f"missing directory: {relative}"


@pytest.mark.parametrize("module", EXPECTED_MODULES)
def test_module_imports(module):
    assert importlib.import_module(module) is not None


@pytest.mark.parametrize("module", EXPECTED_MODULES)
def test_module_is_documented(module):
    doc = importlib.import_module(module).__doc__
    assert doc and "BOLT_SPEC.md" in doc, f"{module} must name the spec section it implements"


def test_gitignore_protects_secrets_and_raw_data(repo_root):
    lines = {
        line.strip()
        for line in (repo_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    for entry in (".env", "data/raw/", ".venv/"):
        assert entry in lines, f".gitignore must contain {entry}"


def test_no_private_key_is_hardcoded_anywhere(repo_root):
    """Section 9: the key is read from the environment only. Never a literal."""
    offenders = []
    for path in (repo_root / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "0x" in text and any(
            token in text for token in ("private_key =", "PRIVATE_KEY =", "privateKey")
        ):
            offenders.append(path)
    assert not offenders, f"possible hardcoded key material in {offenders}"


def test_no_framework_dependency_in_sequence_models(repo_root):
    """Section 6: the LSTM and GRU are pure NumPy. No torch, no tensorflow."""
    for name in ("lstm_numpy.py", "gru_numpy.py"):
        text = (repo_root / "src" / "bolt" / "models" / name).read_text(encoding="utf-8")
        for banned in ("import torch", "import tensorflow", "from torch", "from tensorflow"):
            assert banned not in text, f"{name} must not use a deep-learning framework"
