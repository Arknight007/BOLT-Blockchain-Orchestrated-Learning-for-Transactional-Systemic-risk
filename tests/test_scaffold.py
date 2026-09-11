"""Repository scaffold and safety invariants (BOLT_SPEC.md Sections 1 and 9)."""

from __future__ import annotations

import importlib
import subprocess

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


@pytest.mark.parametrize("relative", EXPECTED_DIRS)
def test_directory_survives_a_clone(repo_root, relative):
    """Every expected directory must contain at least one tracked file.

    Git does not track empty directories, so a directory that exists only on the
    author's disk silently vanishes from a fresh clone -- taking the Section 10
    acceptance standard ("clone the repo, run one script") with it. This caught
    notebooks/ and contracts/ being absent from the published repository.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "--", relative],
        cwd=repo_root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert tracked, (
        f"{relative}/ has no tracked files and will not exist in a clone; "
        f"add a .gitkeep or a README explaining what belongs there"
    )


@pytest.mark.parametrize(
    "path, must_be_ignored",
    [
        (".env", True),
        (".env.local", True),
        ("data/raw/coingecko_BTC_2020-01-01_2025-12-31.json", True),
        ("data/interim/panel_stage1.parquet", True),
        (".venv/pyvenv.cfg", True),
        ("outputs/bolt.log", True),
        # ...but these must still reach the repository:
        (".env.example", False),
        ("data/raw/.gitkeep", False),
        ("config/default.yaml", False),
        ("outputs/tables/model_comparison.csv", False),
    ],
)
def test_gitignore_behaviour(repo_root, path, must_be_ignored):
    """Assert what git actually does, not what .gitignore happens to say.

    An earlier version of this test string-matched .gitignore lines, so a
    cosmetic pattern change (data/raw/ -> data/raw/*) broke it while the
    protection itself was intact -- and, worse, it would have passed had the
    protection been removed some other way.
    """
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", path],
        cwd=repo_root, capture_output=True, text=True,
    )
    is_ignored = result.returncode == 0
    verb = "must be ignored" if must_be_ignored else "must NOT be ignored"
    assert is_ignored == must_be_ignored, f"{path} {verb} by .gitignore"


def test_secrets_template_is_published_but_the_real_file_is_not(repo_root):
    """A clone needs .env.example to copy from; it must never receive a real .env."""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.split()
    assert ".env.example" in tracked
    assert ".env" not in tracked


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
