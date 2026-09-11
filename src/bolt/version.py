"""Code provenance (BOLT_SPEC.md Rule 12.5).

``model_version`` on every on-chain commitment is the git commit SHA of the code
that produced the prediction. A committed prediction that cannot be traced back
to exact code proves nothing, so this is resolved at commit time, never guessed.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

__version__ = "0.1.0"

UNKNOWN = "unknown"
DIRTY_SUFFIX = "-dirty"

#: Paths whose state determines whether the CODE is recoverable from the repo.
#:
#: Deliberately excludes data/ and outputs/. `bolt build` writes the panel, the
#: data card, the tables and its own log, so judging dirtiness over the whole
#: tree meant the build dirtied itself and every data card was stamped "-dirty"
#: no matter how clean the commit - including on an untracked stray log file.
#: The claim a data card makes is "this code produced this data", so the check
#: is over the code.
CODE_PATHS = (
    "src", "config", "contracts", "scripts", "tests",
    "pyproject.toml", "requirements.txt", "BOLT_SPEC.md",
)


@lru_cache(maxsize=8)
def git_commit_sha(repo_root: str | Path = ".", *, short: bool = False) -> str:
    """Return the current commit SHA, suffixed ``-dirty`` if the CODE has changes.

    "Dirty" is judged over :data:`CODE_PATHS` only. A generated dataset or a
    stray log in ``outputs/`` does not make the code unrecoverable, and counting
    them made every data card report ``-dirty`` because the build modifies its
    own outputs before the card is written.

    Returns :data:`UNKNOWN` when git is unavailable or the directory is not a
    repository. Callers that record provenance must treat ``unknown`` as a
    failure, not a default.
    """
    root = Path(repo_root)
    try:
        args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        sha = subprocess.run(
            args, cwd=root, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", *CODE_PATHS],
            cwd=root, capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    if not sha:
        return UNKNOWN
    return sha + (DIRTY_SUFFIX if status else "")
