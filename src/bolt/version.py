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


@lru_cache(maxsize=8)
def git_commit_sha(repo_root: str | Path = ".", *, short: bool = False) -> str:
    """Return the current commit SHA, suffixed ``-dirty`` if the tree has changes.

    Returns :data:`UNKNOWN` when git is unavailable or the directory is not a
    repository. Callers that record provenance must treat ``unknown`` as a
    failure, not a default.
    """
    root = Path(repo_root)
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD"]
        if short:
            args = ["git", "rev-parse", "--short", "HEAD"]
        sha = subprocess.run(
            args, cwd=root, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root, capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    if not sha:
        return UNKNOWN
    return sha + (DIRTY_SUFFIX if status else "")
