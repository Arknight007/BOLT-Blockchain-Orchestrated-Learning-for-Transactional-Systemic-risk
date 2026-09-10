"""Logging configuration (BOLT_SPEC.md Section 11, Phase 0).

Rule 12.3 is "fail loudly". Half of failing loudly is having a log that records
what the pipeline actually did: every cache hit, every data gap, every guard that
ran. This module wires one console handler and one rotating file handler, both
driven by the ``logging`` section of the config.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Mapping

_FORMAT = "%(asctime)s %(levelname)-8s %(name)-28s %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3

_configured = False


def setup_logging(settings: Mapping[str, object] | None = None, *, force: bool = False) -> logging.Logger:
    """Configure the ``bolt`` logger tree. Idempotent unless ``force`` is set."""
    global _configured
    root = logging.getLogger("bolt")
    if _configured and not force:
        return root

    settings = settings or {}
    level = str(settings.get("level", "INFO")).upper()
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    log_file = settings.get("file")
    if log_file:
        path = Path(str(log_file))
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child of the ``bolt`` logger. Use ``get_logger(__name__)``."""
    suffix = name.removeprefix("bolt.").removeprefix("bolt")
    return logging.getLogger("bolt" + (f".{suffix}" if suffix else ""))
