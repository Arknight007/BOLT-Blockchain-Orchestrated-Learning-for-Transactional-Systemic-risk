"""Disk cache for raw API responses (BOLT_SPEC.md Section 3).

Cache key is ``(source, asset, start, end, frequency)``. Phase 1 acceptance
requires that a second run of ``bolt ingest`` makes zero API calls, so every
fetch goes through :func:`cached_fetch`.

Two properties matter beyond speed:

* **Reproducibility.** The frozen dataset must be rebuildable from the cache
  alone, without the network and without depending on an API still being up.
* **Provenance.** Each entry stores when it was fetched and from what URL, so
  ``DATA_CARD.md`` can state where every column came from.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from bolt.logging_setup import get_logger

log = get_logger(__name__)

_META_SUFFIX = ".meta.json"


class IngestError(RuntimeError):
    """A source failed. Rule 12.3: raise, never return a column of zeros."""


@dataclass(frozen=True, slots=True)
class CacheKey:
    """Identifies one raw payload."""

    source: str
    asset: str
    start: str
    end: str
    frequency: str

    def digest(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def filename(self) -> str:
        safe_asset = self.asset.replace("/", "-")
        return f"{self.source}__{safe_asset}__{self.frequency}__{self.digest()}.json"


@dataclass(frozen=True, slots=True)
class CacheEntry:
    payload: Any
    fetched_at: str
    from_cache: bool
    path: Path


class DiskCache:
    """JSON-on-disk cache rooted at ``data/raw``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def path_for(self, key: CacheKey) -> Path:
        return self.root / key.filename()

    def read(self, key: CacheKey) -> CacheEntry | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("cache entry %s is unreadable (%s); refetching", path.name, exc)
            return None
        self.hits += 1
        return CacheEntry(
            payload=document["payload"],
            fetched_at=document.get("fetched_at", "unknown"),
            from_cache=True,
            path=path,
        )

    def write(self, key: CacheKey, payload: Any, *, url: str | None = None) -> CacheEntry:
        path = self.path_for(key)
        fetched_at = datetime.now(timezone.utc).isoformat()
        document = {
            "key": asdict(key),
            "fetched_at": fetched_at,
            "url": url,
            "payload": payload,
        }
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, separators=(",", ":"))
        tmp.replace(path)
        self.misses += 1
        return CacheEntry(payload=payload, fetched_at=fetched_at, from_cache=False, path=path)

    def summary(self) -> str:
        total = self.hits + self.misses
        return f"cache: {self.hits} hit(s), {self.misses} fetch(es) of {total}"


def cached_fetch(
    cache: DiskCache,
    key: CacheKey,
    fetcher: Callable[[], Any],
    *,
    refresh: bool = False,
    url: str | None = None,
) -> CacheEntry:
    """Return the cached payload, or call ``fetcher`` once and cache the result."""
    if not refresh:
        entry = cache.read(key)
        if entry is not None:
            log.debug("cache hit  %s %s", key.source, key.asset)
            return entry
    log.info("fetching   %s %s (%s..%s)", key.source, key.asset, key.start, key.end)
    payload = fetcher()
    return cache.write(key, payload, url=url)


def backoff_sleep(attempt: int, base: float, maximum: float) -> float:
    """Exponential backoff delay for ``attempt`` (0-indexed). Returns seconds slept."""
    delay = min(base * (2 ** attempt), maximum)
    time.sleep(delay)
    return delay
