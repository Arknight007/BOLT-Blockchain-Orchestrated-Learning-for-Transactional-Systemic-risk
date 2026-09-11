"""Shared HTTP client with retry and rate limiting (BOLT_SPEC.md Section 3).

Public APIs rate-limit aggressively. This module centralises backoff so no
ingest module quietly swallows a 429 and returns a short series -- a silently
truncated history is indistinguishable from a real data gap downstream.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import requests

try:  # pragma: no cover - environment dependent
    # Use the operating system's certificate store rather than certifi's bundle.
    # On machines behind TLS-inspecting proxies or corporate AV, the intercepting
    # root lives in the OS store and certifi rejects the chain, so every ingest
    # call fails with CERTIFICATE_VERIFY_FAILED. This is a trust-source change,
    # NOT a weakening: verification stays fully enabled either way.
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # truststore is optional; certifi is the fallback
    pass

from bolt.ingest.cache import IngestError, backoff_sleep
from bolt.logging_setup import get_logger

log = get_logger(__name__)

_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class RateLimiter:
    """Simple per-process minimum-interval limiter."""

    def __init__(self, per_minute: int) -> None:
        self.min_interval = 60.0 / max(per_minute, 1)
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


def get_json(
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: int = 30,
    max_attempts: int = 5,
    backoff_base: float = 2.0,
    backoff_max: float = 60.0,
    limiter: RateLimiter | None = None,
) -> Any:
    """GET a JSON document, retrying transient failures with exponential backoff.

    Raises:
        IngestError: when every attempt fails. The pipeline must stop rather
            than continue with partial data (Rule 12.3).
    """
    last_error: str = "no attempt made"
    for attempt in range(max_attempts):
        if limiter is not None:
            limiter.wait()
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as exc:
                    raise IngestError(f"{url}: response was not JSON ({exc})") from exc
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            if response.status_code not in _RETRYABLE_STATUS:
                raise IngestError(f"{url} failed permanently -- {last_error}")
        if attempt < max_attempts - 1:
            slept = backoff_sleep(attempt, backoff_base, backoff_max)
            log.warning("retry %d/%d for %s after %.1fs -- %s",
                        attempt + 1, max_attempts - 1, url, slept, last_error)
    raise IngestError(f"{url} failed after {max_attempts} attempts -- {last_error}")
