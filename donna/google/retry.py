"""Retry and rate limiting for Google API calls.

Gmail bills requests in "quota units" (messages.get costs 5) against a per-second and a
per-minute per-user budget. Fetching a year of mail in a tight loop exceeds it — observed
directly: a 154-message backfill died partway through with

    HttpError 403 ... Quota exceeded for quota metric 'Total Query Cost'
                      and limit 'Units per minute per user'

Rate limiting is not an edge case here, it is the normal operating condition of a backfill,
so it belongs in the transport layer rather than being sprinkled through callers.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Google signals "slow down" through several codes; 403 needs its reason inspected, since
# a 403 for insufficient permissions must fail immediately rather than be retried.
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_RETRY_REASONS = {
    "rateLimitExceeded",
    "userRateLimitExceeded",
    "backendError",
    "internalError",
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
}


def _reasons(exc: HttpError) -> set[str]:
    """Machine-readable reason codes from the error body.

    Parsed from the JSON rather than matched against str(exc): googleapiclient only folds
    the reason into its string form when the body also carries a `message`, so string
    matching silently misses real rate-limit errors.
    """
    try:
        payload = json.loads(
            exc.content.decode("utf-8") if isinstance(exc.content, bytes) else exc.content
        )
    except (ValueError, AttributeError, UnicodeDecodeError):
        return set()

    error = payload.get("error")
    if not isinstance(error, dict):
        return set()

    found = {item.get("reason", "") for item in error.get("errors") or [] if isinstance(item, dict)}
    if isinstance(error.get("status"), str):
        found.add(error["status"])
    return {r for r in found if r}


def is_transient(exc: Exception) -> bool:
    if not isinstance(exc, HttpError):
        return False
    status = getattr(exc.resp, "status", None)
    if status in _RETRY_STATUSES:
        return True
    if status == 403:
        # A 403 is ambiguous: "slow down" must be retried, "you lack permission" must fail
        # fast rather than hide a clear error behind a minute of backoff.
        reasons = _reasons(exc)
        if reasons:
            return bool(reasons & _RETRY_REASONS)
        return any(reason in str(exc) for reason in _RETRY_REASONS)
    return False


def with_backoff(
    call: Callable[[], T],
    *,
    attempts: int = 6,
    base_delay: float = 2.0,
    max_delay: float = 64.0,
    label: str = "google call",
) -> T:
    """Run `call`, retrying transient failures with exponential backoff and jitter.

    Jitter matters even for a single user: without it, several jobs that hit the limit
    together would retry in lockstep and hit it again.
    """
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            if not is_transient(exc) or attempt == attempts:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, delay * 0.25)
            logger.warning(
                "%s rate-limited (attempt %d/%d); retrying in %.1fs",
                label,
                attempt,
                attempts,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError("unreachable")  # pragma: no cover


class RateLimiter:
    """Minimum spacing between calls, to stay under the per-second budget by construction
    rather than by discovering the limit and backing off."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


def execute(request: Any, *, label: str = "google call") -> Any:
    """googleapiclient request + backoff, the common case."""
    return with_backoff(request.execute, label=label)
