"""Retry classification.

The distinction that matters: a 403 meaning "slow down" must be retried, while a 403
meaning "you lack permission" must fail immediately — retrying the latter just delays a
clear error behind a minute of pointless backoff.
"""

from __future__ import annotations

import pytest
from googleapiclient.errors import HttpError

from amy.google import retry


class _Resp:
    def __init__(self, status: int) -> None:
        self.status = status
        self.reason = "test"


def _http_error(status: int, body: str = "{}") -> HttpError:
    return HttpError(_Resp(status), body.encode("utf-8"), uri="https://example.test")


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_statuses_are_retried(status):
    assert retry.is_transient(_http_error(status))


def test_rate_limit_403_is_retried():
    body = '{"error": {"errors": [{"reason": "rateLimitExceeded"}]}}'
    assert retry.is_transient(_http_error(403, body))


def test_permission_403_is_not_retried():
    body = '{"error": {"errors": [{"reason": "insufficientPermissions"}]}}'
    assert not retry.is_transient(_http_error(403, body))


@pytest.mark.parametrize("status", [400, 401, 404])
def test_client_errors_are_not_retried(status):
    assert not retry.is_transient(_http_error(status))


def test_non_http_exceptions_are_not_retried():
    assert not retry.is_transient(ValueError("nope"))


def test_with_backoff_returns_once_the_call_succeeds(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda _: None)
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _http_error(429)
        return "ok"

    assert retry.with_backoff(flaky, base_delay=0.0) == "ok"
    assert attempts["n"] == 3


def test_with_backoff_gives_up_and_reraises(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda _: None)
    with pytest.raises(HttpError):
        retry.with_backoff(
            lambda: (_ for _ in ()).throw(_http_error(429)), attempts=2, base_delay=0.0
        )


def test_with_backoff_does_not_retry_a_permanent_failure(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda _: None)
    calls = {"n": 0}

    def forbidden():
        calls["n"] += 1
        raise _http_error(404)

    with pytest.raises(HttpError):
        retry.with_backoff(forbidden, base_delay=0.0)
    assert calls["n"] == 1  # failed fast, no backoff


def test_rate_limiter_spaces_calls(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(retry.time, "sleep", slept.append)
    clock = {"t": 100.0}
    monkeypatch.setattr(retry.time, "monotonic", lambda: clock["t"])

    limiter = retry.RateLimiter(min_interval=0.5)
    limiter.wait()  # first call sets the baseline
    limiter.wait()  # immediately after, so it must wait the full interval
    assert slept and slept[-1] == pytest.approx(0.5)
