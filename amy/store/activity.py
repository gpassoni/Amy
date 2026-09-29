"""Recording what Amy is doing.

Used as a context manager, so the finish is not something a caller can forget:

    with activity.record("turn", "inbox", channel="telegram") as run:
        ...
        run.note(intent="inbox_query", tools=["cerca_email"])

The row is written on entry with status 'running', which is what makes the dashboard's live
view possible — anything still running is happening now. An exception marks it failed and
re-raises; it never swallows the error.

Every trace produced inside the block is linked to the activity by passing `run.id` as the
parent trace id, so the dashboard can go from "the inbox agent answered this" down to the
exact prompt that produced it.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from amy.store.db import get_db
from amy.timeutil import iso_utc, now_utc

logger = logging.getLogger(__name__)

TURN = "turn"
CYCLE = "cycle"
SYNC = "sync"
TRIAGE = "triage"
EXTRACT = "extract"
NOTIFY = "notify"

# A run still 'running' after this long is almost certainly the victim of a crash, not slow
# work. The dashboard shows those as stale rather than pretending they are live.
STALE_AFTER_SECONDS = 600


@dataclass(slots=True)
class Run:
    id: str
    kind: str
    actor: str
    summary: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def note(self, **fields: Any) -> None:
        """Attach structured facts as they become known during the run."""
        self.detail.update(fields)

    def describe(self, summary: str) -> None:
        self.summary = summary


@contextmanager
def record(
    kind: str,
    actor: str,
    *,
    channel: str | None = None,
    chat_id: str | None = None,
    summary: str | None = None,
) -> Iterator[Run]:
    run = Run(id=uuid.uuid4().hex, kind=kind, actor=actor, summary=summary)
    started = time.perf_counter()

    _insert(run, channel=channel, chat_id=chat_id)
    try:
        yield run
    except Exception as exc:
        _finish(run, status="failed", duration_ms=_ms(started), error=str(exc)[:500])
        raise
    else:
        _finish(run, status="done", duration_ms=_ms(started))


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _insert(run: Run, *, channel: str | None, chat_id: str | None) -> None:
    try:
        get_db().execute(
            "INSERT INTO activity (id, kind, actor, status, summary, detail, channel, chat_id,"
            " started_at) VALUES (?,?,?,'running',?,?,?,?,?)",
            (
                run.id,
                run.kind,
                run.actor,
                run.summary,
                json.dumps(run.detail, ensure_ascii=False, default=str),
                channel,
                chat_id,
                iso_utc(now_utc()),
            ),
        )
    except Exception:
        # Observability must never break the thing it observes.
        logger.warning("Could not record activity start for %s", run.kind, exc_info=True)


def _finish(run: Run, *, status: str, duration_ms: int, error: str | None = None) -> None:
    try:
        get_db().execute(
            "UPDATE activity SET status = ?, summary = ?, detail = ?, finished_at = ?,"
            " duration_ms = ?, error = ? WHERE id = ?",
            (
                status,
                run.summary,
                json.dumps(run.detail, ensure_ascii=False, default=str),
                iso_utc(now_utc()),
                duration_ms,
                error,
                run.id,
            ),
        )
    except Exception:
        logger.warning("Could not record activity end for %s", run.kind, exc_info=True)


def reassign(run: Run) -> None:
    """Update a running row's actor and detail mid-flight.

    A turn starts as "router" because the agent is not known until routing has run, and the
    live view should show the real agent the moment it is chosen rather than after the turn
    ends.
    """
    try:
        get_db().execute(
            "UPDATE activity SET actor = ?, detail = ? WHERE id = ?",
            (run.actor, json.dumps(run.detail, ensure_ascii=False, default=str), run.id),
        )
    except Exception:
        logger.warning("Could not reassign activity %s", run.id, exc_info=True)


# ---------------------------------------------------------------- queries
def running() -> list:
    return get_db().query(
        "SELECT * FROM activity WHERE status = 'running' ORDER BY started_at DESC"
    )


def recent(limit: int = 30, *, kind: str | None = None) -> list:
    if kind:
        return get_db().query(
            "SELECT * FROM activity WHERE kind = ? ORDER BY started_at DESC LIMIT ?",
            (kind, limit),
        )
    return get_db().query("SELECT * FROM activity ORDER BY started_at DESC LIMIT ?", (limit,))


def get(activity_id: str):
    return get_db().query_one("SELECT * FROM activity WHERE id = ?", (activity_id,))


def traces_for(activity_id: str) -> list:
    """The model calls made during a run, oldest first."""
    return get_db().query(
        "SELECT * FROM traces WHERE parent_id = ? ORDER BY created_at, rowid", (activity_id,)
    )


def actor_stats(limit: int = 20) -> list:
    """Per-agent totals, for the roster view."""
    return get_db().query(
        "SELECT actor, kind, count(*) AS runs,"
        " sum(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failures,"
        " avg(duration_ms) AS avg_ms, max(started_at) AS last_at"
        " FROM activity GROUP BY actor, kind ORDER BY runs DESC LIMIT ?",
        (limit,),
    )


def detail_of(row) -> dict[str, Any]:
    try:
        return json.loads(row["detail"] or "{}")
    except (ValueError, TypeError):
        return {}


def sweep_stale(seconds: int = STALE_AFTER_SECONDS) -> int:
    """Mark long-abandoned 'running' rows as failed.

    Called at startup: a row left running is the fingerprint of a crash mid-turn, and leaving
    it in the live view forever would make the dashboard lie.
    """
    from datetime import timedelta

    cutoff = iso_utc(now_utc() - timedelta(seconds=seconds))
    cursor = get_db().execute(
        "UPDATE activity SET status = 'failed', error = 'interrotto (processo terminato?)',"
        " finished_at = ? WHERE status = 'running' AND started_at < ?",
        (iso_utc(now_utc()), cutoff),
    )
    if cursor.rowcount:
        logger.info("%d attività rimaste appese marcate come interrotte", cursor.rowcount)
    return cursor.rowcount
