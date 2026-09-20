"""Google Tasks -> local mirror. Full re-list and reconcile; personal lists are small."""
from __future__ import annotations

import logging
import time

from googleapiclient.errors import HttpError

from donna.google import tasks
from donna.store import repo
from donna.sync.base import SyncResult
from donna.timeutil import iso_utc, now_utc, parse_iso

logger = logging.getLogger(__name__)

RESOURCE = "tasks"


def _to_row(task: tasks.Task) -> dict[str, object]:
    return {
        "id": task.id,
        "tasklist_id": tasks.DEFAULT_LIST,
        "title": task.title,
        "notes": task.notes,
        "due_ts": task.due_ts,
        "status": task.status,
        "completed_at": task.completed_at,
        "updated_at": iso_utc(parse_iso(task.updated_at)) if task.updated_at else None,
        "position": task.position,
    }


def sync_tasks() -> SyncResult:
    started = time.perf_counter()
    result = SyncResult(resource=RESOURCE, mode="full")

    try:
        items = tasks.list_all()
        counts = repo.replace_all_tasks([_to_row(t) for t in items])
        result.updated = counts["upserted"]
        result.deleted = counts["deleted"]
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        repo.record_sync(RESOURCE, cursor=iso_utc(now_utc()), items=result.updated)
        logger.info("Tasks sync: %s", result.summary())
        return result

    except (HttpError, OSError, RuntimeError) as exc:
        result.error = str(exc)
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        repo.record_sync(RESOURCE, error=str(exc))
        logger.error("Tasks sync failed: %s", exc, exc_info=True)
        return result
