"""Google Tasks access. v1's CRUD, plus a full listing for the mirror.

Personal task lists are small — a few hundred items at most — so the mirror re-lists
everything rather than tracking a cursor. Same reasoning as the calendar horizon: cheap
enough that always-correct beats incremental.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from donna.google.auth import tasks as tasks_service
from donna.timeutil import iso_utc, parse_iso

logger = logging.getLogger(__name__)

DEFAULT_LIST = "@default"


@dataclass(slots=True)
class Task:
    id: str
    title: str
    notes: str
    due_ts: str | None
    status: str
    completed_at: str | None
    updated_at: str | None
    position: str

    @property
    def is_open(self) -> bool:
        return self.status != "completed"


def _to_task(raw: dict[str, Any]) -> Task:
    due = parse_iso(raw.get("due"))
    completed = parse_iso(raw.get("completed"))
    return Task(
        id=raw["id"],
        title=raw.get("title") or "(senza titolo)",
        notes=raw.get("notes") or "",
        due_ts=iso_utc(due) if due else None,
        status=raw.get("status") or "needsAction",
        completed_at=iso_utc(completed) if completed else None,
        updated_at=raw.get("updated"),
        position=raw.get("position") or "",
    )


def list_all(*, include_completed: bool = True) -> list[Task]:
    service = tasks_service()
    items: list[Task] = []
    page_token: str | None = None

    while True:
        response = (
            service.tasks()
            .list(
                tasklist=DEFAULT_LIST,
                showCompleted=include_completed,
                showHidden=include_completed,
                maxResults=100,
                pageToken=page_token,
            )
            .execute()
        )
        items.extend(_to_task(item) for item in response.get("items") or [])
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return items


def create_task(title: str, *, notes: str | None = None, due_iso: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"title": title}
    if notes:
        body["notes"] = notes
    if due_iso:
        # Google Tasks stores only the date part of `due`, but still requires a full
        # RFC 3339 timestamp. Passing a bare date is rejected.
        body["due"] = due_iso

    created = tasks_service().tasks().insert(tasklist=DEFAULT_LIST, body=body).execute()
    logger.info("Created task %s (%s)", created.get("id"), title)
    return created


def complete_task(task_id: str) -> dict[str, Any]:
    service = tasks_service()
    task = service.tasks().get(tasklist=DEFAULT_LIST, task=task_id).execute()
    task["status"] = "completed"
    updated = service.tasks().update(tasklist=DEFAULT_LIST, task=task_id, body=task).execute()
    logger.info("Completed task %s", task_id)
    return updated
