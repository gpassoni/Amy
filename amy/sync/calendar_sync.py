"""Calendar -> local mirror.

Re-lists the whole horizon window each pass and reconciles. See amy/google/calendar.py
for why this beats syncToken here: a bounded window is cheap, and re-listing it means the
slice of calendar Amy reasons over is always exactly right, deletions included.
"""

from __future__ import annotations

import json
import logging
import time

from googleapiclient.errors import HttpError

from amy.google import calendar
from amy.store import repo
from amy.sync.base import SyncResult
from amy.timeutil import iso_utc, now_utc, parse_iso

logger = logging.getLogger(__name__)

RESOURCE = "calendar"


def _to_row(event: calendar.Event) -> dict[str, object]:
    return {
        "id": event.id,
        "calendar_id": "primary",
        "summary": event.summary,
        "description": event.description,
        "location": event.location,
        "start_ts": event.start_ts,
        "end_ts": event.end_ts,
        "start_raw": event.start_raw,
        "end_raw": event.end_raw,
        "all_day": int(event.all_day),
        "status": event.status,
        "organizer": event.organizer,
        "attendees": json.dumps(event.attendees),
        "html_link": event.html_link,
        "recurring_event_id": event.recurring_event_id,
        "updated_at": iso_utc(parse_iso(event.updated_at)) if event.updated_at else None,
    }


def sync_calendar() -> SyncResult:
    started = time.perf_counter()
    result = SyncResult(resource=RESOURCE, mode="horizon")

    try:
        events = calendar.list_horizon()
        start_iso, end_iso = calendar.horizon_bounds()

        # An event with no resolvable start cannot be placed on a timeline, and keeping it
        # would corrupt window queries. Count it rather than guessing at a date.
        rows, unusable = [], 0
        for event in events:
            if event.start_ts is None:
                unusable += 1
                continue
            rows.append(_to_row(event))

        counts = repo.replace_events_in_window(rows, start_iso, end_iso)
        result.updated = counts["upserted"]
        result.deleted = counts["deleted"]
        result.skipped = unusable
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        repo.record_sync(RESOURCE, cursor=iso_utc(now_utc()), items=result.updated)
        logger.info("Calendar sync: %s", result.summary())
        return result

    except (HttpError, OSError, RuntimeError) as exc:
        result.error = str(exc)
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        repo.record_sync(RESOURCE, error=str(exc))
        logger.error("Calendar sync failed: %s", exc, exc_info=True)
        return result
