"""Google Calendar access.

v1's CRUD is carried over. The listing strategy is new.

On sync strategy: Calendar offers syncToken-based incremental sync, but a request carrying
a syncToken may not also carry timeMin/timeMax, so using it means mirroring the entire
calendar history — years of events Donna will never reason about.

Instead we fully re-list a bounded *horizon* (a few weeks back, a few months forward). That
window is small enough to fetch cheaply and re-listing it means the part of the calendar
Donna actually reasons over is always exactly right, including deletions, with no cursor to
corrupt and no divergence to detect. Anything outside the horizon is irrelevant to her.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from donna.config import get_settings
from donna.google.auth import calendar as calendar_service
from donna.timeutil import iso_utc, now_utc, parse_iso

logger = logging.getLogger(__name__)

# How much of the calendar Donna mirrors. Past events matter for "what did I do last week";
# the forward window bounds how far ahead she can plan.
HORIZON_PAST_DAYS = 14
HORIZON_FUTURE_DAYS = 120


@dataclass(slots=True)
class Event:
    id: str
    summary: str
    description: str
    location: str
    start_raw: str
    end_raw: str
    start_ts: str | None
    end_ts: str | None
    all_day: bool
    status: str
    organizer: str
    attendees: list[str]
    html_link: str
    recurring_event_id: str | None
    updated_at: str | None


def _to_event(raw: dict[str, Any]) -> Event:
    start = raw.get("start") or {}
    end = raw.get("end") or {}
    # An all-day event carries "date"; a timed one carries "dateTime".
    all_day = "date" in start
    start_raw = start.get("dateTime") or start.get("date") or ""
    end_raw = end.get("dateTime") or end.get("date") or ""

    start_dt = parse_iso(start_raw)
    end_dt = parse_iso(end_raw)

    return Event(
        id=raw["id"],
        summary=raw.get("summary") or "(senza titolo)",
        description=raw.get("description") or "",
        location=raw.get("location") or "",
        start_raw=start_raw,
        end_raw=end_raw,
        start_ts=iso_utc(start_dt) if start_dt else None,
        end_ts=iso_utc(end_dt) if end_dt else None,
        all_day=all_day,
        status=raw.get("status") or "confirmed",
        organizer=(raw.get("organizer") or {}).get("email") or "",
        attendees=[a.get("email", "") for a in raw.get("attendees") or []],
        html_link=raw.get("htmlLink") or "",
        recurring_event_id=raw.get("recurringEventId"),
        updated_at=raw.get("updated"),
    )


def horizon_bounds() -> tuple[str, str]:
    now = now_utc()
    return (
        iso_utc(now - timedelta(days=HORIZON_PAST_DAYS)),
        iso_utc(now + timedelta(days=HORIZON_FUTURE_DAYS)),
    )


def list_horizon() -> list[Event]:
    """Every event in the mirrored window, recurrences expanded, deletions included.

    singleEvents=True expands recurring series into instances, which is what Donna needs:
    she reasons about "Tuesday at 19:00", not about an RRULE.
    """
    service = calendar_service()
    time_min, time_max = horizon_bounds()
    events: list[Event] = []
    page_token: str | None = None

    while True:
        response = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy="startTime",
                showDeleted=True,
                maxResults=250,
                pageToken=page_token,
            )
            .execute()
        )
        events.extend(_to_event(item) for item in response.get("items") or [])
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return events


def list_between(start_iso: str, end_iso: str) -> list[Event]:
    """Direct query, bypassing the mirror. Used to verify before writing."""
    response = (
        calendar_service()
        .events()
        .list(
            calendarId="primary",
            timeMin=start_iso,
            timeMax=end_iso,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    return [_to_event(item) for item in response.get("items") or []]


# ---------------------------------------------------------------- writes
def create_event(
    summary: str,
    start_iso: str,
    end_iso: str,
    *,
    description: str | None = None,
    location: str | None = None,
    reminder_minutes: int | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    body: dict[str, Any] = {
        "summary": summary,
        "start": {"dateTime": start_iso, "timeZone": settings.calendar_timezone},
        "end": {"dateTime": end_iso, "timeZone": settings.calendar_timezone},
    }
    if description:
        body["description"] = description
    if location:
        body["location"] = location
    if reminder_minutes is not None:
        # useDefault must be turned off, or Google ignores the overrides and applies the
        # calendar's own defaults instead — the reminder would silently not be the one asked
        # for. Google caps the lead time at four weeks.
        body["reminders"] = {
            "useDefault": False,
            "overrides": [
                {"method": "popup", "minutes": max(0, min(int(reminder_minutes), 40320))}
            ],
        }

    created = calendar_service().events().insert(calendarId="primary", body=body).execute()
    logger.info("Created calendar event %s (%s)", created.get("id"), summary)
    return created


def update_event(
    event_id: str,
    *,
    summary: str | None = None,
    start_iso: str | None = None,
    end_iso: str | None = None,
    description: str | None = None,
    location: str | None = None,
) -> dict[str, Any]:
    service = calendar_service()
    settings = get_settings()
    event = service.events().get(calendarId="primary", eventId=event_id).execute()

    if summary is not None:
        event["summary"] = summary
    if description is not None:
        event["description"] = description
    if location is not None:
        event["location"] = location
    if start_iso is not None:
        event["start"] = {"dateTime": start_iso, "timeZone": settings.calendar_timezone}
    if end_iso is not None:
        event["end"] = {"dateTime": end_iso, "timeZone": settings.calendar_timezone}

    updated = (
        service.events().update(calendarId="primary", eventId=event_id, body=event).execute()
    )
    logger.info("Updated calendar event %s", event_id)
    return updated


def delete_event(event_id: str) -> bool:
    calendar_service().events().delete(calendarId="primary", eventId=event_id).execute()
    logger.info("Deleted calendar event %s", event_id)
    return True
