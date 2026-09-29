"""Time handling, in one place.

Rules enforced here, because getting this wrong is the most likely source of silent bugs:

  * Every *_ts column in SQLite holds a UTC ISO-8601 string with a "+00:00" offset.
    The format is uniform, so lexicographic ordering equals chronological ordering and
    plain SQL BETWEEN works on it.
  * Anything shown to the user, or fed to a model as "now", is in the user's local zone.
  * Date arithmetic is done here in code. Models are never asked to compute a date —
    they extract the phrase, we resolve it.
"""
from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

from donna.config import get_settings

UTC = UTC

_WEEKDAYS_IT = (
    "lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica",
)
_MONTHS_IT = (
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
    "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
)


def local_zone() -> ZoneInfo:
    return ZoneInfo(get_settings().calendar_timezone)


def now_utc() -> datetime:
    return datetime.now(UTC)


def now_local() -> datetime:
    return datetime.now(local_zone())


def to_utc(dt: datetime) -> datetime:
    """Naive datetimes are assumed to be in the user's local zone, not UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=local_zone())
    return dt.astimezone(UTC)


def to_local(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(local_zone())


def iso_utc(dt: datetime) -> str:
    """Canonical storage form. Always ends in +00:00."""
    return to_utc(dt).isoformat(timespec="seconds")


def parse_iso(value: str | None) -> datetime | None:
    """Parse a stored timestamp, or any ISO-8601 string Google hands us.

    Accepts a trailing "Z", and treats a bare date as local midnight.
    """
    if not value:
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=local_zone())
    return parsed


def parse_email_date(value: str | None) -> datetime | None:
    """Parse an RFC 2822 Date: header, as Gmail returns it."""
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=local_zone())
    return parsed


def day_bounds_utc(day: date, *, days: int = 1) -> tuple[str, str]:
    """UTC ISO bounds covering `days` local days starting at local midnight of `day`.

    Used for "today", "tomorrow", "this week" queries — the boundaries must be local
    midnight, not UTC midnight, or events land on the wrong day.
    """
    zone = local_zone()
    start = datetime.combine(day, time.min, tzinfo=zone)
    end = start + timedelta(days=days)
    return iso_utc(start), iso_utc(end)


def window_utc(start: datetime, days: int) -> tuple[str, str]:
    return iso_utc(start), iso_utc(start + timedelta(days=days))


def format_it(dt: datetime, *, with_time: bool = True) -> str:
    """Human, Italian, local zone: 'martedì 24 settembre, 15:00'."""
    local = to_local(dt)
    stamp = f"{_WEEKDAYS_IT[local.weekday()]} {local.day} {_MONTHS_IT[local.month - 1]}"
    return f"{stamp}, {local:%H:%M}" if with_time else stamp


def format_range_it(start: datetime, end: datetime | None) -> str:
    if end is None:
        return format_it(start)
    local_start, local_end = to_local(start), to_local(end)
    if local_start.date() == local_end.date():
        return f"{format_it(local_start)}–{local_end:%H:%M}"
    return f"{format_it(local_start)} → {format_it(local_end)}"


def weekday_name_it(dt: datetime) -> str:
    return _WEEKDAYS_IT[to_local(dt).weekday()]


def humanize_duration(minutes: int) -> str:
    hours, mins = divmod(int(minutes), 60)
    if hours and mins:
        return f"{hours}h{mins:02d}"
    if hours:
        return f"{hours}h"
    return f"{mins}min"
