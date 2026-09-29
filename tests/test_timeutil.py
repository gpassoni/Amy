"""Time handling is the most likely source of silent wrongness in this system, so it is
the most thoroughly tested part of it.

The cases that matter are the ones where a naive implementation looks correct in July and
breaks in December, or puts an event on the wrong day because it used UTC midnight.
"""
from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from donna import timeutil


def test_storage_format_is_uniform_so_string_order_is_time_order():
    # The whole schema relies on this: BETWEEN on TEXT columns must work.
    earlier = timeutil.iso_utc(datetime(2026, 9, 20, 10, 0, tzinfo=UTC))
    later = timeutil.iso_utc(datetime(2026, 9, 20, 11, 0, tzinfo=UTC))
    assert earlier < later
    assert earlier.endswith("+00:00")
    assert len(earlier) == len(later)


def test_naive_datetimes_are_local_not_utc():
    # Google hands back naive local times in places; assuming UTC would shift events.
    naive = datetime(2026, 9, 20, 15, 0)
    assert timeutil.iso_utc(naive) == "2026-09-20T13:00:00+00:00"  # CEST is UTC+2


def test_naive_datetime_in_winter_uses_cet_not_cest():
    # Same code path, different offset. A hardcoded +02:00 would fail here.
    assert timeutil.iso_utc(datetime(2026, 1, 15, 15, 0)) == "2026-01-15T14:00:00+00:00"


def test_parse_iso_accepts_trailing_z():
    parsed = timeutil.parse_iso("2026-09-20T13:00:00Z")
    assert parsed == datetime(2026, 9, 20, 13, 0, tzinfo=UTC)


def test_parse_iso_treats_bare_date_as_local_midnight():
    # Google all-day events arrive as "2026-09-24" with no time and no zone.
    parsed = timeutil.parse_iso("2026-09-24")
    assert parsed is not None
    assert timeutil.iso_utc(parsed) == "2026-09-23T22:00:00+00:00"


@pytest.mark.parametrize("bad", [None, "", "not a date", "2026-13-45"])
def test_parse_iso_returns_none_rather_than_raising(bad):
    # Sync must survive one malformed field without aborting the whole batch.
    assert timeutil.parse_iso(bad) is None


def test_parse_email_date_handles_rfc2822_with_offset():
    parsed = timeutil.parse_email_date("Sat, 20 Sep 2026 15:30:00 +0200")
    assert timeutil.iso_utc(parsed) == "2026-09-20T13:30:00+00:00"


def test_parse_email_date_survives_garbage():
    assert timeutil.parse_email_date("yesterday-ish") is None


def test_day_bounds_use_local_midnight_not_utc_midnight():
    # This is the bug that puts a 00:30 event on the previous day.
    start, end = timeutil.day_bounds_utc(date(2026, 9, 20))
    assert start == "2026-09-19T22:00:00+00:00"
    assert end == "2026-09-20T22:00:00+00:00"


def test_day_bounds_span_multiple_days():
    start, end = timeutil.day_bounds_utc(date(2026, 9, 20), days=7)
    assert start == "2026-09-19T22:00:00+00:00"
    assert end == "2026-09-26T22:00:00+00:00"


def test_day_bounds_across_dst_change_is_still_seven_local_days():
    # Europe/Rome leaves DST on 2026-10-25. A naive timedelta on UTC would drift an hour
    # and silently exclude or double-count an event at the boundary.
    start, end = timeutil.day_bounds_utc(date(2026, 10, 22), days=7)
    assert start == "2026-10-21T22:00:00+00:00"
    assert end == "2026-10-28T23:00:00+00:00"


def test_format_it_is_italian_and_local():
    formatted = timeutil.format_it(datetime(2026, 9, 24, 13, 0, tzinfo=UTC))
    assert formatted == "giovedì 24 settembre, 15:00"


def test_format_it_without_time():
    assert (
        timeutil.format_it(datetime(2026, 9, 24, 13, 0, tzinfo=UTC), with_time=False)
        == "giovedì 24 settembre"
    )


def test_format_range_collapses_same_day():
    start = datetime(2026, 9, 24, 13, 0, tzinfo=UTC)
    end = datetime(2026, 9, 24, 14, 30, tzinfo=UTC)
    assert timeutil.format_range_it(start, end) == "giovedì 24 settembre, 15:00–16:30"


def test_format_range_spans_days():
    start = datetime(2026, 9, 24, 13, 0, tzinfo=UTC)
    end = datetime(2026, 9, 25, 9, 0, tzinfo=UTC)
    assert "→" in timeutil.format_range_it(start, end)


@pytest.mark.parametrize(
    ("minutes", "expected"),
    [(30, "30min"), (60, "1h"), (90, "1h30"), (125, "2h05"), (0, "0min")],
)
def test_humanize_duration(minutes, expected):
    assert timeutil.humanize_duration(minutes) == expected
