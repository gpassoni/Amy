"""Date resolution, the weakest link in the whole pipeline.

Every case below either failed or misparsed at some point during development. dateparser
alone got 15 of these wrong or returned nothing; the normalisation layer in
amy/pipeline/dates.py exists entirely because of them.

Reference for all tests: Monday 21 September 2026, 09:00 Europe/Rome.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from amy.pipeline import dates
from amy.timeutil import parse_iso, to_local

REFERENCE = datetime(2026, 9, 21, 7, 0, tzinfo=UTC)  # 09:00 local, a Monday


def resolve(phrase, model_iso=None, **kwargs):
    return dates.resolve(phrase, model_iso, REFERENCE, **kwargs)


def local_start(result) -> str:
    return f"{to_local(parse_iso(result.start_ts)):%Y-%m-%d %H:%M}"


def duration_minutes(result) -> int:
    start, end = parse_iso(result.start_ts), parse_iso(result.end_ts)
    return int((end - start).total_seconds() // 60)


# ---------------------------------------------------------------- timed appointments
@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("giovedì 24 settembre alle 15:00", "2026-09-24 15:00"),
        # "ore" alone used to send this to 14 December.
        ("24/09/2026 ore 15:00", "2026-09-24 15:00"),
        ("24/09/2026 15:00", "2026-09-24 15:00"),
        # A leading article used to make dateparser return nothing at all.
        ("il 24 settembre alle 15", "2026-09-24 15:00"),
        ("3 ottobre alle 9:30", "2026-10-03 09:30"),
        ("domani alle 9:30", "2026-09-22 09:30"),
        ("stasera alle 21", "2026-09-21 21:00"),
        # dateparser does not know this word.
        ("dopodomani alle 14", "2026-09-23 14:00"),
    ],
)
def test_timed_phrases(phrase, expected):
    result = resolve(phrase)
    assert result is not None, f"{phrase!r} did not resolve"
    assert local_start(result) == expected
    assert not result.all_day


def test_day_month_order_is_italian_not_american():
    # 09/04 must be 9 April, never 4 September.
    result = resolve("09/04/2027 alle 10")
    assert local_start(result) == "2027-04-09 10:00"


# ---------------------------------------------------------------- weekdays
def test_bare_weekday_means_the_coming_one():
    assert local_start(resolve("martedì alle 15")) == "2026-09-22 15:00"


def test_weekday_prossimo_skips_a_week():
    assert local_start(resolve("martedì prossimo alle 15")) == "2026-09-29 15:00"


def test_the_same_weekday_as_today_means_next_week():
    # Said on a Monday, "lunedì" is not today.
    assert local_start(resolve("lunedì alle 10")) == "2026-09-28 10:00"


def test_a_numeric_day_wins_over_a_contradictory_weekday_name():
    # 24 September 2026 is a Thursday. Senders get this wrong; the number is authoritative.
    assert local_start(resolve("martedì 24 settembre alle 15:00")) == "2026-09-24 15:00"


# ---------------------------------------------------------------- all-day
@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("il 15 dicembre", "2026-12-15 00:00"),
        ("tutto il giorno del 2 novembre", "2026-11-02 00:00"),
        ("entro il 30 settembre", "2026-09-30 00:00"),
        ("dopodomani", "2026-09-23 00:00"),
    ],
)
def test_a_date_with_no_time_is_an_all_day_entry(phrase, expected):
    result = resolve(phrase)
    assert result is not None, f"{phrase!r} did not resolve"
    assert result.all_day
    assert local_start(result) == expected
    assert duration_minutes(result) == 1440


def test_all_day_starts_at_local_midnight_not_utc_midnight():
    result = resolve("il 15 dicembre")
    # December is CET (+01:00), so local midnight is 23:00 UTC the day before.
    assert result.start_ts == "2026-12-14T23:00:00+00:00"


def test_all_day_hint_can_be_forced():
    result = dates.resolve("24 settembre alle 15", None, REFERENCE, all_day_hint=True)
    assert result.all_day


# ---------------------------------------------------------------- durations
def test_default_duration_is_one_hour():
    assert duration_minutes(resolve("domani alle 9:30")) == 60


def test_explicit_duration_in_words():
    result = resolve("28 settembre alle 10 per due ore")
    assert local_start(result) == "2026-09-28 10:00"
    assert duration_minutes(result) == 120


def test_explicit_duration_in_minutes():
    assert duration_minutes(resolve("domani alle 9 per 30 minuti")) == 30


def test_a_time_range_starts_at_its_beginning():
    # This returned 16:00 at first: \balle does not match inside "dalle", so the general
    # pattern skipped the start time and matched the end one.
    result = resolve("venerdì dalle 14 alle 16")
    assert local_start(result) == "2026-09-25 14:00"
    assert duration_minutes(result) == 120


# ---------------------------------------------------------------- nothing to find
@pytest.mark.parametrize(
    "phrase",
    ["un giorno di questi", "dal 1998", "", None, "appena possibile", "il prima possibile"],
)
def test_phrases_with_no_real_date_resolve_to_nothing(phrase):
    # Most email mentions no appointment, so None is the common correct answer.
    assert resolve(phrase) is None


def test_a_date_far_in_the_past_is_rejected():
    # A signature footer or a "since 2019" must not become an appointment.
    assert resolve("15 marzo 2019") is None


def test_a_date_absurdly_far_ahead_is_rejected():
    assert resolve("15 marzo 2040") is None


# ---------------------------------------------------------------- model cross-check
def test_agreement_with_the_model_is_recorded():
    result = resolve("giovedì 24 settembre alle 15:00", "2026-09-24T15:00:00")
    assert result.agreed
    assert result.source == "agreed"


def test_small_disagreements_still_count_as_agreement():
    # Models routinely drop or round minutes; within an hour is not a real conflict.
    result = resolve("giovedì 24 settembre alle 15:00", "2026-09-24T15:30:00")
    assert result.agreed


def test_the_phrase_wins_when_the_model_disagrees():
    result = resolve("giovedì 24 settembre alle 15:00", "2026-11-30T09:00:00")
    assert local_start(result) == "2026-09-24 15:00"
    assert result.source == "phrase"
    assert not result.agreed
    assert result.note is not None  # the disagreement is surfaced, not hidden


def test_the_model_guess_is_used_when_there_is_no_phrase():
    result = resolve(None, "2026-09-25T10:00:00")
    assert local_start(result) == "2026-09-25 10:00"
    assert result.source == "model"


def test_an_implausible_model_guess_is_discarded():
    assert resolve(None, "1999-01-01T10:00:00") is None


def test_reference_time_anchors_relative_phrases():
    # Reprocessing an old email must yield the date it meant when it was written, not a
    # date relative to today.
    older = datetime(2026, 3, 2, 8, 0, tzinfo=UTC)  # Monday 2 March
    result = dates.resolve("domani alle 10", None, older)
    assert f"{to_local(parse_iso(result.start_ts)):%Y-%m-%d %H:%M}" == "2026-03-03 10:00"
