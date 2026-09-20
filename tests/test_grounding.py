"""Guards against the model asserting things that are not true.

Every test here corresponds to a real answer Donna gave. The pattern across all of them is the
same: the model is fluent and confident about details it has no basis for, and the fix is
never a firmer instruction — it is removing the opportunity to guess.
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import pytest

from donna.agents import base, fallback, prefetch
from donna.context import builder
from donna.store import repo
from donna.store.db import Database
from donna.timeutil import iso_utc, now_local, now_utc, to_utc


def _event(db: Database, *, days_ahead: int, hour: int, end_hour: int, summary: str) -> None:
    start = to_utc(
        now_local().replace(hour=hour, minute=0, second=0, microsecond=0)
        + timedelta(days=days_ahead)
    )
    end = to_utc(
        now_local().replace(hour=end_hour, minute=30, second=0, microsecond=0)
        + timedelta(days=days_ahead)
    )
    repo.replace_events_in_window(
        [
            {
                "id": f"e{days_ahead}-{hour}", "calendar_id": "primary", "summary": summary,
                "description": "", "location": "", "start_ts": iso_utc(start),
                "end_ts": iso_utc(end), "start_raw": iso_utc(start), "end_raw": iso_utc(end),
                "all_day": 0, "status": "confirmed", "organizer": "", "attendees": "[]",
                "html_link": "", "recurring_event_id": None, "updated_at": None,
            }
        ],
        iso_utc(start - timedelta(hours=1)),
        iso_utc(end + timedelta(hours=1)),
    )


# ---------------------------------------------------------------- the end time
def test_the_week_list_shows_when_events_end(db: Database):
    """Asked to schedule something "after work" on a day ending at 16:30, Donna said work
    finished at 18:00 — the week list only printed start times, so she could not have known."""
    _event(db, days_ahead=2, hour=8, end_hour=16, summary="Lavoro")
    week = builder.build().week
    line = next(l for l in week if "Lavoro" in l)
    assert "08:00" in line
    assert "16:30" in line, "senza l'ora di fine, 'dopo il lavoro' è indovinato"


def test_todays_events_also_show_their_end(db: Database):
    _event(db, days_ahead=0, hour=9, end_hour=10, summary="Riunione")
    assert any("–10:30" in line for line in builder.build().today)


# ---------------------------------------------------------------- absence
def test_a_day_with_nothing_on_it_is_listed_as_empty(db: Database):
    """Absence has to be stated, not inferred from a gap in a list.

    With tomorrow simply missing from the week list, the model borrowed an end time from the
    other days and asserted that work finished at 16:30 on a day with no work at all.
    """
    _event(db, days_ahead=2, hour=8, end_hour=16, summary="Lavoro")
    week = builder.build().week
    assert any("niente in programma" in line for line in week)


def test_the_day_prefetch_says_plainly_when_a_day_is_empty(db: Database):
    block = prefetch.day_agenda(1)
    assert "NIENTE" in block
    assert "inventare" in block


def test_the_day_prefetch_reports_the_last_end_time(db: Database):
    _event(db, days_ahead=1, hour=8, end_hour=16, summary="Lavoro")
    block = prefetch.day_agenda(1)
    assert "16:30" in block
    assert "finisce alle 16:30" in block


@pytest.mark.parametrize(
    ("message", "offset"),
    [
        ("domani dopo il lavoro", 1),
        ("dopodomani alle 10", 2),
        ("stasera", 0),
        ("quante proposte hai", None),
    ],
)
def test_day_references_are_resolved(message, offset):
    assert prefetch.referenced_day(message) == offset


def test_dates_are_spelled_out_rather_than_left_as_arithmetic(db: Database):
    # Given only "domenica 20 settembre", the model resolved "domani" to a date a week away.
    line = builder.build().now_line
    assert "domani è" in line
    assert "dopodomani è" in line


# ---------------------------------------------------------------- false claims
@pytest.mark.parametrize(
    "text",
    [
        'Ti ho inserito "Lavaggio moto" dalle 19:00 alle 20:00.',
        "Ho preparato una proposta per le 17:30.",
        "Te l'ho messa in calendario.",
        "Proposta creata, confermala tu.",
        "L'appuntamento è ora in calendario.",
    ],
)
def test_claiming_an_action_without_calling_a_tool_is_caught(text):
    """The failure this exists for: «Ti ho inserito "Lavaggio moto"…» with no tool called.

    Nothing existed, and the sentence was indistinguishable from a successful one.
    """
    assert base._claims_without_doing(base.AgentReply(text=text, agent="schedule"))


@pytest.mark.parametrize(
    "text",
    [
        "Domani non hai lavoro in calendario.",
        "Vuoi che te la prepari?",
        "Ti va bene se lo metto alle 17:30?",
        "Giovedì sei libero dalle 16:30.",
    ],
)
def test_honest_replies_are_not_flagged(text):
    assert not base._claims_without_doing(base.AgentReply(text=text, agent="schedule"))


def test_a_claim_is_fine_when_a_tool_actually_ran():
    reply = base.AgentReply(
        text="Te l'ho preparata.", agent="schedule",
        tool_calls=[("proponi_evento", "PROPOSTA #7")],
    )
    assert not base._claims_without_doing(reply)


# ---------------------------------------------------------------- structured fallback
def _fake_llm(value):
    class _Result:
        def __init__(self, v):
            self.value, self.model, self.trace_id = v, "qwen3.5:9b", "t1"
            self.latency_ms, self.attempts = 10, 1

    class _Client:
        def structured(self, *a, **k):
            return _Result(value)

    return _Client()


def test_the_fallback_creates_a_proposal_when_the_tool_was_refused(db: Database):
    """Tool calling failed three times for this intent, once right after agreeing to comply.

    Filling a schema is not a decision the model gets to make — the grammar leaves it no
    alternative — so the request is re-asked as a structured extraction instead.
    """
    start = (now_local() + timedelta(days=2)).replace(hour=17, minute=30, second=0, microsecond=0)
    request = fallback.ScheduleRequest(
        ragionamento="il lavoro finisce alle 16:30",
        titolo="Lavaggio moto",
        inizio_iso=f"{start:%Y-%m-%dT%H:%M}",
        fine_iso=f"{start + timedelta(hours=1):%Y-%m-%dT%H:%M}",
        promemoria_minuti=10,
    )
    with patch.object(fallback, "get_llm", return_value=_fake_llm(request)):
        outcome = fallback.propose_from_request("lava la moto", context="")

    assert outcome is not None and outcome.proposal_id is not None
    payload = repo.proposal_payload(repo.get_proposal(outcome.proposal_id))
    assert payload["title"] == "Lavaggio moto"
    assert payload["reminder_minutes"] == 10


def test_the_fallback_keeps_a_genuine_question_instead_of_inventing(db: Database):
    request = fallback.ScheduleRequest(
        ragionamento="non so quando", titolo="Cena",
        dati_sufficienti=False, domanda="A che ora vuoi cenare?",
    )
    with patch.object(fallback, "get_llm", return_value=_fake_llm(request)):
        outcome = fallback.propose_from_request("organizza una cena", context="")

    assert outcome.proposal_id is None
    assert outcome.text == "A che ora vuoi cenare?"
    assert repo.pending_proposal_count() == 0


def test_the_fallback_refuses_a_date_in_the_past(db: Database):
    # A misparse, not an intention. Filing a proposal for last Tuesday is worse than silence.
    past = now_local() - timedelta(days=3)
    request = fallback.ScheduleRequest(
        ragionamento="x", titolo="Vecchio", inizio_iso=f"{past:%Y-%m-%dT%H:%M}"
    )
    with patch.object(fallback, "get_llm", return_value=_fake_llm(request)):
        assert fallback.propose_from_request("qualcosa", context="") is None
    assert repo.pending_proposal_count() == 0


def test_the_fallback_defaults_a_missing_end_to_one_hour(db: Database):
    start = (now_local() + timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0)
    request = fallback.ScheduleRequest(
        ragionamento="x", titolo="Cosa", inizio_iso=f"{start:%Y-%m-%dT%H:%M}", fine_iso=None
    )
    with patch.object(fallback, "get_llm", return_value=_fake_llm(request)):
        outcome = fallback.propose_from_request("cosa", context="")
    payload = repo.proposal_payload(repo.get_proposal(outcome.proposal_id))
    from donna.timeutil import parse_iso

    span = parse_iso(payload["end_ts"]) - parse_iso(payload["start_ts"])
    assert span == timedelta(hours=1)


# ---------------------------------------------------------------- reminders
def test_a_reminder_reaches_google_as_an_override(db: Database):
    """useDefault must be false, or Google quietly applies the calendar's own defaults and the
    reminder is not the one that was asked for."""
    from donna.google import calendar

    with patch.object(calendar, "calendar_service") as service:
        service.return_value.events.return_value.insert.return_value.execute.return_value = {
            "id": "e1"
        }
        calendar.create_event("X", "2026-12-01T09:00:00+00:00", "2026-12-01T10:00:00+00:00",
                              reminder_minutes=10)

    body = service.return_value.events.return_value.insert.call_args.kwargs["body"]
    assert body["reminders"]["useDefault"] is False
    assert body["reminders"]["overrides"][0]["minutes"] == 10


def test_an_event_without_a_reminder_leaves_the_defaults_alone(db: Database):
    from donna.google import calendar

    with patch.object(calendar, "calendar_service") as service:
        service.return_value.events.return_value.insert.return_value.execute.return_value = {
            "id": "e1"
        }
        calendar.create_event("X", "2026-12-01T09:00:00+00:00", "2026-12-01T10:00:00+00:00")

    body = service.return_value.events.return_value.insert.call_args.kwargs["body"]
    assert "reminders" not in body
