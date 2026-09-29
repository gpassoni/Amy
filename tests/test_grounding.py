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


def _action(**overrides) -> fallback.ScheduleAction:
    base_fields = dict(
        frase="lava la moto", tipo="crea", cosa_cambia="niente", ragionamento="x", titolo="Cosa", id_evento=None,
        inizio_iso=None, fine_iso=None, promemoria_minuti=0, dati_sufficienti=True, domanda=None,
    )
    return fallback.ScheduleAction(**{**base_fields, **overrides})


def _local(days: int, hour: int, minute: int = 0) -> str:
    when = (now_local() + timedelta(days=days)).replace(hour=hour, minute=minute, second=0, microsecond=0)
    return f"{when:%Y-%m-%dT%H:%M}"


def _plan(*actions, count: int | None = None) -> fallback.SchedulePlan:
    return fallback.SchedulePlan(numero_richieste=count or len(actions), azioni=list(actions))


def test_the_fallback_creates_a_proposal_when_the_tool_was_refused(db: Database):
    """Tool calling failed three times for this intent, once right after agreeing to comply.

    Filling a schema is not a decision the model gets to make — the grammar leaves it no
    alternative — so the request is re-asked as a structured extraction instead.
    """
    plan = _plan(
        _action(
            titolo="Lavaggio moto", ragionamento="il lavoro finisce alle 16:30",
            inizio_iso=_local(2, 17, 30), fine_iso=_local(2, 18, 30), promemoria_minuti=10,
        )
    )
    with patch.object(fallback, "get_llm", return_value=_fake_llm(plan)):
        outcome = fallback.propose_from_request("lava la moto", context="")

    assert outcome is not None and outcome.proposal_id is not None
    payload = repo.proposal_payload(repo.get_proposal(outcome.proposal_id))
    assert payload["title"] == "Lavaggio moto"
    assert payload["reminder_minutes"] == 10


def test_two_requests_in_one_message_become_two_proposals(db: Database):
    """The real failure: «Aggiungi una corsa di 10km domenica pomeriggio. Inserisci anche uno
    slot di burocrazia la mattina» produced one proposal (#23) and a confident «Fatto:». The
    model had understood both — its reasoning said so — but the schema held a single event."""
    message = "Aggiungi una corsa di 10km domenica pomeriggio. Inserisci anche uno slot di burocrazia la mattina"
    plan = _plan(
        _action(frase="Aggiungi una corsa di 10km domenica pomeriggio", titolo="Corsa 10km",
                inizio_iso=_local(3, 16, 30), fine_iso=_local(3, 18)),
        _action(frase="Inserisci anche uno slot di burocrazia la mattina", titolo="Burocrazia",
                inizio_iso=_local(3, 9), fine_iso=_local(3, 10)),
    )
    with patch.object(fallback, "get_llm", return_value=_fake_llm(plan)):
        outcome = fallback.propose_from_request(message, context="")

    assert len(outcome.proposal_ids) == 2
    assert repo.pending_proposal_count() == 2
    assert "Corsa 10km" in outcome.text and "Burocrazia" in outcome.text
    assert "2 proposte" in outcome.text


def test_an_action_with_no_basis_in_the_message_is_dropped_and_reported(db: Database):
    plan = _plan(
        _action(frase="lava la moto giovedì", titolo="Moto", inizio_iso=_local(2, 17)),
        _action(frase="prenota una cena al ristorante giapponese", titolo="Cena",
                inizio_iso=_local(3, 20)),
    )
    with patch.object(fallback, "get_llm", return_value=_fake_llm(plan)):
        outcome = fallback.propose_from_request("lava la moto giovedì", context="")

    assert len(outcome.proposal_ids) == 1
    assert "Non ho preparato" in outcome.text
    assert "non l'ho ritrovata" in outcome.text


def test_a_shortfall_between_counted_and_handled_requests_is_stated(db: Database):
    """If the model counts three requests and returns two actions, saying nothing would be
    the original bug again — a part of the request gone without a trace."""
    plan = _plan(
        _action(frase="lava la moto", titolo="Moto", inizio_iso=_local(2, 17)),
        _action(frase="e vai in piscina", titolo="Piscina", inizio_iso=_local(3, 17)),
        count=3,
    )
    with patch.object(fallback, "get_llm", return_value=_fake_llm(plan)):
        outcome = fallback.propose_from_request("lava la moto e vai in piscina e poi il dentista", context="")

    assert len(outcome.proposal_ids) == 2
    assert "contato 3 richieste" in outcome.text


def test_one_unclear_part_does_not_block_the_others(db: Database):
    plan = _plan(
        _action(frase="lava la moto", titolo="Moto", inizio_iso=_local(2, 17)),
        _action(frase="organizza una cena", titolo="Cena", dati_sufficienti=False,
                domanda="A che ora vuoi cenare?"),
    )
    with patch.object(fallback, "get_llm", return_value=_fake_llm(plan)):
        outcome = fallback.propose_from_request("lava la moto e organizza una cena", context="")

    assert len(outcome.proposal_ids) == 1
    assert "A che ora vuoi cenare?" in outcome.text
    assert repo.pending_proposal_count() == 1


def test_the_fallback_keeps_a_genuine_question_instead_of_inventing(db: Database):
    plan = _plan(
        _action(frase="organizza una cena", titolo="Cena", dati_sufficienti=False,
                domanda="A che ora vuoi cenare?")
    )
    with patch.object(fallback, "get_llm", return_value=_fake_llm(plan)):
        outcome = fallback.propose_from_request("organizza una cena", context="")

    assert outcome.proposal_id is None
    assert "A che ora vuoi cenare?" in outcome.text
    assert repo.pending_proposal_count() == 0


def test_the_fallback_refuses_a_date_in_the_past(db: Database):
    # A misparse, not an intention. Filing a proposal for last Tuesday is worse than silence —
    # and the reply says so instead of pretending nothing was asked.
    plan = _plan(_action(frase="qualcosa", titolo="Vecchio", inizio_iso=_local(-3, 10)))
    with patch.object(fallback, "get_llm", return_value=_fake_llm(plan)):
        outcome = fallback.propose_from_request("qualcosa", context="")
    assert repo.pending_proposal_count() == 0
    assert outcome is not None and "Non ho preparato" in outcome.text


def test_an_empty_plan_hands_the_message_back_to_the_agent(db: Database):
    with patch.object(fallback, "get_llm", return_value=_fake_llm(_plan(count=0))):
        assert fallback.propose_from_request("che tempo fa", context="") is None


def test_the_fallback_defaults_a_missing_end_to_one_hour(db: Database):
    plan = _plan(_action(frase="cosa", titolo="Cosa", inizio_iso=_local(1, 10), fine_iso=None))
    with patch.object(fallback, "get_llm", return_value=_fake_llm(plan)):
        outcome = fallback.propose_from_request("cosa", context="")
    payload = repo.proposal_payload(repo.get_proposal(outcome.proposal_id))
    from donna.timeutil import parse_iso

    assert parse_iso(payload["end_ts"]) - parse_iso(payload["start_ts"]) == timedelta(hours=1)


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ("lava la moto giovedì", True),
        ("Lava la moto, giovedì!", True),                       # punctuation and case slips
        ("prenota una cena al ristorante giapponese", False),   # invented
        ("", False),
    ],
)
def test_quote_grounding_tolerates_slips_but_not_inventions(quote, expected):
    assert fallback.is_grounded(quote, "lava la moto giovedì e vai in piscina") is expected


# ---------------------------------------------------------------- changing what exists
def test_a_shift_that_ends_earlier_is_a_move_not_a_new_event(db: Database):
    """The real failure (#22): «mercoledì finisce alle 15:00» was filed as a new event called
    'Aggiornamento turno', because the schema could only create. It is a change to an event that
    already exists, and the start it did not mention must be kept."""
    _event(db, days_ahead=2, hour=8, end_hour=16, summary="Lavoro")      # 08:00–16:30
    plan = _plan(_action(frase="finisce alle 15:00", tipo="sposta", cosa_cambia="solo_fine", id_evento="e2-8",
                         fine_iso=_local(2, 15)))
    with patch.object(fallback, "get_llm", return_value=_fake_llm(plan)):
        outcome = fallback.propose_from_request("il turno di mercoledì finisce alle 15:00", context="")

    row = repo.get_proposal(outcome.proposal_id)
    assert row["kind"] == "calendar_move"
    payload = repo.proposal_payload(row)
    assert payload["title"] == "Lavoro"                     # from the calendar, not the model
    assert payload["start_ts"] == payload["old_start_ts"]   # the unmentioned start is kept
    assert payload["end_ts"] != payload["old_end_ts"]
    assert outcome.made[0][1] == "sposta_evento"


def test_moving_only_the_start_keeps_the_duration(db: Database):
    _event(db, days_ahead=2, hour=9, end_hour=10, summary="Riunione")    # 09:00–10:30
    plan = _plan(_action(frase="sposta la riunione alle 14", tipo="sposta", cosa_cambia="intero", id_evento="e2-9",
                         inizio_iso=_local(2, 14)))
    with patch.object(fallback, "get_llm", return_value=_fake_llm(plan)):
        outcome = fallback.propose_from_request("sposta la riunione alle 14", context="")

    from donna.timeutil import parse_iso

    payload = repo.proposal_payload(repo.get_proposal(outcome.proposal_id))
    assert parse_iso(payload["end_ts"]) - parse_iso(payload["start_ts"]) == timedelta(hours=1, minutes=30)


def test_a_shift_that_starts_later_keeps_its_end(db: Database):
    """«Giovedì inizio alle 9 invece che alle 8» came back as 09:00–17:30 twice: the model shifted
    the whole shift and no prompt wording stopped it. Which half changes is now a field it
    classifies, and the untouched half is taken from the event by code."""
    _event(db, days_ahead=2, hour=8, end_hour=16, summary="Lavoro")      # 08:00–16:30
    plan = _plan(_action(frase="inizio alle 9", tipo="sposta", cosa_cambia="solo_inizio",
                         id_evento="e2-8", inizio_iso=_local(2, 9), fine_iso=_local(2, 17, 30)))
    with patch.object(fallback, "get_llm", return_value=_fake_llm(plan)):
        outcome = fallback.propose_from_request("inizio alle 9", context="")

    payload = repo.proposal_payload(repo.get_proposal(outcome.proposal_id))
    assert payload["end_ts"] == payload["old_end_ts"]                    # end untouched, whatever the model said
    assert payload["start_ts"] != payload["old_start_ts"]


def test_the_model_cannot_move_the_unchanged_half_of_a_shift(db: Database):
    _event(db, days_ahead=2, hour=8, end_hour=16, summary="Lavoro")
    plan = _plan(_action(frase="finisce alle 15", tipo="sposta", cosa_cambia="solo_fine",
                         id_evento="e2-8", inizio_iso=_local(2, 15), fine_iso=_local(2, 15)))
    with patch.object(fallback, "get_llm", return_value=_fake_llm(plan)):
        outcome = fallback.propose_from_request("finisce alle 15", context="")

    payload = repo.proposal_payload(repo.get_proposal(outcome.proposal_id))
    assert payload["start_ts"] == payload["old_start_ts"]                # its wrong start is ignored


def test_a_decorated_event_id_still_resolves(db: Database):
    """The model copies ids from «[abc] domani — Lavoro» with the brackets attached."""
    _event(db, days_ahead=2, hour=9, end_hour=10, summary="Riunione")
    plan = _plan(_action(frase="cancella la riunione", tipo="elimina", id_evento="[e2-9]"))
    with patch.object(fallback, "get_llm", return_value=_fake_llm(plan)):
        outcome = fallback.propose_from_request("cancella la riunione", context="")
    assert outcome.proposal_id is not None


def test_every_plan_field_is_required_in_the_schema():
    """An optional field is one the grammar lets the model skip. It skipped the title and both
    times once, and two understood requests came back as «non ho una data utilizzabile»."""
    required = set(fallback.ScheduleAction.model_json_schema()["required"])
    assert required == set(fallback.ScheduleAction.model_fields)


def test_an_invented_event_id_is_not_turned_into_a_proposal(db: Database):
    plan = _plan(_action(frase="sposta il dentista", tipo="sposta", cosa_cambia="intero", id_evento="inventato",
                         inizio_iso=_local(2, 14)))
    with patch.object(fallback, "get_llm", return_value=_fake_llm(plan)):
        outcome = fallback.propose_from_request("sposta il dentista", context="")

    assert repo.pending_proposal_count() == 0
    assert "non trovo in calendario" in outcome.text


def test_accepting_a_move_updates_google_and_the_mirror(db: Database):
    from donna.pipeline import resolve

    _event(db, days_ahead=2, hour=8, end_hour=16, summary="Lavoro")
    plan = _plan(_action(frase="finisce alle 15:00", tipo="sposta", cosa_cambia="solo_fine", id_evento="e2-8",
                         fine_iso=_local(2, 15)))
    with patch.object(fallback, "get_llm", return_value=_fake_llm(plan)):
        outcome = fallback.propose_from_request("il turno finisce alle 15:00", context="")

    with patch.object(resolve.calendar, "update_event", return_value={"htmlLink": "x"}) as update:
        result = resolve.accept(outcome.proposal_id, via="test")

    assert result.state == "accepted"
    assert update.call_args.args == ("e2-8",)
    assert repo.get_event("e2-8")["end_ts"] == repo.proposal_payload(repo.get_proposal(outcome.proposal_id))["end_ts"]


def test_accepting_a_delete_removes_it_from_google_and_the_mirror(db: Database):
    from donna.pipeline import resolve

    _event(db, days_ahead=2, hour=9, end_hour=10, summary="Riunione")
    plan = _plan(_action(frase="cancella la riunione", tipo="elimina", id_evento="e2-9"))
    with patch.object(fallback, "get_llm", return_value=_fake_llm(plan)):
        outcome = fallback.propose_from_request("cancella la riunione", context="")

    assert "Elimina" in outcome.text
    assert repo.get_event("e2-9") is not None               # proposing changes nothing
    with patch.object(resolve.calendar, "delete_event") as delete:
        resolve.accept(outcome.proposal_id, via="test")
    delete.assert_called_once_with("e2-9")
    assert repo.get_event("e2-9") is None


def test_the_move_tool_no_longer_writes_straight_to_google(db: Database):
    """sposta_evento used to call Google directly, with no approval — unlike creation."""
    from donna.agents import tools as toolkit

    _event(db, days_ahead=2, hour=9, end_hour=10, summary="Riunione")
    with patch("donna.google.calendar.calendar_service") as service:
        reply = toolkit.sposta_evento("e2-9", _local(2, 14))
    service.assert_not_called()
    assert reply.startswith("PROPOSTA #")
    assert repo.pending_proposal_count() == 1


# ---------------------------------------------------------------- the orchestrator
def test_a_calendar_change_never_depends_on_the_model_calling_a_tool(db: Database):
    """The agent loop must not run for schedule_mutate when a plan exists. Before, one tool call
    out of two was enough to make the second change disappear."""
    from donna.agents import orchestrator, router

    plan = _plan(
        _action(frase="lava la moto martedì", titolo="Moto", inizio_iso=_local(2, 17)),
        _action(frase="vai in piscina giovedì", titolo="Piscina", inizio_iso=_local(4, 17)),
    )
    with (
        patch.object(router, "route", return_value=router.Route("schedule_mutate", 0.9, "regex")),
        patch.object(fallback, "get_llm", return_value=_fake_llm(plan)),
        patch.object(orchestrator, "run_agent", side_effect=AssertionError("agent loop ran")),
    ):
        result = orchestrator.handle(
            "lava la moto martedì e vai in piscina giovedì", channel="test", chat_id="t", learn=False
        )

    assert repo.pending_proposal_count() == 2
    assert [name for name, _ in result.tool_calls] == ["proponi_evento", "proponi_evento"]
    assert "2 proposte" in result.text


def test_a_question_answered_with_an_existing_proposal_is_not_called_a_lie(db: Database):
    """«Ho preparato una proposta per spostarla» is true when the proposal exists from an earlier
    turn. The claim check knew only about tools called this turn, so a correct answer to «cosa
    ho in calendario?» was replaced by «Stavo per dirti che l'avevo fatto, ma non l'ho fatto»."""
    from donna.agents import orchestrator, router

    class _Result:
        content = "Mercoledì c'è la corsa (ma ho preparato una proposta per spostarla alle 20)."
        tool_calls: list = []
        trace_id = "t1"

    class _Client:
        def chat(self, *a, **k):
            return _Result()

    with (
        patch.object(router, "route", return_value=router.Route("schedule_query", 0.9, "regex")),
        patch.object(base, "get_llm", return_value=_Client()),
    ):
        result = orchestrator.handle("cosa ho mercoledì", channel="test", chat_id="t", learn=False)

    assert "ho preparato una proposta" in result.text
    assert "Stavo per dirti" not in result.text


def test_a_request_to_act_still_gets_the_claim_check(db: Database):
    from donna.agents import orchestrator, router

    class _Result:
        content = "Ho preparato la proposta, è pronta."
        tool_calls: list = []
        trace_id = "t1"

    class _Client:
        def chat(self, *a, **k):
            return _Result()

    with (
        patch.object(router, "route", return_value=router.Route("task_mutate", 0.9, "regex")),
        patch.object(base, "get_llm", return_value=_Client()),
    ):
        result = orchestrator.handle("aggiungi una task", channel="test", chat_id="t", learn=False)

    assert result.text == base.HONEST_FAILURE


def test_when_no_plan_comes_out_the_agent_still_answers(db: Database):
    from donna.agents import orchestrator, router

    reply = base.AgentReply(text="Giovedì sei libero.", agent="schedule")
    with (
        patch.object(router, "route", return_value=router.Route("schedule_mutate", 0.9, "regex")),
        patch.object(fallback, "get_llm", return_value=_fake_llm(_plan(count=0))),
        patch.object(orchestrator, "run_agent", return_value=reply) as agent,
    ):
        result = orchestrator.handle("com'è giovedì?", channel="test", chat_id="t", learn=False)

    agent.assert_called_once()
    assert result.text == "Giovedì sei libero."


# ---------------------------------------------------------------- what "precomputed" means
def test_every_day_a_message_names_gets_its_agenda():
    """«martedì… e giovedì…» is about two days. Stating only the first left the second to be
    guessed, which is precisely what the agenda block exists to prevent."""
    thursday, tuesday = prefetch.referenced_day("giovedì"), prefetch.referenced_day("martedì")
    assert thursday != tuesday
    # Both, in the order the message mentions them.
    assert prefetch.referenced_days("giovedì vado in piscina e martedì lavo la moto") == [thursday, tuesday]


def test_prefetch_says_what_it_computed(db: Database):
    _event(db, days_ahead=1, hour=8, end_hour=16, summary="Lavoro")
    result = prefetch.for_message("domani dopo il lavoro")
    assert result is not None
    assert result.parts and result.parts[0].startswith("agenda di")
    assert "16:30" in result.text


def test_prefetch_is_absent_when_nothing_applies(db: Database):
    assert prefetch.for_message("grazie mille") is None


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
