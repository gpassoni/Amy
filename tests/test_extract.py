"""Extraction decisions, with the model mocked.

What is tested here is the judgement *around* the model call — whether to propose at all —
because that is where the bugs were. The model's own accuracy is measured by the eval
harness (`python -m donna.eval.cli run`), which needs a real model and is too slow for here.
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import pytest

from donna.pipeline import extract
from donna.pipeline.dates import ResolvedWhen
from donna.pipeline.schemas import Commitment
from donna.store import repo
from donna.store.db import Database
from donna.timeutil import iso_utc, now_utc


def make_commitment(**kwargs) -> Commitment:
    defaults = dict(
        evidence="l'appuntamento è giovedì alle 15:00",
        date_phrase="giovedì alle 15:00",
        start_iso=None,
        location=None,
        kind="appuntamento",
        title="Dentista",
        all_day=False,
        has_commitment=True,
        confidence=0.9,
    )
    defaults.update(kwargs)
    return Commitment(**defaults)


def fake_structured(commitment: Commitment):
    """Stand in for llm.structured(), returning a fixed Commitment."""

    class _Result:
        value = commitment
        model = "qwen3.5:9b"
        trace_id = "trace-1"
        latency_ms = 100
        attempts = 1

    class _Client:
        def structured(self, *args, **kwargs):
            return _Result()

    return _Client()


def run_extract(commitment: Commitment, *, received_at: str | None = None):
    received = received_at or iso_utc(now_utc() - timedelta(hours=1))
    with patch("donna.pipeline.extract.get_llm", return_value=fake_structured(commitment)):
        return extract.extract_one(
            sender_name="Studio",
            sender_addr="studio@example.it",
            subject="Conferma",
            body="Le confermiamo l'appuntamento.",
            received_at=received,
        )


# ---------------------------------------------------------------- the verdict
def test_no_commitment_is_skipped_without_a_date_lookup(db: Database):
    result = run_extract(make_commitment(has_commitment=False, kind="nessuno"))
    assert not result.proposable
    assert result.skip_reason == extract.SKIP_NO_COMMITMENT


def test_kind_nessuno_counts_as_no_commitment(db: Database):
    # The two fields can disagree; either one saying "nothing" means nothing.
    result = run_extract(make_commitment(has_commitment=True, kind="nessuno"))
    assert result.skip_reason == extract.SKIP_NO_COMMITMENT


def test_unresolvable_phrase_is_skipped_rather_than_guessed(db: Database):
    result = run_extract(make_commitment(date_phrase="appena possibile", start_iso=None))
    assert not result.proposable
    assert result.skip_reason == extract.SKIP_NO_DATE


def test_a_resolvable_phrase_produces_a_proposable_extraction(db: Database):
    result = run_extract(make_commitment(date_phrase="domani alle 15:00"))
    assert result.proposable
    assert result.when is not None


# ---------------------------------------------------------------- the past-date guard
def test_a_commitment_that_has_already_passed_is_not_proposed(db: Database):
    """The bug this guards, seen on the real mailbox.

    A year-old email announcing a renewal on 15 November 2025 resolves correctly to
    15 November 2025 — and a calendar entry ten months in the past is noise. All three of
    the first proposals ever generated were of this kind.
    """
    old = iso_utc(now_utc() - timedelta(days=300))
    result = run_extract(make_commitment(date_phrase="domani alle 15:00"), received_at=old)
    assert not result.proposable
    assert result.skip_reason == extract.SKIP_PAST


def test_plausibility_is_judged_against_the_email_but_actionability_against_now(db: Database):
    # Resolution still succeeds — the date is only rejected for being spent, not for being
    # implausible, and the resolved window is kept so the UI can explain the silence.
    old = iso_utc(now_utc() - timedelta(days=300))
    result = run_extract(make_commitment(date_phrase="domani alle 15:00"), received_at=old)
    assert result.when is not None


def test_a_commitment_later_today_is_still_proposed(db: Database):
    soon = now_utc() + timedelta(hours=3)
    result = run_extract(
        make_commitment(date_phrase=f"oggi alle {soon.astimezone().hour}:00"),
        received_at=iso_utc(now_utc() - timedelta(minutes=30)),
    )
    assert result.skip_reason != extract.SKIP_PAST


# ---------------------------------------------------------------- confidence
def test_agreement_between_phrase_and_model_raises_confidence():
    when = ResolvedWhen("x", "y", False, source="agreed", agreed=True)
    assert extract._effective_confidence(make_commitment(confidence=0.8), when) == pytest.approx(0.9)


def test_a_model_only_date_is_penalised():
    # No phrase to verify against is the configuration most likely to be hallucinated.
    when = ResolvedWhen("x", "y", False, source="model", agreed=False)
    assert extract._effective_confidence(make_commitment(confidence=0.8), when) == pytest.approx(0.56)


def test_a_phrase_derived_date_is_taken_at_face_value():
    when = ResolvedWhen("x", "y", False, source="phrase", agreed=False)
    assert extract._effective_confidence(make_commitment(confidence=0.8), when) == pytest.approx(0.8)


def test_low_confidence_does_not_reach_the_user(db: Database, monkeypatch):
    monkeypatch.setenv("PROPOSAL_CONFIDENCE_FLOOR", "0.9")
    from donna.config import get_settings

    get_settings.cache_clear()
    result = run_extract(make_commitment(date_phrase="domani alle 15:00", confidence=0.3))
    assert result.skip_reason == extract.SKIP_LOW_CONFIDENCE
    get_settings.cache_clear()


# ---------------------------------------------------------------- proposing
def _proposable(db: Database, **kwargs):
    return run_extract(make_commitment(date_phrase="domani alle 15:00", **kwargs))


def test_propose_creates_a_row_with_its_reasoning_and_evidence(db: Database):
    result = _proposable(db)
    proposal_id = extract.propose(
        email_id="m1", subject="Conferma", sender_name="Studio",
        sender_addr="studio@example.it", extraction=result,
    )
    assert proposal_id is not None
    row = repo.get_proposal(proposal_id)
    assert row["state"] == "pending"
    assert row["kind"] == "calendar_event"
    assert row["evidence_quote"]
    assert "Studio" in row["reasoning"]
    payload = repo.proposal_payload(row)
    assert payload["title"] == "Dentista"
    assert payload["start_ts"]


def test_the_same_email_cannot_produce_two_proposals(db: Database):
    result = _proposable(db)
    first = extract.propose(
        email_id="m1", subject="Conferma", sender_name="S", sender_addr="s@x.it",
        extraction=result,
    )
    second = extract.propose(
        email_id="m1", subject="Conferma", sender_name="S", sender_addr="s@x.it",
        extraction=result,
    )
    assert first is not None
    assert second is None, "being asked twice about one email is worse than being asked late"


def test_nothing_is_proposed_when_the_event_is_already_on_the_calendar(db: Database):
    result = _proposable(db)
    assert result.when is not None

    repo.replace_events_in_window(
        [
            {
                "id": "e1", "calendar_id": "primary", "summary": "Dentista igiene",
                "description": "", "location": "", "start_ts": result.when.start_ts,
                "end_ts": result.when.end_ts, "start_raw": result.when.start_ts,
                "end_raw": result.when.end_ts, "all_day": 0, "status": "confirmed",
                "organizer": "", "attendees": "[]", "html_link": "",
                "recurring_event_id": None, "updated_at": None,
            }
        ],
        result.when.start_ts,
        result.when.end_ts,
    )

    assert extract.propose(
        email_id="m1", subject="Conferma", sender_name="S", sender_addr="s@x.it",
        extraction=result,
    ) is None


def test_an_unrelated_event_at_the_same_time_does_not_block_a_proposal(db: Database):
    result = _proposable(db)
    assert result.when is not None
    repo.replace_events_in_window(
        [
            {
                "id": "e1", "calendar_id": "primary", "summary": "Partita di calcio",
                "description": "", "location": "", "start_ts": result.when.start_ts,
                "end_ts": result.when.end_ts, "start_raw": result.when.start_ts,
                "end_raw": result.when.end_ts, "all_day": 0, "status": "confirmed",
                "organizer": "", "attendees": "[]", "html_link": "",
                "recurring_event_id": None, "updated_at": None,
            }
        ],
        result.when.start_ts,
        result.when.end_ts,
    )
    assert extract.propose(
        email_id="m1", subject="Conferma", sender_name="S", sender_addr="s@x.it",
        extraction=result,
    ) is not None


def test_a_missing_title_falls_back_to_the_subject(db: Database):
    result = _proposable(db, title=None)
    proposal_id = extract.propose(
        email_id="m1", subject="Conferma appuntamento", sender_name="S",
        sender_addr="s@x.it", extraction=result,
    )
    payload = repo.proposal_payload(repo.get_proposal(proposal_id))
    assert payload["title"] == "Conferma appuntamento"


def test_reasoning_names_the_sender_and_the_phrase(db: Database):
    result = _proposable(db)
    proposal_id = extract.propose(
        email_id="m1", subject="x", sender_name="Studio Bianchi",
        sender_addr="s@x.it", extraction=result,
    )
    reasoning = repo.get_proposal(proposal_id)["reasoning"]
    # Assembled from known facts, never generated — so it cannot describe a different email.
    assert "Studio Bianchi" in reasoning
    assert "domani alle 15:00" in reasoning
