"""The approval gate: the only path by which anything reaches Google.

The tests that matter here are the concurrency and failure ones. A proposal is acted on from
Telegram, the web UI and the CLI, and a double tap must not produce two calendar events.
"""

from __future__ import annotations

import json
from datetime import timedelta
from unittest.mock import patch

import pytest
from googleapiclient.errors import HttpError

from donna.pipeline import resolve
from donna.store import repo
from donna.store.db import Database
from donna.timeutil import iso_utc, now_utc


def make_proposal(db: Database, **overrides) -> int:
    start = now_utc() + timedelta(days=2)
    payload = {
        "kind": "appuntamento",
        "title": "Dentista",
        "start_ts": iso_utc(start),
        "end_ts": iso_utc(start + timedelta(hours=1)),
        "all_day": False,
        "location": "via Verdi 12",
    }
    payload.update(overrides.pop("payload", {}))
    kwargs = dict(
        kind="calendar_event",
        source_type="email",
        source_id="m1",
        payload=payload,
        reasoning="Studio indica giovedì alle 15:00",
        evidence_quote="l'appuntamento è giovedì alle 15:00",
        confidence=0.9,
        trace_id="trace-1",
        dedupe_key=None,
    )
    kwargs.update(overrides)
    proposal_id = repo.create_proposal(**kwargs)
    assert proposal_id is not None
    return proposal_id


class _Created(dict):
    """Minimal stand-in for a Google Calendar insert response."""


def fake_create(event_id: str = "evt-1"):
    return _Created({"id": event_id, "htmlLink": f"https://calendar.google.com/{event_id}"})


def _http_error(status: int = 500) -> HttpError:
    class _Resp:
        def __init__(self) -> None:
            self.status = status
            self.reason = "boom"

    return HttpError(_Resp(), b"{}", uri="https://example.test")


# ---------------------------------------------------------------- accept
def test_accept_creates_the_event_and_closes_the_proposal(db: Database):
    proposal_id = make_proposal(db)
    with patch(
        "donna.pipeline.resolve.calendar.create_event", return_value=fake_create()
    ) as create:
        result = resolve.accept(proposal_id)

    create.assert_called_once()
    assert result.state == "accepted"
    assert result.result_ref == "evt-1"
    row = repo.get_proposal(proposal_id)
    assert row["state"] == "accepted"
    assert row["result_ref"] == "evt-1"
    assert row["resolved_via"] == "cli"


def test_accept_records_provenance_in_the_mirror_immediately(db: Database):
    """Without this the link is lost within seconds.

    The first implementation left the mirror to the next calendar sync, which inserts what
    Google reports — and Google does not report "Donna created this from proposal N". The
    answer to "why is this on my calendar?" disappeared 30 seconds after creating it.
    """
    proposal_id = make_proposal(db)
    with patch("donna.pipeline.resolve.calendar.create_event", return_value=fake_create()):
        resolve.accept(proposal_id)

    event = db.query_one("SELECT * FROM events WHERE id = 'evt-1'")
    assert event is not None
    assert event["source"] == "donna"
    assert event["origin_proposal_id"] == proposal_id


def test_a_later_sync_does_not_erase_that_provenance(db: Database):
    proposal_id = make_proposal(db)
    with patch("donna.pipeline.resolve.calendar.create_event", return_value=fake_create()):
        resolve.accept(proposal_id)

    payload = repo.proposal_payload(repo.get_proposal(proposal_id))
    # Google reports the same event back, knowing nothing about Donna.
    repo.replace_events_in_window(
        [
            {
                "id": "evt-1",
                "calendar_id": "primary",
                "summary": "Dentista",
                "description": "",
                "location": "",
                "start_ts": payload["start_ts"],
                "end_ts": payload["end_ts"],
                "start_raw": payload["start_ts"],
                "end_raw": payload["end_ts"],
                "all_day": 0,
                "status": "confirmed",
                "organizer": "",
                "attendees": "[]",
                "html_link": "",
                "recurring_event_id": None,
                "updated_at": None,
            }
        ],
        payload["start_ts"],
        payload["end_ts"],
    )

    event = db.query_one("SELECT * FROM events WHERE id = 'evt-1'")
    assert event["source"] == "donna", "sync overwrote provenance it does not own"
    assert event["origin_proposal_id"] == proposal_id


def test_accepting_twice_creates_one_event(db: Database):
    # The double-tap case: Telegram buttons are easy to press twice.
    proposal_id = make_proposal(db)
    with patch(
        "donna.pipeline.resolve.calendar.create_event", return_value=fake_create()
    ) as create:
        first = resolve.accept(proposal_id)
        second = resolve.accept(proposal_id)

    assert create.call_count == 1
    assert first.state == "accepted"
    assert "già gestita" in second.message


def test_accept_records_feedback(db: Database):
    proposal_id = make_proposal(db)
    with patch("donna.pipeline.resolve.calendar.create_event", return_value=fake_create()):
        resolve.accept(proposal_id)
    assert repo.feedback_counts() == {"proposal_accept": 1}


def test_a_failed_google_write_reopens_the_proposal(db: Database):
    """The proposal is claimed before the write, so a failure has to release the claim.

    Otherwise it sits in `accepted` with no event to show for it, and the user cannot retry.
    """
    proposal_id = make_proposal(db)
    with patch("donna.pipeline.resolve.calendar.create_event", side_effect=_http_error()):
        with pytest.raises(resolve.ProposalError):
            resolve.accept(proposal_id)

    row = repo.get_proposal(proposal_id)
    assert row["state"] == "pending"
    assert row["result_ref"] is None


def test_a_reopened_proposal_can_be_accepted_afterwards(db: Database):
    proposal_id = make_proposal(db)
    with patch("donna.pipeline.resolve.calendar.create_event", side_effect=_http_error()):
        with pytest.raises(resolve.ProposalError):
            resolve.accept(proposal_id)
    with patch("donna.pipeline.resolve.calendar.create_event", return_value=fake_create()):
        result = resolve.accept(proposal_id)
    assert result.state == "accepted"


def test_accept_of_an_unknown_proposal_is_an_error(db: Database):
    with pytest.raises(resolve.ProposalError, match="non trovata"):
        resolve.accept(9999)


def test_a_proposal_without_a_time_window_is_refused(db: Database):
    proposal_id = make_proposal(db, payload={"start_ts": None, "end_ts": None})
    with patch("donna.pipeline.resolve.calendar.create_event", return_value=fake_create()):
        with pytest.raises(resolve.ProposalError, match="intervallo"):
            resolve.accept(proposal_id)
    # And it is released, not stranded in accepted.
    assert repo.get_proposal(proposal_id)["state"] == "pending"


# ---------------------------------------------------------------- reject
def test_reject_closes_the_proposal_and_writes_nothing_to_google(db: Database):
    proposal_id = make_proposal(db)
    with patch("donna.pipeline.resolve.calendar.create_event") as create:
        result = resolve.reject(proposal_id, note="non mi serve")
    create.assert_not_called()
    assert result.state == "rejected"
    assert repo.get_proposal(proposal_id)["state"] == "rejected"


def test_rejection_keeps_the_note_as_training_signal(db: Database):
    proposal_id = make_proposal(db)
    resolve.reject(proposal_id, note="il bollo lo pago sempre a fine mese")
    row = db.query_one("SELECT * FROM feedback WHERE kind = 'proposal_reject'")
    assert row["note"] == "il bollo lo pago sempre a fine mese"
    assert row["proposal_id"] == proposal_id
    assert row["email_id"] == "m1"


def test_rejecting_twice_is_harmless(db: Database):
    proposal_id = make_proposal(db)
    resolve.reject(proposal_id)
    second = resolve.reject(proposal_id)
    assert "già gestita" in second.message
    assert db.scalar("SELECT count(*) FROM feedback") == 1


# ---------------------------------------------------------------- edit
def test_edit_and_accept_applies_the_correction(db: Database):
    proposal_id = make_proposal(db)
    new_start = iso_utc(now_utc() + timedelta(days=3))
    with patch(
        "donna.pipeline.resolve.calendar.create_event", return_value=fake_create()
    ) as create:
        resolve.edit_and_accept(proposal_id, title="Igiene dentale", start_ts=new_start)

    # The corrected values are what reach Google.
    assert create.call_args[0][0] == "Igiene dentale"
    assert create.call_args[0][1] == new_start


def test_edit_records_both_the_original_and_the_correction(db: Database):
    proposal_id = make_proposal(db)
    with patch("donna.pipeline.resolve.calendar.create_event", return_value=fake_create()):
        resolve.edit_and_accept(proposal_id, title="Igiene dentale")

    row = db.query_one("SELECT * FROM feedback WHERE kind = 'proposal_edit'")
    assert row is not None
    assert json.loads(row["original_output"])["title"] == "Dentista"
    assert json.loads(row["corrected_output"])["title"] == "Igiene dentale"


# ---------------------------------------------------------------- expiry
def test_unanswered_proposals_expire(db: Database):
    proposal_id = make_proposal(db)
    db.execute(
        "UPDATE proposals SET created_at = ? WHERE id = ?",
        (iso_utc(now_utc() - timedelta(days=30)), proposal_id),
    )
    assert resolve.expire_stale(days=7) == 1
    assert repo.get_proposal(proposal_id)["state"] == "expired"


def test_recent_proposals_do_not_expire(db: Database):
    proposal_id = make_proposal(db)
    assert resolve.expire_stale(days=7) == 0
    assert repo.get_proposal(proposal_id)["state"] == "pending"


def test_an_expired_proposal_cannot_be_accepted(db: Database):
    proposal_id = make_proposal(db)
    db.execute(
        "UPDATE proposals SET created_at = ? WHERE id = ?",
        (iso_utc(now_utc() - timedelta(days=30)), proposal_id),
    )
    resolve.expire_stale(days=7)
    with patch("donna.pipeline.resolve.calendar.create_event") as create:
        result = resolve.accept(proposal_id)
    create.assert_not_called()
    assert "già gestita" in result.message


# ---------------------------------------------------------------- description
def test_the_calendar_entry_explains_itself(db: Database):
    """Six months on, "why is this on my calendar?" should be answerable from the entry."""
    proposal_id = make_proposal(db)
    with patch(
        "donna.pipeline.resolve.calendar.create_event", return_value=fake_create()
    ) as create:
        resolve.accept(proposal_id)

    description = create.call_args.kwargs["description"]
    assert "Donna" in description
    assert "giovedì alle 15:00" in description  # the reasoning
    assert "l'appuntamento è giovedì" in description  # the evidence quote
    assert "mail.google.com" in description  # a link back to the source


def test_describe_renders_an_all_day_proposal_without_a_time(db: Database):
    start = now_utc() + timedelta(days=3)
    proposal_id = make_proposal(
        db,
        payload={
            "all_day": True,
            "start_ts": iso_utc(start.replace(hour=0, minute=0)),
            "end_ts": iso_utc(start.replace(hour=0, minute=0) + timedelta(days=1)),
            "title": "Pagamento bollo",
        },
    )
    text = resolve.describe(repo.get_proposal(proposal_id))
    assert "tutto il giorno" in text
    assert "Pagamento bollo" in text
