"""Dashboard: every page renders, every action does what it says.

These are smoke-and-contract tests rather than markup assertions. A template that raises is a
500 the moment you open the page, and a Jinja typo is exactly the kind of thing that only
shows up in the browser — so the point is that every route renders against a real (if small)
database.
"""
from __future__ import annotations

import json
from datetime import timedelta
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from donna.store import activity, repo
from donna.store.db import Database
from donna.timeutil import iso_utc, now_local, now_utc, to_utc


@pytest.fixture()
def client(db: Database) -> TestClient:
    from donna.interfaces.web.app import app

    return TestClient(app)


@pytest.fixture()
def proposal(db: Database) -> int:
    start = to_utc(now_local().replace(hour=15, minute=0, second=0, microsecond=0) + timedelta(days=2))
    now = iso_utc(now_utc())
    repo.upsert_email(
        id="m1", thread_id="t1", from_addr="studio@x.it", from_name="Studio Bianchi",
        to_addrs="me", subject="Conferma appuntamento", snippet="",
        body="le confermiamo l'appuntamento di giovedì alle 15:00", received_at=now,
        label_ids=[], is_unread=True,
    )
    proposal_id = repo.create_proposal(
        kind="calendar_event", source_type="email", source_id="m1",
        payload={
            "title": "Igiene dentale",
            "start_ts": iso_utc(start),
            "end_ts": iso_utc(start + timedelta(hours=1)),
            "all_day": False,
            "location": "via Verdi 12",
            "date_phrase": "giovedì alle 15:00",
            "date_source": "phrase",
        },
        reasoning="Studio Bianchi indica giovedì alle 15:00",
        evidence_quote="le confermiamo l'appuntamento di giovedì alle 15:00",
        confidence=0.95, trace_id="t1",
    )
    return proposal_id


# ---------------------------------------------------------------- pages render
@pytest.mark.parametrize(
    "path",
    ["/", "/agents", "/proposals", "/inbox", "/chat", "/memory", "/training", "/traces",
     "/health", "/api/activity", "/activity/live"],
)
def test_every_page_renders(client: TestClient, proposal: int, path: str):
    response = client.get(path)
    assert response.status_code == 200, f"{path} -> {response.status_code}"


def test_pages_render_on_an_empty_database(client: TestClient):
    # First run: no email, no events, no proposals. Empty states must not crash.
    for path in ("/", "/agents", "/proposals", "/inbox", "/training", "/memory"):
        assert client.get(path).status_code == 200


def test_a_missing_activity_is_a_page_not_a_crash(client: TestClient):
    response = client.get("/activity/does-not-exist")
    assert response.status_code == 200
    assert "Non trovato" in response.text


def test_a_missing_trace_is_a_page_not_a_crash(client: TestClient):
    assert client.get("/traces/nope").status_code == 200


# ---------------------------------------------------------------- what the pages say
def test_the_proposal_page_shows_the_evidence(client: TestClient, proposal: int):
    body = client.get("/proposals").text
    assert "Igiene dentale" in body
    assert "le confermiamo l&#39;appuntamento" in body or "le confermiamo l'appuntamento" in body
    assert "Studio Bianchi" in body


def test_the_proposal_page_offers_an_edit_form(client: TestClient, proposal: int):
    body = client.get("/proposals").text
    assert 'name="start_local"' in body
    assert 'type="datetime-local"' in body


def test_the_edit_form_is_prefilled_in_local_time(client: TestClient, proposal: int):
    """A UTC value in the input would show the user a time offset from the one they are
    correcting — and the time is the field most likely to need correcting."""
    body = client.get("/proposals").text
    # The fixture pins the start to 15:00 local. Europe/Rome is UTC+1 or +2, so a UTC value
    # would render as 13:00 or 14:00 here.
    assert "T15:00" in body
    assert "T13:00" not in body and "T14:00" not in body


def test_the_agents_page_lists_the_roster(client: TestClient):
    body = client.get("/agents").text
    for name in ("schedule", "inbox", "tasks", "proposals", "briefing", "chat"):
        assert name in body
    assert "elenca_eventi" in body


def test_the_live_fragment_shows_running_work(client: TestClient, db: Database):
    with activity.record(activity.TURN, "inbox", summary="sto lavorando"):
        body = client.get("/activity/live").text
        assert "inbox" in body
        assert "sto lavorando" in body


def test_the_live_fragment_says_so_when_idle(client: TestClient):
    assert "In attesa" in client.get("/activity/live").text


# ---------------------------------------------------------------- actions
def test_accept_creates_the_event_and_clears_the_proposal(client: TestClient, proposal: int):
    with patch(
        "donna.pipeline.resolve.calendar.create_event",
        return_value={"id": "evt-1", "htmlLink": "https://x"},
    ) as create:
        response = client.post(f"/proposals/{proposal}/accept", follow_redirects=False)

    assert response.status_code == 303
    create.assert_called_once()
    assert repo.get_proposal(proposal)["state"] == "accepted"


def test_reject_keeps_the_reason(client: TestClient, proposal: int, db: Database):
    client.post(f"/proposals/{proposal}/reject", data={"note": "lo pago sempre a fine mese"},
                follow_redirects=False)
    assert repo.get_proposal(proposal)["state"] == "rejected"
    row = db.query_one("SELECT * FROM feedback WHERE kind = 'proposal_reject'")
    assert row["note"] == "lo pago sempre a fine mese"


def test_edit_records_both_versions_and_applies_the_correction(
    client: TestClient, proposal: int, db: Database
):
    """The interaction the whole dashboard is built around.

    An accept says "this was fine" and a reject says "this was not". Only an edit says what
    the right answer was, which is the difference between a preference signal and something a
    fine-tune can be supervised on.
    """
    with patch(
        "donna.pipeline.resolve.calendar.create_event",
        return_value={"id": "evt-1", "htmlLink": "https://x"},
    ) as create:
        client.post(
            f"/proposals/{proposal}/edit",
            data={
                "title": "Igiene dentale — studio Bianchi",
                "start_local": "2026-12-01T09:30",
                "end_local": "2026-12-01T10:15",
                "location": "via Verdi 12",
            },
            follow_redirects=False,
        )

    assert create.call_args[0][0] == "Igiene dentale — studio Bianchi"

    row = db.query_one("SELECT * FROM feedback WHERE kind = 'proposal_edit'")
    assert json.loads(row["original_output"])["title"] == "Igiene dentale"
    assert json.loads(row["corrected_output"])["title"] == "Igiene dentale — studio Bianchi"


def test_an_edit_with_an_unparseable_date_changes_nothing(client: TestClient, proposal: int):
    with patch("donna.pipeline.resolve.calendar.create_event") as create:
        client.post(
            f"/proposals/{proposal}/edit",
            data={"title": "X", "start_local": "not a date"},
            follow_redirects=False,
        )
    create.assert_not_called()
    assert repo.get_proposal(proposal)["state"] == "pending"


def test_reclassifying_from_the_inbox_records_feedback(client: TestClient, proposal: int, db: Database):
    repo.set_email_category(
        "m1", category="inutile", confidence=0.9, reason="x", model="m", trace_id="t",
        signal="promozione",
    )
    with patch("donna.pipeline.triage.gmail.apply_category_label", return_value=True):
        client.post("/inbox/m1/reclassify", data={"category": "importante"},
                    follow_redirects=False)

    assert repo.get_email("m1")["category"] == "importante"
    row = db.query_one("SELECT * FROM feedback WHERE kind = 'reclassify'")
    assert row["original_output"] == "inutile"
    assert row["corrected_output"] == "importante"


def test_memory_can_be_added_and_forgotten(client: TestClient, db: Database):
    from donna.context import memory

    with patch.object(memory, "get_llm") as llm:
        llm.return_value.embed_one.return_value = [0.1] * 8
        client.post("/memory/add", data={"text": "vado in palestra il martedì"},
                    follow_redirects=False)

    facts = memory.all_facts()
    assert len(facts) == 1
    client.post(f"/memory/{facts[0]['id']}/forget", follow_redirects=False)
    assert memory.all_facts() == []


# ---------------------------------------------------------------- api
def test_the_activity_api_reports_running_work(client: TestClient, db: Database):
    with activity.record(activity.CYCLE, "pipeline", summary="sync"):
        payload = client.get("/api/activity").json()
        assert len(payload["running"]) == 1
        assert payload["running"][0]["actor"] == "pipeline"


def test_health_reports_the_mirror(client: TestClient, proposal: int):
    payload = client.get("/health").json()
    assert payload["ok"] is True
    assert payload["pending_proposals"] == 1
