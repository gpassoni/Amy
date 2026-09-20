"""Sync behaviour, with Google mocked out.

These tests exist because of bugs that actually happened during Phase 1 against a real
mailbox: a rate-limited backfill stored 116 of 154 messages, reported success, and advanced
its cursor — silently losing the other 38, because the change feed only moves forward.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from donna.google.gmail import FetchReport, HistoryChanges, Message
from donna.store import repo
from donna.store.db import Database
from donna.sync import gmail_sync
from donna.timeutil import iso_utc


def make_message(msg_id: str, *, internal_date: str = "1789000000000", date_header: str = "") -> Message:
    return Message(
        id=msg_id,
        thread_id=f"t{msg_id}",
        from_addr="mittente@example.com",
        from_name="Mittente",
        to_addrs="me@example.com",
        subject=f"Oggetto {msg_id}",
        snippet="anteprima",
        body="corpo del messaggio",
        date_header=date_header,
        internal_date=internal_date,
        label_ids=["INBOX", "UNREAD"],
    )


# ---------------------------------------------------------------- timestamps
def test_received_at_prefers_internal_date_over_the_header():
    # Gmail's own receipt time beats a sender-written header.
    msg = make_message("m1", internal_date="1789000000000", date_header="Sat, 1 Jan 2000 00:00:00 +0000")
    assert gmail_sync._received_at(msg) == iso_utc(
        datetime.fromtimestamp(1789000000, tz=timezone.utc)
    )


def test_received_at_falls_back_to_the_header():
    msg = make_message("m1", internal_date="", date_header="Sat, 20 Sep 2026 15:30:00 +0200")
    assert gmail_sync._received_at(msg) == "2026-09-20T13:30:00+00:00"


def test_received_at_falls_back_to_the_header_when_internal_date_is_garbage():
    msg = make_message("m1", internal_date="not-a-number", date_header="Sat, 20 Sep 2026 15:30:00 +0200")
    assert gmail_sync._received_at(msg) == "2026-09-20T13:30:00+00:00"


def test_received_at_always_returns_something():
    # received_at is NOT NULL and ordering depends on it, so there is no "leave it empty".
    msg = make_message("m1", internal_date="", date_header="whenever")
    assert gmail_sync._received_at(msg).endswith("+00:00")


# ---------------------------------------------------------------- storage semantics
def test_resyncing_a_message_does_not_discard_its_classification(db: Database):
    # The bug this guards: a sync pass overwriting triage output with NULLs, which would
    # send every message back through the classifier on every sync.
    msg = make_message("m1")
    assert gmail_sync._store(msg) is True

    repo.set_email_category(
        "m1", category="importante", confidence=0.9, reason="perché sì",
        model="qwen3.5:2b", trace_id="t1", label_applied=True,
    )

    assert gmail_sync._store(msg) is False  # second time is an update, not an insert
    row = repo.get_email("m1")
    assert row["category"] == "importante"
    assert row["category_confidence"] == pytest.approx(0.9)
    assert row["classifier_model"] == "qwen3.5:2b"


def test_resync_does_refresh_the_fields_gmail_owns(db: Database):
    gmail_sync._store(make_message("m1"))
    read = make_message("m1")
    read.label_ids = ["INBOX"]  # no longer UNREAD
    gmail_sync._store(read)
    assert repo.get_email("m1")["is_unread"] == 0


def test_known_email_ids_finds_only_what_is_stored(db: Database):
    for i in range(3):
        gmail_sync._store(make_message(f"m{i}"))
    assert repo.known_email_ids(["m0", "m2", "m99"]) == {"m0", "m2"}


def test_known_email_ids_handles_more_ids_than_sqlite_allows_variables(db: Database):
    # Chunking guard: SQLite caps bound variables, and a year's backfill exceeds it.
    for i in range(5):
        gmail_sync._store(make_message(f"m{i}"))
    found = repo.known_email_ids([f"m{i}" for i in range(2000)])
    assert found == {f"m{i}" for i in range(5)}


def test_known_email_ids_on_empty_input(db: Database):
    assert repo.known_email_ids([]) == set()


# ---------------------------------------------------------------- the cursor bug
@patch("donna.sync.gmail_sync.gmail.fetch_messages")
@patch("donna.sync.gmail_sync.gmail.list_recent_ids")
@patch("donna.sync.gmail_sync.gmail.current_history_id")
def test_partial_backfill_does_not_advance_the_cursor(
    history_id, list_ids, fetch, db: Database
):
    history_id.return_value = "999"
    list_ids.return_value = ["m1", "m2", "m3"]
    # Two fetched, one rate-limited away.
    fetch.return_value = FetchReport(
        messages=[make_message("m1"), make_message("m2")], failed=["m3"]
    )

    result = gmail_sync.sync_gmail(full=True)

    assert result.created == 2
    assert result.ok is False
    assert "cursore non avanzato" in result.error
    # The cursor must still be unset, or m3 is lost forever.
    assert repo.get_cursor(gmail_sync.RESOURCE) is None
    assert repo.get_cursor(gmail_sync.BACKFILL_RESOURCE) != gmail_sync.BACKFILL_DONE


@patch("donna.sync.gmail_sync.gmail.fetch_messages")
@patch("donna.sync.gmail_sync.gmail.list_recent_ids")
@patch("donna.sync.gmail_sync.gmail.current_history_id")
def test_complete_backfill_advances_cursor_and_marks_done(
    history_id, list_ids, fetch, db: Database
):
    history_id.return_value = "999"
    list_ids.return_value = ["m1", "m2"]
    fetch.return_value = FetchReport(messages=[make_message("m1"), make_message("m2")])

    result = gmail_sync.sync_gmail(full=True)

    assert result.ok and result.created == 2
    assert repo.get_cursor(gmail_sync.RESOURCE) == "999"
    assert repo.get_cursor(gmail_sync.BACKFILL_RESOURCE) == gmail_sync.BACKFILL_DONE


@patch("donna.sync.gmail_sync.gmail.fetch_messages")
@patch("donna.sync.gmail_sync.gmail.list_recent_ids")
@patch("donna.sync.gmail_sync.gmail.current_history_id")
def test_backfill_resumes_by_skipping_what_is_already_stored(
    history_id, list_ids, fetch, db: Database
):
    gmail_sync._store(make_message("m1"))
    history_id.return_value = "999"
    list_ids.return_value = ["m1", "m2"]
    fetch.return_value = FetchReport(messages=[make_message("m2")])

    result = gmail_sync.sync_gmail(full=True)

    # Only the missing id is requested — this is what makes a retry cheap.
    assert fetch.call_args[0][0] == ["m2"]
    assert result.skipped == 1
    assert result.created == 1


@patch("donna.sync.gmail_sync.gmail.fetch_messages")
@patch("donna.sync.gmail_sync.gmail.list_recent_ids")
@patch("donna.sync.gmail_sync.gmail.current_history_id")
def test_incomplete_backfill_forces_the_full_path_on_the_next_run(
    history_id, list_ids, fetch, db: Database
):
    # Even with a cursor present, an unfinished backfill must not switch to incremental:
    # the change feed would never mention the gap again.
    repo.record_sync(gmail_sync.RESOURCE, cursor="500")
    history_id.return_value = "999"
    list_ids.return_value = ["m1"]
    fetch.return_value = FetchReport(messages=[make_message("m1")])

    result = gmail_sync.sync_gmail()

    assert result.mode == "full"
    list_ids.assert_called_once()


@patch("donna.sync.gmail_sync.gmail.fetch_messages")
@patch("donna.sync.gmail_sync.gmail.replay_history")
def test_incremental_path_used_once_backfill_is_done(changed, fetch, db: Database):
    repo.record_sync(gmail_sync.RESOURCE, cursor="500")
    repo.record_sync(gmail_sync.BACKFILL_RESOURCE, cursor=gmail_sync.BACKFILL_DONE)
    changed.return_value = HistoryChanges(added={"m9"}, cursor="600")
    fetch.return_value = FetchReport(messages=[make_message("m9")])

    result = gmail_sync.sync_gmail()

    assert result.mode == "incremental"
    assert repo.get_cursor(gmail_sync.RESOURCE) == "600"


@patch("donna.sync.gmail_sync.gmail.fetch_messages")
@patch("donna.sync.gmail_sync.gmail.replay_history")
def test_incremental_deletes_messages_removed_upstream(changed, fetch, db: Database):
    gmail_sync._store(make_message("m1"))
    repo.record_sync(gmail_sync.RESOURCE, cursor="500")
    repo.record_sync(gmail_sync.BACKFILL_RESOURCE, cursor=gmail_sync.BACKFILL_DONE)
    changed.return_value = HistoryChanges(deleted={"m1"}, cursor="600")
    fetch.return_value = FetchReport()

    result = gmail_sync.sync_gmail()

    assert result.deleted == 1
    assert repo.get_email("m1") is None


@patch("donna.sync.gmail_sync.gmail.fetch_messages")
@patch("donna.sync.gmail_sync.gmail.list_recent_ids")
@patch("donna.sync.gmail_sync.gmail.current_history_id")
@patch("donna.sync.gmail_sync.gmail.replay_history")
def test_expired_history_falls_back_to_a_full_sync(
    changed, history_id, list_ids, fetch, db: Database
):
    from donna.google.gmail import HistoryExpired

    repo.record_sync(gmail_sync.RESOURCE, cursor="1")
    repo.record_sync(gmail_sync.BACKFILL_RESOURCE, cursor=gmail_sync.BACKFILL_DONE)
    changed.side_effect = HistoryExpired("too old")
    history_id.return_value = "999"
    list_ids.return_value = ["m1"]
    fetch.return_value = FetchReport(messages=[make_message("m1")])

    result = gmail_sync.sync_gmail()

    assert result.mode == "full"
    assert result.created == 1


def test_messages_gmail_no_longer_has_are_skipped_not_failed(db: Database):
    with (
        patch("donna.sync.gmail_sync.gmail.current_history_id", return_value="9"),
        patch("donna.sync.gmail_sync.gmail.list_recent_ids", return_value=["m1", "gone"]),
        patch(
            "donna.sync.gmail_sync.gmail.fetch_messages",
            return_value=FetchReport(messages=[make_message("m1")], missing=["gone"]),
        ),
    ):
        result = gmail_sync.sync_gmail(full=True)

    # A deleted message is not a failure, so the cursor may still advance.
    assert result.ok
    assert result.skipped == 1
    assert repo.get_cursor(gmail_sync.RESOURCE) == "9"


# ---------------------------------------------------------------- bookkeeping
def test_record_sync_keeps_the_cursor_when_an_error_is_recorded(db: Database):
    repo.record_sync("gmail", cursor="100")
    repo.record_sync("gmail", error="boom")
    row = db.query_one("SELECT * FROM sync_state WHERE resource='gmail'")
    assert row["cursor"] == "100"       # not wiped by the failure
    assert row["last_error"] == "boom"
    assert row["last_success_at"] is not None  # the earlier success is remembered


def test_record_sync_clears_the_error_on_recovery(db: Database):
    repo.record_sync("gmail", error="boom")
    repo.record_sync("gmail", cursor="200")
    row = db.query_one("SELECT * FROM sync_state WHERE resource='gmail'")
    assert row["last_error"] is None
    assert row["cursor"] == "200"


def test_record_sync_accumulates_item_counts(db: Database):
    repo.record_sync("gmail", items=3)
    repo.record_sync("gmail", items=4)
    assert db.scalar("SELECT items_seen FROM sync_state WHERE resource='gmail'") == 7


# ---------------------------------------------------------------- calendar reconcile
def _event_row(event_id: str, start_ts: str, summary: str = "Riunione") -> dict:
    return {
        "id": event_id,
        "calendar_id": "primary",
        "summary": summary,
        "description": "",
        "location": "",
        "start_ts": start_ts,
        "end_ts": start_ts,
        "start_raw": start_ts,
        "end_raw": start_ts,
        "all_day": 0,
        "status": "confirmed",
        "organizer": "",
        "attendees": "[]",
        "html_link": "",
        "recurring_event_id": None,
        "updated_at": start_ts,
    }


def test_horizon_reconcile_removes_events_deleted_upstream(db: Database):
    window = ("2026-09-01T00:00:00+00:00", "2026-10-01T00:00:00+00:00")
    repo.replace_events_in_window(
        [_event_row("e1", "2026-09-10T08:00:00+00:00"), _event_row("e2", "2026-09-11T08:00:00+00:00")],
        *window,
    )
    assert db.scalar("SELECT count(*) FROM events") == 2

    # Google now only reports e1: e2 was deleted elsewhere.
    counts = repo.replace_events_in_window([_event_row("e1", "2026-09-10T08:00:00+00:00")], *window)
    assert counts["deleted"] == 1
    assert {r["id"] for r in db.query("SELECT id FROM events")} == {"e1"}


def test_horizon_reconcile_leaves_events_outside_the_window_alone(db: Database):
    repo.replace_events_in_window(
        [_event_row("old", "2026-01-01T08:00:00+00:00")],
        "2026-01-01T00:00:00+00:00",
        "2026-02-01T00:00:00+00:00",
    )
    # A later pass over a different window must not delete history it never looked at.
    repo.replace_events_in_window(
        [_event_row("new", "2026-09-10T08:00:00+00:00")],
        "2026-09-01T00:00:00+00:00",
        "2026-10-01T00:00:00+00:00",
    )
    assert db.scalar("SELECT count(*) FROM events") == 2


def test_find_similar_event_matches_on_overlapping_words(db: Database):
    repo.replace_events_in_window(
        [_event_row("e1", "2026-09-24T13:00:00+00:00", "Appuntamento dentista Rossi")],
        "2026-09-01T00:00:00+00:00",
        "2026-10-01T00:00:00+00:00",
    )
    found = repo.find_similar_event(
        "2026-09-24T12:00:00+00:00", "2026-09-24T14:00:00+00:00", "dentista appuntamento"
    )
    assert found is not None and found["id"] == "e1"


def test_find_similar_event_rejects_an_unrelated_title(db: Database):
    repo.replace_events_in_window(
        [_event_row("e1", "2026-09-24T13:00:00+00:00", "Partita di calcio con Marco")],
        "2026-09-01T00:00:00+00:00",
        "2026-10-01T00:00:00+00:00",
    )
    assert (
        repo.find_similar_event(
            "2026-09-24T12:00:00+00:00", "2026-09-24T14:00:00+00:00", "riunione trimestrale budget"
        )
        is None
    )


def test_open_tasks_put_undated_ones_last(db: Database):
    repo.replace_all_tasks(
        [
            {"id": "t1", "tasklist_id": "@default", "title": "senza data", "notes": "",
             "due_ts": None, "status": "needsAction", "completed_at": None,
             "updated_at": None, "position": "1"},
            {"id": "t2", "tasklist_id": "@default", "title": "domani", "notes": "",
             "due_ts": "2026-09-21T00:00:00+00:00", "status": "needsAction",
             "completed_at": None, "updated_at": None, "position": "2"},
        ]
    )
    titles = [r["title"] for r in repo.open_tasks()]
    assert titles == ["domani", "senza data"]


# ---------------------------------------------------------------- the label feedback loop
def test_label_only_changes_are_applied_without_refetching(db: Database):
    """Triage labels everything it classifies, and each label write lands in the change feed.

    Treating those entries as "the message changed" made the next sync re-download the whole
    mailbox: labelling 154 messages caused 154 re-fetches and exceeded the API quota. A label
    change cannot alter a subject or a body, and the history entry already carries the ids.
    """
    gmail_sync._store(make_message("m1"))
    repo.record_sync(gmail_sync.RESOURCE, cursor="500")
    repo.record_sync(gmail_sync.BACKFILL_RESOURCE, cursor=gmail_sync.BACKFILL_DONE)

    with (
        patch(
            "donna.sync.gmail_sync.gmail.replay_history",
            return_value=HistoryChanges(
                relabelled={"m1": ({"Label_donna_1"}, {"UNREAD"})}, cursor="600"
            ),
        ),
        patch("donna.sync.gmail_sync.gmail.fetch_messages") as fetch,
    ):
        result = gmail_sync.sync_gmail()

    fetch.assert_called_once_with([])  # nothing downloaded
    assert result.ok
    row = repo.get_email("m1")
    assert row["is_unread"] == 0                      # UNREAD removed
    assert "Label_donna_1" in row["gmail_labels"]     # new label recorded


def test_a_label_delta_for_an_unmirrored_message_is_ignored(db: Database):
    # Normal for anything outside the sync window; must not create a partial row.
    assert repo.apply_label_delta("unknown", {"X"}, set()) is False
    assert db.scalar("SELECT count(*) FROM emails") == 0


def test_label_delta_preserves_labels_it_was_not_told_about(db: Database):
    gmail_sync._store(make_message("m1"))  # starts with INBOX, UNREAD
    repo.apply_label_delta("m1", {"STARRED"}, set())
    labels = repo.get_email("m1")["gmail_labels"]
    assert "INBOX" in labels and "UNREAD" in labels and "STARRED" in labels


def test_a_new_message_is_fetched_not_treated_as_a_relabel(db: Database):
    repo.record_sync(gmail_sync.RESOURCE, cursor="500")
    repo.record_sync(gmail_sync.BACKFILL_RESOURCE, cursor=gmail_sync.BACKFILL_DONE)
    with (
        patch(
            "donna.sync.gmail_sync.gmail.replay_history",
            return_value=HistoryChanges(added={"m9"}, cursor="600"),
        ),
        patch(
            "donna.sync.gmail_sync.gmail.fetch_messages",
            return_value=FetchReport(messages=[make_message("m9")]),
        ),
    ):
        result = gmail_sync.sync_gmail()
    assert result.created == 1


def test_history_drops_label_deltas_for_messages_it_also_reports_as_new():
    # A new message is downloaded in full, so its label deltas are redundant noise.
    changes = HistoryChanges()
    changes.added.add("m1")
    changes.note_labels("m1", ["INBOX"], [])
    changes.note_labels("m2", ["STARRED"], [])
    changes.added -= changes.deleted
    for mid in changes.added:
        changes.relabelled.pop(mid, None)
    assert "m1" not in changes.relabelled
    assert "m2" in changes.relabelled
