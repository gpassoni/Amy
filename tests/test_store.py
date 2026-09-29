"""Store behaviour that the rest of the system assumes: migrations are atomic and
idempotent, the FTS indexes track their content tables, and the partial indexes that back
the pipeline queues actually exist.
"""

from __future__ import annotations

import sqlite3

import pytest

from donna.store.db import Database, upsert
from donna.timeutil import now_utc


def test_migrate_creates_the_expected_tables(db: Database):
    tables = set(db.table_names())
    for expected in (
        "emails",
        "events",
        "tasks",
        "proposals",
        "facts",
        "conversations",
        "traces",
        "feedback",
        "sync_state",
        "schema_migrations",
    ):
        assert expected in tables


def test_migrate_is_idempotent(db: Database):
    assert db.migrate() == []  # already applied by the fixture


def test_wal_mode_is_on_so_sync_does_not_block_readers(db: Database):
    assert db.scalar("PRAGMA journal_mode").lower() == "wal"


def _insert_email(db: Database, email_id: str, subject: str, body: str) -> None:
    db.execute(
        "INSERT INTO emails (id, subject, body, received_at, synced_at) VALUES (?,?,?,?,?)",
        (email_id, subject, body, now_utc().isoformat(), now_utc().isoformat()),
    )


def test_fts_finds_an_inserted_email(db: Database):
    _insert_email(db, "m1", "Conferma appuntamento", "Appuntamento dal dentista martedì")
    rows = db.query("SELECT rowid FROM emails_fts WHERE emails_fts MATCH 'dentista'")
    assert len(rows) == 1


def test_fts_is_accent_and_case_insensitive_enough_for_italian(db: Database):
    _insert_email(db, "m1", "Riunione", "Appuntamento martedì pomeriggio")
    assert db.query("SELECT rowid FROM emails_fts WHERE emails_fts MATCH 'APPUNTAMENTO'")


def test_triage_update_does_not_corrupt_the_fts_index(db: Database):
    # The FTS triggers are scoped to UPDATE OF subject, body. A classification write must
    # therefore leave the index intact and still findable.
    _insert_email(db, "m1", "Conferma appuntamento", "Appuntamento dal dentista")
    db.execute(
        "UPDATE emails SET category = 'importante', category_confidence = 0.9 WHERE id = 'm1'"
    )
    assert db.query("SELECT rowid FROM emails_fts WHERE emails_fts MATCH 'dentista'")


def test_editing_the_body_reindexes(db: Database):
    _insert_email(db, "m1", "Oggetto", "testo originale")
    db.execute("UPDATE emails SET body = 'testo sostituito' WHERE id = 'm1'")
    assert not db.query("SELECT rowid FROM emails_fts WHERE emails_fts MATCH 'originale'")
    assert db.query("SELECT rowid FROM emails_fts WHERE emails_fts MATCH 'sostituito'")


def test_deleting_an_email_removes_it_from_the_index(db: Database):
    _insert_email(db, "m1", "Oggetto", "dentista")
    db.execute("DELETE FROM emails WHERE id = 'm1'")
    assert not db.query("SELECT rowid FROM emails_fts WHERE emails_fts MATCH 'dentista'")


def test_dedupe_key_is_unique_but_nulls_are_unconstrained(db: Database):
    # Several proposals may legitimately have no dedupe key; two with the same key is a bug.
    now = now_utc().isoformat()
    for _ in range(2):
        db.execute(
            "INSERT INTO proposals (kind, source_type, payload_json, created_at)"
            " VALUES ('calendar_event','email','{}',?)",
            (now,),
        )
    db.execute(
        "INSERT INTO proposals (kind, source_type, payload_json, created_at, dedupe_key)"
        " VALUES ('calendar_event','email','{}',?,'k1')",
        (now,),
    )
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO proposals (kind, source_type, payload_json, created_at, dedupe_key)"
            " VALUES ('calendar_event','email','{}',?,'k1')",
            (now,),
        )


def test_proposal_defaults_to_pending(db: Database):
    db.execute(
        "INSERT INTO proposals (kind, source_type, payload_json, created_at)"
        " VALUES ('calendar_event','email','{}',?)",
        (now_utc().isoformat(),),
    )
    assert db.scalar("SELECT state FROM proposals") == "pending"


def test_a_fact_cannot_be_stored_twice(db: Database):
    now = now_utc().isoformat()
    db.execute("INSERT INTO facts (text, created_at) VALUES ('va in palestra', ?)", (now,))
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO facts (text, created_at) VALUES ('va in palestra', ?)", (now,))


def test_tx_rolls_back_on_error(db: Database):
    with pytest.raises(RuntimeError):
        with db.tx():
            _insert_email(db, "m1", "Oggetto", "corpo")
            raise RuntimeError("boom")
    assert db.scalar("SELECT count(*) FROM emails") == 0


def test_upsert_builder_updates_every_column_but_the_key():
    sql = upsert("emails", {"id": "m1", "subject": "s", "body": "b"})
    assert sql.startswith("INSERT INTO emails (id, subject, body)")
    assert "ON CONFLICT(id) DO UPDATE SET subject=excluded.subject, body=excluded.body" in sql
    assert "id=excluded.id" not in sql


def test_upsert_round_trip_replaces_the_row(db: Database):
    now = now_utc().isoformat()
    row = {"id": "m1", "subject": "prima", "received_at": now, "synced_at": now}
    db.execute(upsert("emails", row), list(row.values()))
    row["subject"] = "dopo"
    db.execute(upsert("emails", row), list(row.values()))
    assert db.scalar("SELECT count(*) FROM emails") == 1
    assert db.scalar("SELECT subject FROM emails") == "dopo"


def test_pipeline_queue_indexes_exist(db: Database):
    names = {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type='index'")}
    assert "idx_emails_untriaged" in names
    assert "idx_emails_unextracted" in names
    assert "idx_proposals_pending" in names
