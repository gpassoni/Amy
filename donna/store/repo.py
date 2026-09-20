"""Repositories over the mirror.

Kept as plain functions rather than classes: there is one database and one user, so an
object to hold that state would be ceremony. Everything here is deliberately explicit
about which columns it touches, so a sync pass can never clobber pipeline output (a sync
must not reset a triage decision just because Gmail sent the message again).
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable, Sequence

from donna.store.db import get_db, upsert
from donna.timeutil import iso_utc, now_utc

CATEGORIES = ("importante", "da_leggere", "inutile")


# ================================================================== emails
def upsert_email(
    *,
    id: str,
    thread_id: str,
    from_addr: str,
    from_name: str,
    to_addrs: str,
    subject: str,
    snippet: str,
    body: str,
    received_at: str,
    label_ids: Sequence[str],
    is_unread: bool,
) -> bool:
    """Insert or refresh a message. Returns True if this was a new message.

    Triage columns are intentionally absent from the UPDATE set: re-syncing a message must
    not discard its classification. Only the fields Gmail owns are refreshed.
    """
    db = get_db()
    existed = db.scalar("SELECT 1 FROM emails WHERE id = ?", (id,)) is not None
    now = iso_utc(now_utc())

    if existed:
        db.execute(
            "UPDATE emails SET gmail_labels = ?, is_unread = ?, synced_at = ? WHERE id = ?",
            (json.dumps(list(label_ids)), int(is_unread), now, id),
        )
        return False

    row = {
        "id": id,
        "thread_id": thread_id,
        "from_addr": from_addr,
        "from_name": from_name,
        "to_addrs": to_addrs,
        "subject": subject,
        "snippet": snippet,
        "body": body,
        "received_at": received_at,
        "gmail_labels": json.dumps(list(label_ids)),
        "is_unread": int(is_unread),
        "synced_at": now,
    }
    db.execute(upsert("emails", row), list(row.values()))
    return True


def known_email_ids(ids: Sequence[str]) -> set[str]:
    """Which of `ids` we already hold.

    Lets a backfill resume: a pass interrupted by a rate limit re-lists the same ids next
    time and only fetches the ones still missing.
    """
    if not ids:
        return set()
    found: set[str] = set()
    db = get_db()
    # Chunked to stay under SQLite's variable limit on a large backfill.
    for offset in range(0, len(ids), 500):
        chunk = ids[offset : offset + 500]
        placeholders = ",".join("?" for _ in chunk)
        found.update(
            r["id"] for r in db.query(f"SELECT id FROM emails WHERE id IN ({placeholders})", chunk)
        )
    return found


def delete_emails(ids: Iterable[str]) -> int:
    ids = list(ids)
    if not ids:
        return 0
    placeholders = ",".join("?" for _ in ids)
    cursor = get_db().execute(f"DELETE FROM emails WHERE id IN ({placeholders})", ids)
    return cursor.rowcount


def untriaged_emails(limit: int) -> list[sqlite3.Row]:
    """The triage queue: newest first, so a backlog surfaces what is current."""
    return get_db().query(
        "SELECT id, from_addr, from_name, subject, body, received_at FROM emails"
        " WHERE category IS NULL ORDER BY received_at DESC LIMIT ?",
        (limit,),
    )


def emails_awaiting_extraction(limit: int) -> list[sqlite3.Row]:
    return get_db().query(
        "SELECT id, from_addr, from_name, subject, body, received_at FROM emails"
        " WHERE category = 'importante' AND extracted_at IS NULL"
        " ORDER BY received_at DESC LIMIT ?",
        (limit,),
    )


def set_email_category(
    email_id: str,
    *,
    category: str,
    confidence: float,
    reason: str,
    model: str,
    trace_id: str,
    label_applied: bool = False,
) -> None:
    if category not in CATEGORIES:
        raise ValueError(f"Categoria sconosciuta: {category!r}; attese {CATEGORIES}")
    get_db().execute(
        "UPDATE emails SET category = ?, category_confidence = ?, category_reason = ?,"
        " classified_at = ?, classifier_model = ?, classify_trace_id = ?, label_applied = ?"
        " WHERE id = ?",
        (
            category,
            confidence,
            reason,
            iso_utc(now_utc()),
            model,
            trace_id,
            int(label_applied),
            email_id,
        ),
    )


def mark_email_extracted(email_id: str, trace_id: str | None) -> None:
    get_db().execute(
        "UPDATE emails SET extracted_at = ?, extract_trace_id = ? WHERE id = ?",
        (iso_utc(now_utc()), trace_id, email_id),
    )


def get_email(email_id: str) -> sqlite3.Row | None:
    return get_db().query_one("SELECT * FROM emails WHERE id = ?", (email_id,))


def search_emails(query: str, limit: int = 10) -> list[sqlite3.Row]:
    """Full-text search over subject and body."""
    return get_db().query(
        "SELECT e.* FROM emails e JOIN emails_fts f ON f.rowid = e.rowid"
        " WHERE emails_fts MATCH ? ORDER BY e.received_at DESC LIMIT ?",
        (query, limit),
    )


def emails_by_category(category: str, *, since: str | None = None, limit: int = 20) -> list[sqlite3.Row]:
    if since:
        return get_db().query(
            "SELECT * FROM emails WHERE category = ? AND received_at >= ?"
            " ORDER BY received_at DESC LIMIT ?",
            (category, since, limit),
        )
    return get_db().query(
        "SELECT * FROM emails WHERE category = ? ORDER BY received_at DESC LIMIT ?",
        (category, limit),
    )


def category_counts(since: str | None = None) -> dict[str, int]:
    sql = "SELECT COALESCE(category, 'da_classificare') AS k, count(*) AS n FROM emails"
    params: list[Any] = []
    if since:
        sql += " WHERE received_at >= ?"
        params.append(since)
    sql += " GROUP BY k"
    return {r["k"]: r["n"] for r in get_db().query(sql, params)}


# ================================================================== events
def replace_events_in_window(events: list[dict[str, Any]], start_iso: str, end_iso: str) -> dict[str, int]:
    """Reconcile the mirrored horizon against what Google just returned.

    Rows in the window that Google no longer lists were deleted elsewhere, so they are
    removed. Events Donna created are matched by id like any other, so a deletion made in
    Google Calendar propagates here.
    """
    db = get_db()
    seen = {event["id"] for event in events}

    with db.tx():
        existing = {
            r["id"]
            for r in db.query(
                "SELECT id FROM events WHERE start_ts >= ? AND start_ts < ?", (start_iso, end_iso)
            )
        }
        stale = existing - seen
        if stale:
            placeholders = ",".join("?" for _ in stale)
            db.execute(f"DELETE FROM events WHERE id IN ({placeholders})", list(stale))

        now = iso_utc(now_utc())
        for event in events:
            row = dict(event)
            row["synced_at"] = now
            db.execute(upsert("events", row), list(row.values()))

    return {"upserted": len(events), "deleted": len(stale)}


def events_between(start_iso: str, end_iso: str, *, include_cancelled: bool = False) -> list[sqlite3.Row]:
    sql = (
        "SELECT * FROM events WHERE start_ts >= ? AND start_ts < ?"
        + ("" if include_cancelled else " AND status != 'cancelled'")
        + " ORDER BY start_ts"
    )
    return get_db().query(sql, (start_iso, end_iso))


def find_similar_event(start_iso: str, end_iso: str, title: str) -> sqlite3.Row | None:
    """An event already near this slot with a comparable title.

    Used to avoid proposing something that is already on the calendar. Deliberately fuzzy
    on the title and tight on the time: the same appointment rarely moves, but its wording
    varies between the invite and what the user typed.
    """
    candidates = get_db().query(
        "SELECT * FROM events WHERE start_ts >= ? AND start_ts < ? AND status != 'cancelled'",
        (start_iso, end_iso),
    )
    needle = {w for w in title.lower().split() if len(w) > 3}
    if not needle:
        return candidates[0] if candidates else None

    for row in candidates:
        words = {w for w in (row["summary"] or "").lower().split() if len(w) > 3}
        if words and len(needle & words) / len(needle | words) >= 0.3:
            return row
    return None


def link_event_to_proposal(event_id: str, proposal_id: int) -> None:
    get_db().execute(
        "UPDATE events SET origin_proposal_id = ?, source = 'donna' WHERE id = ?",
        (proposal_id, event_id),
    )


# ================================================================== tasks
def replace_all_tasks(tasks: list[dict[str, Any]]) -> dict[str, int]:
    db = get_db()
    seen = {task["id"] for task in tasks}

    with db.tx():
        existing = {r["id"] for r in db.query("SELECT id FROM tasks")}
        stale = existing - seen
        if stale:
            placeholders = ",".join("?" for _ in stale)
            db.execute(f"DELETE FROM tasks WHERE id IN ({placeholders})", list(stale))

        now = iso_utc(now_utc())
        for task in tasks:
            row = dict(task)
            row["synced_at"] = now
            db.execute(upsert("tasks", row), list(row.values()))

    return {"upserted": len(tasks), "deleted": len(stale)}


def open_tasks(limit: int = 25) -> list[sqlite3.Row]:
    # NULLs last: an undated task should not outrank one that is due tomorrow.
    return get_db().query(
        "SELECT * FROM tasks WHERE status = 'needsAction'"
        " ORDER BY due_ts IS NULL, due_ts LIMIT ?",
        (limit,),
    )


def overdue_tasks(limit: int = 25) -> list[sqlite3.Row]:
    return get_db().query(
        "SELECT * FROM tasks WHERE status = 'needsAction' AND due_ts IS NOT NULL AND due_ts < ?"
        " ORDER BY due_ts LIMIT ?",
        (iso_utc(now_utc()), limit),
    )


def find_task(needle: str) -> sqlite3.Row | None:
    """Resolve a task by id, then by substring — v1's behaviour, which users relied on."""
    db = get_db()
    exact = db.query_one("SELECT * FROM tasks WHERE id = ?", (needle,))
    if exact:
        return exact
    return db.query_one(
        "SELECT * FROM tasks WHERE status = 'needsAction' AND lower(title) LIKE ?"
        " ORDER BY due_ts IS NULL, due_ts LIMIT 1",
        (f"%{needle.lower()}%",),
    )


# ================================================================== sync bookkeeping
def get_cursor(resource: str) -> str | None:
    return get_db().scalar("SELECT cursor FROM sync_state WHERE resource = ?", (resource,))


def record_sync(
    resource: str,
    *,
    cursor: str | None = None,
    error: str | None = None,
    items: int = 0,
) -> None:
    now = iso_utc(now_utc())
    get_db().execute(
        "INSERT INTO sync_state (resource, cursor, last_run_at, last_success_at, last_error,"
        " items_seen) VALUES (?,?,?,?,?,?)"
        " ON CONFLICT(resource) DO UPDATE SET"
        "   cursor = COALESCE(excluded.cursor, sync_state.cursor),"
        "   last_run_at = excluded.last_run_at,"
        "   last_success_at = CASE WHEN excluded.last_error IS NULL"
        "     THEN excluded.last_success_at ELSE sync_state.last_success_at END,"
        "   last_error = excluded.last_error,"
        "   items_seen = sync_state.items_seen + excluded.items_seen",
        (resource, cursor, now, None if error else now, error, items),
    )


def sync_status() -> list[sqlite3.Row]:
    return get_db().query("SELECT * FROM sync_state ORDER BY resource")
