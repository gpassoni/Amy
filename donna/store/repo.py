"""Repositories over the mirror.

Kept as plain functions rather than classes: there is one database and one user, so an
object to hold that state would be ceremony. Everything here is deliberately explicit
about which columns it touches, so a sync pass can never clobber pipeline output (a sync
must not reset a triage decision just because Gmail sent the message again).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import timedelta
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


def apply_label_delta(message_id: str, added: set[str], removed: set[str]) -> bool:
    """Update an email's Gmail labels from a history delta, without re-fetching it.

    A label change cannot alter the subject or the body, so there is nothing to download.
    This is what stops triage's own label writes from causing the next sync to re-read the
    whole mailbox. Returns False when the message is not mirrored, which is normal for
    anything outside the sync window.
    """
    row = get_db().query_one("SELECT gmail_labels FROM emails WHERE id = ?", (message_id,))
    if row is None:
        return False

    try:
        labels = set(json.loads(row["gmail_labels"] or "[]"))
    except (ValueError, TypeError):
        labels = set()

    labels |= added
    labels -= removed
    get_db().execute(
        "UPDATE emails SET gmail_labels = ?, is_unread = ?, synced_at = ? WHERE id = ?",
        (
            json.dumps(sorted(labels)),
            int("UNREAD" in labels),
            iso_utc(now_utc()),
            message_id,
        ),
    )
    return True


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
    signal: str | None = None,
) -> None:
    """Store the triage verdict, keeping the raw signal beside the derived category."""
    if category not in CATEGORIES:
        raise ValueError(f"Categoria sconosciuta: {category!r}; attese {CATEGORIES}")
    get_db().execute(
        "UPDATE emails SET category = ?, category_signal = ?, category_confidence = ?,"
        " category_reason = ?, classified_at = ?, classifier_model = ?, classify_trace_id = ?,"
        " label_applied = ? WHERE id = ?",
        (
            category,
            signal,
            confidence,
            reason,
            iso_utc(now_utc()),
            model,
            trace_id,
            int(label_applied),
            email_id,
        ),
    )


def signal_counts() -> dict[str, int]:
    return {
        r["category_signal"]: r["n"]
        for r in get_db().query(
            "SELECT category_signal, count(*) n FROM emails"
            " WHERE category_signal IS NOT NULL GROUP BY category_signal ORDER BY n DESC"
        )
    }


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


def recent_emails(limit: int = 50) -> list[sqlite3.Row]:
    return get_db().query(
        "SELECT * FROM emails ORDER BY received_at DESC LIMIT ?", (limit,)
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
            # `source` and `origin_proposal_id` belong to the approval flow, not to Google.
            # Without preserving them, a sync immediately after an accept relabels Donna's
            # own event as a plain Google one and the provenance is lost.
            db.execute(
                upsert("events", row, preserve=("source", "origin_proposal_id")),
                list(row.values()),
            )

    return {"upserted": len(events), "deleted": len(stale)}


def events_between(start_iso: str, end_iso: str, *, include_cancelled: bool = False) -> list[sqlite3.Row]:
    sql = (
        "SELECT * FROM events WHERE start_ts >= ? AND start_ts < ?"
        + ("" if include_cancelled else " AND status != 'cancelled'")
        + " ORDER BY start_ts"
    )
    return get_db().query(sql, (start_iso, end_iso))


def get_event(event_id: str) -> sqlite3.Row | None:
    """A live (not cancelled) event from the mirror, by id."""
    return get_db().query_one(
        "SELECT * FROM events WHERE id = ? AND status != 'cancelled'", (event_id,)
    )


def find_events_by_id_prefix(prefix: str, *, limit: int = 2) -> list[sqlite3.Row]:
    """Live events whose id starts with `prefix`. `substr` rather than LIKE: ids contain `_`."""
    return get_db().query(
        "SELECT * FROM events WHERE substr(id, 1, ?) = ? AND status != 'cancelled' LIMIT ?",
        (len(prefix), prefix, limit),
    )


def apply_event_move(event_id: str, start_ts: str, end_ts: str) -> None:
    """Reflect an accepted move in the mirror straight away, rather than at the next sync."""
    get_db().execute(
        "UPDATE events SET start_ts = ?, end_ts = ?, start_raw = ?, end_raw = ?, updated_at = ?"
        " WHERE id = ?",
        (start_ts, end_ts, start_ts, end_ts, iso_utc(now_utc()), event_id),
    )


def mark_event_cancelled(event_id: str) -> None:
    """Reflect an accepted deletion in the mirror straight away."""
    get_db().execute(
        "UPDATE events SET status = 'cancelled', updated_at = ? WHERE id = ?",
        (iso_utc(now_utc()), event_id),
    )


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


def record_donna_event(
    *,
    event_id: str,
    proposal_id: int,
    summary: str,
    start_ts: str,
    end_ts: str,
    all_day: bool,
    location: str | None,
    html_link: str | None,
) -> None:
    """Insert an event Donna just created, with its provenance.

    Called by the approval flow rather than left to the next calendar sync. The sync would
    insert the row with source='google' and no proposal link — which is what happened the
    first time, losing the answer to "why is this on my calendar?" within 30 seconds of
    creating it.
    """
    now = iso_utc(now_utc())
    row = {
        "id": event_id,
        "calendar_id": "primary",
        "summary": summary,
        "description": "",
        "location": location or "",
        "start_ts": start_ts,
        "end_ts": end_ts,
        "start_raw": start_ts,
        "end_raw": end_ts,
        "all_day": int(all_day),
        "status": "confirmed",
        "organizer": "",
        "attendees": "[]",
        "html_link": html_link or "",
        "recurring_event_id": None,
        "updated_at": now,
        "source": "donna",
        "origin_proposal_id": proposal_id,
        "synced_at": now,
    }
    get_db().execute(upsert("events", row), list(row.values()))


def donna_events(limit: int = 50) -> list[sqlite3.Row]:
    """Events Donna created, newest first. The provenance view."""
    return get_db().query(
        "SELECT * FROM events WHERE origin_proposal_id IS NOT NULL"
        " ORDER BY start_ts DESC LIMIT ?",
        (limit,),
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


# ================================================================== proposals
# The approval gate. Donna never writes an inferred action to Google; she writes a row here
# and asks. Being real rows rather than in-memory state means proposals survive a restart
# and can be reviewed from Telegram, the web UI or the CLI interchangeably.
PROPOSAL_STATES = ("pending", "accepted", "rejected", "edited", "expired")


def create_proposal(
    *,
    kind: str,
    source_type: str,
    source_id: str | None,
    payload: dict[str, Any],
    reasoning: str | None = None,
    evidence_quote: str | None = None,
    confidence: float | None = None,
    trace_id: str | None = None,
    dedupe_key: str | None = None,
) -> int | None:
    """Insert a proposal. Returns None if an identical one already exists.

    The dedupe key is enforced by a unique index, so a duplicate is rejected by the
    database rather than by a racy pre-check. Deliberately this also suppresses re-proposing
    something the user already rejected: being asked twice about the same email is worse
    than missing it.
    """
    try:
        cursor = get_db().execute(
            "INSERT INTO proposals (kind, source_type, source_id, payload_json, reasoning,"
            " evidence_quote, confidence, state, created_at, trace_id, dedupe_key)"
            " VALUES (?,?,?,?,?,?,?,'pending',?,?,?)",
            (
                kind,
                source_type,
                source_id,
                json.dumps(payload, ensure_ascii=False),
                reasoning,
                evidence_quote,
                confidence,
                iso_utc(now_utc()),
                trace_id,
                dedupe_key,
            ),
        )
    except sqlite3.IntegrityError:
        return None
    return cursor.lastrowid


def get_proposal(proposal_id: int) -> sqlite3.Row | None:
    return get_db().query_one("SELECT * FROM proposals WHERE id = ?", (proposal_id,))


def pending_proposals(limit: int = 20) -> list[sqlite3.Row]:
    return get_db().query(
        "SELECT * FROM proposals WHERE state = 'pending' ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )


def resolved_proposals(limit: int = 20) -> list[sqlite3.Row]:
    """Recently decided proposals — the record of what she got right and wrong."""
    return get_db().query(
        "SELECT * FROM proposals WHERE state != 'pending' ORDER BY resolved_at DESC LIMIT ?",
        (limit,),
    )


def pending_proposal_count() -> int:
    return get_db().scalar(
        "SELECT count(*) FROM proposals WHERE state = 'pending'", default=0
    )


def unnotified_proposals(limit: int = 10) -> list[sqlite3.Row]:
    return get_db().query(
        "SELECT * FROM proposals WHERE state = 'pending' AND notified_at IS NULL"
        " ORDER BY created_at LIMIT ?",
        (limit,),
    )


def mark_proposal_notified(proposal_id: int) -> None:
    get_db().execute(
        "UPDATE proposals SET notified_at = ? WHERE id = ?", (iso_utc(now_utc()), proposal_id)
    )


def resolve_proposal(
    proposal_id: int,
    *,
    state: str,
    via: str,
    result_ref: str | None = None,
) -> bool:
    """Move a proposal out of pending. Only a pending proposal can be resolved.

    The state guard is what makes a double tap on a Telegram button harmless: the second
    one matches no rows and changes nothing, rather than creating a second calendar event.
    """
    if state not in PROPOSAL_STATES:
        raise ValueError(f"Stato proposta non valido: {state!r}")
    cursor = get_db().execute(
        "UPDATE proposals SET state = ?, resolved_at = ?, resolved_via = ?, result_ref = ?"
        " WHERE id = ? AND state = 'pending'",
        (state, iso_utc(now_utc()), via, result_ref, proposal_id),
    )
    return cursor.rowcount > 0


def reopen_proposal(proposal_id: int) -> bool:
    """Put a proposal back to pending after a failed write.

    `accept` claims the proposal before calling Google, so that a double tap cannot create
    two events. When the Google call then fails, the claim has to be released or the
    proposal is stuck in `accepted` with nothing to show for it.
    """
    cursor = get_db().execute(
        "UPDATE proposals SET state = 'pending', resolved_at = NULL, resolved_via = NULL,"
        " result_ref = NULL WHERE id = ? AND state = 'accepted' AND result_ref IS NULL",
        (proposal_id,),
    )
    return cursor.rowcount > 0


def attach_proposal_result(proposal_id: int, result_ref: str | None) -> None:
    get_db().execute(
        "UPDATE proposals SET result_ref = ? WHERE id = ?", (result_ref, proposal_id)
    )


def update_proposal_payload(proposal_id: int, payload: dict[str, Any]) -> None:
    get_db().execute(
        "UPDATE proposals SET payload_json = ? WHERE id = ?",
        (json.dumps(payload, ensure_ascii=False), proposal_id),
    )


def expire_old_proposals(days: int) -> int:
    """Age out proposals nobody answered, so the pending list stays meaningful."""
    cutoff = iso_utc(now_utc() - timedelta(days=days))
    cursor = get_db().execute(
        "UPDATE proposals SET state = 'expired', resolved_at = ?, resolved_via = 'expiry'"
        " WHERE state = 'pending' AND created_at < ?",
        (iso_utc(now_utc()), cutoff),
    )
    return cursor.rowcount


def proposal_payload(row: sqlite3.Row) -> dict[str, Any]:
    try:
        return json.loads(row["payload_json"])
    except (ValueError, TypeError):
        return {}


# ================================================================== conversation
def append_turn(
    channel: str,
    chat_id: str,
    *,
    role: str,
    content: str,
    agent: str | None = None,
    intent: str | None = None,
    trace_id: str | None = None,
) -> int:
    cursor = get_db().execute(
        "INSERT INTO conversations (channel, chat_id, role, content, agent, intent, trace_id,"
        " created_at) VALUES (?,?,?,?,?,?,?,?)",
        (channel, chat_id, role, content, agent, intent, trace_id, iso_utc(now_utc())),
    )
    return cursor.lastrowid


def recent_turns(channel: str, chat_id: str, *, limit: int = 8) -> list[dict[str, str]]:
    """The last `limit` turns, oldest first, shaped for a chat payload.

    Fetched newest-first with a LIMIT and then reversed, so a long history costs nothing: the
    index on (channel, chat_id, id DESC) makes this a partial scan rather than a full one.
    """
    rows = get_db().query(
        "SELECT role, content FROM conversations WHERE channel = ? AND chat_id = ?"
        " ORDER BY id DESC LIMIT ?",
        (channel, chat_id, limit),
    )
    return [{"role": r["role"], "content": r["content"] or ""} for r in reversed(rows)]


def conversation(channel: str, chat_id: str, *, limit: int = 40) -> list[sqlite3.Row]:
    """Full turns with metadata, for display. Unlike recent_turns, nothing is trimmed."""
    rows = get_db().query(
        "SELECT * FROM conversations WHERE channel = ? AND chat_id = ?"
        " ORDER BY id DESC LIMIT ?",
        (channel, chat_id, limit),
    )
    return list(reversed(rows))


def clear_turns(channel: str, chat_id: str) -> int:
    cursor = get_db().execute(
        "DELETE FROM conversations WHERE channel = ? AND chat_id = ?", (channel, chat_id)
    )
    get_db().execute(
        "DELETE FROM conversation_summaries WHERE channel = ? AND chat_id = ?", (channel, chat_id)
    )
    return cursor.rowcount


def turn_count(channel: str, chat_id: str) -> int:
    return get_db().scalar(
        "SELECT count(*) FROM conversations WHERE channel = ? AND chat_id = ?",
        (channel, chat_id),
        default=0,
    )


# ================================================================== feedback
def record_feedback(
    *,
    kind: str,
    trace_id: str | None = None,
    proposal_id: int | None = None,
    email_id: str | None = None,
    original_output: str | None = None,
    corrected_output: str | None = None,
    note: str | None = None,
) -> int:
    """Every correction the user makes. This accumulates the fine-tuning dataset from day
    one, which is the whole reason the SLM plan can start with prompts and keep LoRA open.
    """
    cursor = get_db().execute(
        "INSERT INTO feedback (kind, trace_id, proposal_id, email_id, original_output,"
        " corrected_output, note, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (
            kind,
            trace_id,
            proposal_id,
            email_id,
            original_output,
            corrected_output,
            note,
            iso_utc(now_utc()),
        ),
    )
    return cursor.lastrowid


def feedback_counts() -> dict[str, int]:
    return {
        r["kind"]: r["n"]
        for r in get_db().query("SELECT kind, count(*) n FROM feedback GROUP BY kind")
    }


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
