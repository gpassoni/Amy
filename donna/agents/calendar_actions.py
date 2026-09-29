"""Turning a calendar request into proposals.

One place builds them, so the tool path (the model calls `proponi_evento`) and the structured
path (`fallback.py`) cannot drift apart. Three kinds exist, and all of them wait for approval:

    calendar_event   create something new
    calendar_move    change when an existing event happens
    calendar_delete  remove an existing event

Moves and deletes used to skip the gate: `sposta_evento` and `elimina_evento` wrote straight to
Google. That was inconsistent with creation — a wrong guess about which event to move, or what
"finishes at 15:00" means, landed in the calendar with nothing between the model and Google.
The event a move or delete refers to is always looked up in code, so its title in the proposal
is what the calendar says and not what the model remembers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from donna.store import repo
from donna.timeutil import format_range_it, iso_utc, parse_iso


class ActionError(ValueError):
    """A request that cannot become a proposal. The message is written for the user."""


@dataclass(slots=True)
class Prepared:
    proposal_id: int | None
    description: str  # one line, what the proposal does
    tool: str = "proponi_evento"  # the agent tool this corresponds to, for the activity log


def _clean_id(raw: str | None) -> str:
    """The id as the model copied it, minus the decoration it copied along.

    `elenca_eventi` prints ids as «[abc123] domani 08:00 — Lavoro», and the model hands back
    «[abc123]», brackets included. The id is still the only way to name an event — this only
    strips what was never part of it.
    """
    return (raw or "").strip().strip("[](){}<>\"'` ").strip()


def _event_or_raise(event_id: str | None):
    cleaned = _clean_id(event_id)
    if not cleaned:
        raise ActionError("non so di quale impegno parli")
    row = repo.get_event(cleaned)
    if row is None and len(cleaned) >= 12:
        # A slip in the tail of a 60-character id is common; a unique prefix is unambiguous.
        matches = repo.find_events_by_id_prefix(cleaned)
        row = matches[0] if len(matches) == 1 else None
    if row is None:
        raise ActionError(f"non trovo in calendario nessun impegno con id {cleaned!r}")
    return row


def _range(row) -> tuple[datetime, datetime]:
    start, end = parse_iso(row["start_ts"]), parse_iso(row["end_ts"])
    if start is None:
        raise ActionError(f"l'impegno «{row['summary']}» non ha un orario che io possa leggere")
    return start, end or start + timedelta(hours=1)


def propose_create(
    *,
    title: str,
    start: datetime,
    end: datetime,
    location: str | None = None,
    reminder_minutes: int | None = None,
    reasoning: str = "Me l'hai chiesto tu in chat.",
    confidence: float = 1.0,
    trace_id: str | None = None,
) -> Prepared:
    payload: dict[str, Any] = {
        "kind": "appuntamento",
        "title": title,
        "start_ts": iso_utc(start),
        "end_ts": iso_utc(end),
        "all_day": False,
        "location": location,
    }
    if reminder_minutes:
        payload["reminder_minutes"] = int(reminder_minutes)

    proposal_id = repo.create_proposal(
        kind="calendar_event",
        source_type="conversation",
        source_id=None,
        payload=payload,
        reasoning=reasoning,
        confidence=confidence,
        trace_id=trace_id,
    )
    reminder = f", con promemoria {reminder_minutes} minuti prima" if reminder_minutes else ""
    return Prepared(proposal_id, f"{title} — {format_range_it(start, end)}{reminder}")


def propose_move(
    *,
    event_id: str,
    start: datetime | None,
    end: datetime | None,
    keep_end: bool = False,
    reasoning: str = "Me l'hai chiesto tu in chat.",
    confidence: float = 1.0,
    trace_id: str | None = None,
) -> Prepared:
    """Move an existing event.

    `keep_end` is for «inizia alle 9»: the start changes and the end stays where it is. Without
    it a new start moves the whole event and keeps its length, which is right for «spostalo alle
    18» and wrong for a shift that merely starts later.

    Anything the request leaves unsaid is kept from the event itself. "Finisce alle 15:00" names
    a new end and no start; "spostalo alle 18" names a new start and no end, and defaulting that
    end to one hour would silently turn a two-hour meeting into a one-hour one.
    """
    row = _event_or_raise(event_id)
    old_start, old_end = _range(row)
    new_start = start or old_start
    if keep_end:
        new_end = old_end
    else:
        new_end = end or (new_start + (old_end - old_start))
    if new_end <= new_start:
        raise ActionError(f"il nuovo orario di «{row['summary']}» finirebbe prima di iniziare")
    if (new_start, new_end) == (old_start, old_end):
        raise ActionError(f"«{row['summary']}» è già a quell'orario")

    payload = {
        "kind": "spostamento",
        "event_id": row["id"],
        "title": row["summary"],
        "start_ts": iso_utc(new_start),
        "end_ts": iso_utc(new_end),
        "old_start_ts": row["start_ts"],
        "old_end_ts": row["end_ts"],
        "all_day": False,
    }
    proposal_id = repo.create_proposal(
        kind="calendar_move",
        source_type="conversation",
        source_id=None,
        payload=payload,
        reasoning=reasoning,
        confidence=confidence,
        trace_id=trace_id,
    )
    return Prepared(
        proposal_id,
        tool="sposta_evento",
        description=f"Sposta «{row['summary']}»: {format_range_it(old_start, old_end)}"
        f" → {format_range_it(new_start, new_end)}",
    )


def propose_delete(
    *,
    event_id: str,
    reasoning: str = "Me l'hai chiesto tu in chat.",
    confidence: float = 1.0,
    trace_id: str | None = None,
) -> Prepared:
    row = _event_or_raise(event_id)
    start, end = _range(row)
    payload = {
        "kind": "eliminazione",
        "event_id": row["id"],
        "title": row["summary"],
        "start_ts": row["start_ts"],
        "end_ts": row["end_ts"],
        "all_day": bool(row["all_day"]),
    }
    proposal_id = repo.create_proposal(
        kind="calendar_delete",
        source_type="conversation",
        source_id=None,
        payload=payload,
        reasoning=reasoning,
        confidence=confidence,
        trace_id=trace_id,
    )
    return Prepared(
        proposal_id,
        tool="elimina_evento",
        description=f"Elimina «{row['summary']}» ({format_range_it(start, end)})",
    )
