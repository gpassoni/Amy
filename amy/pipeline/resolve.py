"""Acting on a proposal.

This is the only place in the pipeline that writes to Google, and it only ever runs because
the user said so. Everything upstream produces rows in `proposals`.

Two invariants:

  * the state transition is the lock. `repo.resolve_proposal` only matches a row that is
    still pending, so a double tap on a Telegram button, or the same proposal accepted from
    the web UI and the CLI at once, creates one event and not two.
  * the calendar write happens only after the transition succeeds. Ordering it the other way
    would risk creating an event and then failing to record that we did.

Every outcome writes a `feedback` row, accepted and rejected alike. Rejections are the more
valuable half of that dataset: they are the only signal for what Amy should have left
alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from googleapiclient.errors import HttpError

from amy.google import calendar, tasks
from amy.store import repo
from amy.timeutil import format_it, format_range_it, parse_iso

logger = logging.getLogger(__name__)


class ProposalError(RuntimeError):
    """The proposal could not be acted on."""


@dataclass(slots=True)
class Resolution:
    proposal_id: int
    state: str
    message: str
    result_ref: str | None = None
    link: str | None = None


def _payload_window(payload: dict) -> tuple[str, str]:
    start, end = payload.get("start_ts"), payload.get("end_ts")
    if not start or not end:
        raise ProposalError("la proposta non ha un intervallo di tempo valido")
    return start, end


def describe(row) -> str:
    """One-line human description of a proposal, for any interface to render."""
    payload = repo.proposal_payload(row)
    start, end = parse_iso(payload.get("start_ts")), parse_iso(payload.get("end_ts"))
    title = payload.get("title") or "(senza titolo)"
    if row["kind"] == "calendar_move":
        old_start, old_end = (
            parse_iso(payload.get("old_start_ts")),
            parse_iso(payload.get("old_end_ts")),
        )
        before = f"{format_range_it(old_start, old_end)} → " if old_start else ""
        return f"Sposta «{title}»: {before}{format_range_it(start, end) if start else '?'}"
    if row["kind"] == "calendar_delete":
        return f"Elimina «{title}»" + (f" ({format_range_it(start, end)})" if start else "")
    if start is None:
        return title
    if payload.get("all_day"):
        return f"{title} — {format_it(start, with_time=False)} (tutto il giorno)"
    return f"{title} — {format_range_it(start, end)}"


def accept(proposal_id: int, *, via: str = "cli") -> Resolution:
    """Create the proposed thing in Google, then close the proposal."""
    row = repo.get_proposal(proposal_id)
    if row is None:
        raise ProposalError(f"proposta {proposal_id} non trovata")
    if row["state"] != "pending":
        # Not an error worth raising: the common cause is a second tap on the same button.
        return Resolution(
            proposal_id, row["state"], f"già gestita ({row['state']})", row["result_ref"]
        )

    payload = repo.proposal_payload(row)

    # Validate before claiming. A malformed payload is not a reason to move the proposal out
    # of pending — it would be stranded in `accepted` with nothing created, and the user
    # could not retry. (Found by a test: ProposalError escaped the handler below, which only
    # released the claim for Google-side failures.)
    start = end = None
    if row["kind"] in ("calendar_event", "calendar_move"):
        start, end = _payload_window(payload)
    elif row["kind"] not in ("task", "calendar_delete"):
        raise ProposalError(f"tipo di proposta non gestito: {row['kind']!r}")
    if row["kind"] in ("calendar_move", "calendar_delete") and not payload.get("event_id"):
        raise ProposalError("la proposta non dice quale evento toccare")

    # Claim the proposal. If this fails, someone else already took it.
    if not repo.resolve_proposal(proposal_id, state="accepted", via=via):
        current = repo.get_proposal(proposal_id)
        state = current["state"] if current else "sconosciuto"
        return Resolution(proposal_id, state, f"già gestita ({state})")

    try:
        if row["kind"] == "calendar_event":
            assert start is not None and end is not None
            created = calendar.create_event(
                payload.get("title") or "Impegno",
                start,
                end,
                description=_description_for(row, payload),
                location=payload.get("location"),
                reminder_minutes=payload.get("reminder_minutes"),
            )
            result_ref, link = created.get("id"), created.get("htmlLink")
        elif row["kind"] == "calendar_move":
            assert start is not None and end is not None
            updated = calendar.update_event(payload["event_id"], start_iso=start, end_iso=end)
            result_ref, link = payload["event_id"], updated.get("htmlLink")
        elif row["kind"] == "calendar_delete":
            calendar.delete_event(payload["event_id"])
            result_ref, link = payload["event_id"], None
        elif row["kind"] == "task":
            created = tasks.create_task(
                payload.get("title") or "Da fare",
                notes=_description_for(row, payload),
                due_iso=payload.get("start_ts"),
            )
            result_ref, link = created.get("id"), None
    except Exception as exc:
        # Anything from here on happens with the claim already taken, so the claim has to be
        # released or the proposal is stranded in `accepted` with nothing created and no way
        # for the user to retry. Deliberately broad: the failure mode is identical whether
        # Google refused, the network died, or the payload turned out to be malformed.
        repo.reopen_proposal(proposal_id)
        logger.error("Creazione da proposta %d fallita: %s", proposal_id, exc, exc_info=True)
        if isinstance(exc, (HttpError, OSError)):
            raise ProposalError(f"Google ha rifiutato la creazione: {exc}") from exc
        raise

    repo.attach_proposal_result(proposal_id, result_ref)
    # Google has already done it; the mirror is updated outside the try above so that a local
    # failure cannot reopen a proposal whose effect has landed.
    if row["kind"] == "calendar_move":
        repo.apply_event_move(payload["event_id"], start, end)
    elif row["kind"] == "calendar_delete":
        repo.mark_event_cancelled(payload["event_id"])
    if row["kind"] == "calendar_event" and result_ref:
        # Record it in the mirror now, with its provenance, rather than leaving it to the
        # next calendar sync. The sync inserts what Google reports, which does not include
        # "Amy created this from proposal N" — so waiting lost the link entirely.
        repo.record_amy_event(
            event_id=result_ref,
            proposal_id=proposal_id,
            summary=payload.get("title") or "Impegno",
            start_ts=start,
            end_ts=end,
            all_day=bool(payload.get("all_day")),
            location=payload.get("location"),
            html_link=link,
        )

    repo.record_feedback(
        kind="proposal_accept",
        trace_id=row["trace_id"],
        proposal_id=proposal_id,
        email_id=row["source_id"] if row["source_type"] == "email" else None,
        original_output=row["payload_json"],
    )

    logger.info("Proposta %d accettata (%s), %s", proposal_id, row["kind"], result_ref)
    return Resolution(proposal_id, "accepted", describe(row), result_ref, link)


def reject(proposal_id: int, *, via: str = "cli", note: str | None = None) -> Resolution:
    """Discard a proposal. The rejection is the training signal, so it is recorded."""
    row = repo.get_proposal(proposal_id)
    if row is None:
        raise ProposalError(f"proposta {proposal_id} non trovata")
    if row["state"] != "pending":
        return Resolution(proposal_id, row["state"], f"già gestita ({row['state']})")

    if not repo.resolve_proposal(proposal_id, state="rejected", via=via):
        current = repo.get_proposal(proposal_id)
        state = current["state"] if current else "sconosciuto"
        return Resolution(proposal_id, state, f"già gestita ({state})")

    repo.record_feedback(
        kind="proposal_reject",
        trace_id=row["trace_id"],
        proposal_id=proposal_id,
        email_id=row["source_id"] if row["source_type"] == "email" else None,
        original_output=row["payload_json"],
        note=note,
    )
    logger.info("Proposta %d rifiutata", proposal_id)
    return Resolution(proposal_id, "rejected", describe(row))


def edit_and_accept(
    proposal_id: int,
    *,
    title: str | None = None,
    start_ts: str | None = None,
    end_ts: str | None = None,
    location: str | None = None,
    all_day: bool | None = None,
    via: str = "cli",
) -> Resolution:
    """Amend a proposal and accept the amended version.

    The correction is recorded against the original payload, which makes this the single
    most informative feedback kind: it says not just that Amy was wrong but what the right
    answer was.
    """
    row = repo.get_proposal(proposal_id)
    if row is None:
        raise ProposalError(f"proposta {proposal_id} non trovata")
    if row["state"] != "pending":
        return Resolution(proposal_id, row["state"], f"già gestita ({row['state']})")

    original = row["payload_json"]
    payload = repo.proposal_payload(row)
    if title is not None:
        payload["title"] = title
    if start_ts is not None:
        payload["start_ts"] = start_ts
    if end_ts is not None:
        payload["end_ts"] = end_ts
    if location is not None:
        payload["location"] = location
    if all_day is not None:
        payload["all_day"] = all_day
    payload["edited"] = True

    repo.update_proposal_payload(proposal_id, payload)
    repo.record_feedback(
        kind="proposal_edit",
        trace_id=row["trace_id"],
        proposal_id=proposal_id,
        email_id=row["source_id"] if row["source_type"] == "email" else None,
        original_output=original,
        corrected_output=repo.get_proposal(proposal_id)["payload_json"],
    )

    resolution = accept(proposal_id, via=via)
    # `accept` records its own feedback row; the edit row above is the interesting one.
    return Resolution(
        proposal_id, "edited", resolution.message, resolution.result_ref, resolution.link
    )


def _description_for(row, payload: dict) -> str:
    """What Amy writes into the calendar entry itself.

    Deliberately includes where it came from. Six months on, "why is this on my calendar?"
    should be answerable from the calendar entry alone, without opening Amy.
    """
    lines = ["Aggiunto da Amy."]
    if row["reasoning"]:
        lines.append(f"Motivo: {row['reasoning']}")
    if row["evidence_quote"]:
        lines.append(f'Dall\'email: "{row["evidence_quote"]}"')
    if row["source_type"] == "email" and row["source_id"]:
        lines.append(f"https://mail.google.com/mail/u/0/#all/{row['source_id']}")
    return "\n".join(lines)


def expire_stale(days: int) -> int:
    count = repo.expire_old_proposals(days)
    if count:
        logger.info("%d proposte scadute senza risposta", count)
    return count
