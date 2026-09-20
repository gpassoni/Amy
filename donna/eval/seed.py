"""Inject fixture emails into the mirror as if they had just arrived.

Needed because a mailbox with no future commitments cannot exercise the half of the pipeline
that matters most: extraction, proposal, approval, calendar write. The test account has 154
messages and not one appointment, and waiting for a real one to show up is not a test
strategy.

Seeded rows carry a `seed:` id prefix so they are recognisable, never confused with real
Gmail ids, and removable in one statement. They are never pushed to Gmail — they exist only
in the local mirror, so the Gmail labelling step is skipped for them.

Timestamps are rewritten relative to now rather than to the fixtures' fixed reference date,
so "domani alle 14:30" resolves to an actual tomorrow and the proposal is actionable.
"""
from __future__ import annotations

import logging
from datetime import timedelta

from donna.eval.fixtures import CASES, EmailCase
from donna.store import repo
from donna.store.db import get_db
from donna.timeutil import iso_utc, local_zone, now_local, now_utc

logger = logging.getLogger(__name__)

SEED_PREFIX = "seed:"


def is_seeded(email_id: str) -> bool:
    return email_id.startswith(SEED_PREFIX)


def _received_at() -> str:
    """A receipt time that makes relative phrasing resolve usefully.

    Pinned to 09:00 local this morning: the fixtures are written as if received on a weekday
    morning, and "domani alle 14:30" from 09:00 today is unambiguous.
    """
    morning = now_local().replace(hour=9, minute=0, second=0, microsecond=0, tzinfo=local_zone())
    if morning > now_local():
        morning -= timedelta(days=1)
    return iso_utc(morning)


def seed(cases: list[EmailCase] | None = None, *, only_commitments: bool = False) -> int:
    """Insert (or refresh) the fixture emails. Returns how many were written."""
    chosen = cases or CASES
    if only_commitments:
        chosen = [c for c in chosen if c.expect_commitment]

    received = _received_at()
    written = 0
    for case in chosen:
        email_id = f"{SEED_PREFIX}{case.name}"
        # Remove first so a reseed re-runs triage and extraction from scratch rather than
        # being skipped as already classified.
        get_db().execute("DELETE FROM emails WHERE id = ?", (email_id,))
        repo.upsert_email(
            id=email_id,
            thread_id=email_id,
            from_addr=case.sender_addr,
            from_name=case.sender_name,
            to_addrs="me",
            subject=case.subject,
            snippet=case.body[:120],
            body=case.body,
            received_at=received,
            label_ids=["INBOX", "UNREAD"],
            is_unread=True,
        )
        written += 1

    logger.info("Seed: %d email di test inserite nel mirror", written)
    return written


def clear() -> dict[str, int]:
    """Remove every seeded artefact: emails, their proposals, and their traces."""
    db = get_db()
    proposals = db.execute(
        "DELETE FROM proposals WHERE source_type = 'email' AND source_id LIKE ?",
        (f"{SEED_PREFIX}%",),
    ).rowcount
    emails = db.execute("DELETE FROM emails WHERE id LIKE ?", (f"{SEED_PREFIX}%",)).rowcount
    return {"emails": emails, "proposals": proposals}


def seeded_count() -> int:
    return get_db().scalar(
        "SELECT count(*) FROM emails WHERE id LIKE ?", (f"{SEED_PREFIX}%",), default=0
    )
