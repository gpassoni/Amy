"""Turning corrections into training examples.

Every accept, reject, edit and reclassify is already recorded in `feedback`. This module joins
those rows back to their source — the email text, the original model output, the corrected
output — and emits examples in a shape a fine-tune can consume.

Why this exists as its own module rather than a script: the value of the feedback table depends
entirely on whether it can still be turned into examples in six months. Keeping the join in
code, tested, next to the schema is what makes that true. A one-off notebook would not be.

Three example kinds come out of it:

    triage      email -> signal          (from reclassify)
    extraction  email -> commitment      (from proposal_edit, and from accept/reject as a
                                          positive/negative label on the whole extraction)
    proposal    email -> accept/reject   (the cheapest signal, and the most plentiful)

The `prompt` field is reconstructed with the *current* prompt builders on purpose, so an
exported dataset always reflects how the model is actually asked today.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from amy.pipeline import prompts
from amy.store.db import get_db
from amy.timeutil import format_it, parse_iso

logger = logging.getLogger(__name__)

TRIAGE = "triage"
EXTRACTION = "extraction"
PROPOSAL = "proposal"


@dataclass(slots=True)
class Example:
    kind: str
    feedback_id: int
    created_at: str
    system: str
    user: str
    # What the model said, and what it should have said. For accept/reject there is no
    # corrected output — the label is the decision itself, carried in `label`.
    produced: Any = None
    expected: Any = None
    label: str | None = None
    source_email_id: str | None = None
    trace_id: str | None = None
    note: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _email(email_id: str | None):
    if not email_id:
        return None
    return get_db().query_one("SELECT * FROM emails WHERE id = ?", (email_id,))


def _loads(raw: str | None) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def iter_examples(limit: int = 1000) -> Iterator[Example]:
    """Walk the feedback log, newest first, yielding what can be turned into an example."""
    rows = get_db().query("SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (limit,))

    for row in rows:
        email = _email(row["email_id"])

        if row["kind"] == "reclassify":
            if email is None:
                continue  # the email was deleted; the correction is no longer groundable
            yield Example(
                kind=TRIAGE,
                feedback_id=row["id"],
                created_at=row["created_at"],
                system=prompts.CLASSIFY_SYSTEM,
                user=prompts.classify_user(
                    email["from_name"] or "",
                    email["from_addr"] or "",
                    email["subject"] or "",
                    email["body"] or "",
                ),
                produced=row["original_output"],
                expected=row["corrected_output"],
                label=row["corrected_output"],
                source_email_id=row["email_id"],
                trace_id=row["trace_id"],
                note=row["note"],
            )
            continue

        if row["kind"] in {"proposal_accept", "proposal_reject", "proposal_edit"}:
            if email is None:
                continue
            received = parse_iso(email["received_at"])
            user = prompts.extract_user(
                email["from_name"] or "",
                email["from_addr"] or "",
                email["subject"] or "",
                email["body"] or "",
                format_it(received) if received else "",
            )

            if row["kind"] == "proposal_edit":
                # The richest signal available: not just that she was wrong, but what right
                # looked like.
                yield Example(
                    kind=EXTRACTION,
                    feedback_id=row["id"],
                    created_at=row["created_at"],
                    system=prompts.EXTRACT_SYSTEM,
                    user=user,
                    produced=_loads(row["original_output"]),
                    expected=_loads(row["corrected_output"]),
                    label="edited",
                    source_email_id=row["email_id"],
                    trace_id=row["trace_id"],
                    note=row["note"],
                )
                continue

            accepted = row["kind"] == "proposal_accept"
            yield Example(
                kind=PROPOSAL,
                feedback_id=row["id"],
                created_at=row["created_at"],
                system=prompts.EXTRACT_SYSTEM,
                user=user,
                produced=_loads(row["original_output"]),
                label="accept" if accepted else "reject",
                source_email_id=row["email_id"],
                trace_id=row["trace_id"],
                note=row["note"],
                meta={"proposal_id": row["proposal_id"]},
            )


def counts() -> dict[str, int]:
    """How many usable examples exist, by kind. What the dashboard shows."""
    totals: dict[str, int] = {TRIAGE: 0, EXTRACTION: 0, PROPOSAL: 0}
    for example in iter_examples(limit=100_000):
        totals[example.kind] = totals.get(example.kind, 0) + 1
    return totals


def export_jsonl(path: Path, *, kind: str | None = None, limit: int = 100_000) -> int:
    """Write examples to JSONL. Returns how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for example in iter_examples(limit=limit):
            if kind and example.kind != kind:
                continue
            handle.write(json.dumps(asdict(example), ensure_ascii=False, default=str) + "\n")
            written += 1
    logger.info("Esportati %d esempi in %s", written, path)
    return written


def as_chat_messages(example: Example) -> list[dict[str, str]]:
    """An example in the messages shape most fine-tuning toolchains expect.

    Only meaningful where a corrected answer exists: an accept/reject is a preference label,
    not a target completion, and pretending otherwise would train on the model's own output.
    """
    if example.expected is None:
        raise ValueError(
            f"L'esempio {example.feedback_id} ({example.kind}) non ha una risposta corretta: "
            "è un'etichetta di preferenza, non un target."
        )
    expected = (
        example.expected
        if isinstance(example.expected, str)
        else json.dumps(example.expected, ensure_ascii=False)
    )
    return [
        {"role": "system", "content": example.system},
        {"role": "user", "content": example.user},
        {"role": "assistant", "content": expected},
    ]
