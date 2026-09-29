"""Commitment extraction: find the thing in an email that belongs on the calendar, and
propose it.

The division of labour, which is the whole point of this module:

    the model   decides whether there is a commitment, quotes the date phrase verbatim,
                and writes a calendar-style title
    the code    resolves the phrase to an actual timestamp, applies a duration, checks the
                calendar for a duplicate, and decides whether to bother the user at all

Nothing here writes to Google. The output is a row in `proposals`, which the user accepts
or rejects. That is by design: an extraction that is wrong in a proposal costs a tap, and
the same extraction wrong in a calendar entry costs trust.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta

from amy.config import get_settings
from amy.llm import registry
from amy.llm.client import LLMError, get_llm
from amy.pipeline import dates, prompts
from amy.pipeline.schemas import Commitment
from amy.store import repo
from amy.timeutil import format_it, iso_utc, now_utc, parse_iso

logger = logging.getLogger(__name__)

# How far either side of a proposed start to look for an event that is already on the
# calendar. Wide enough to catch the same appointment recorded at a slightly different time,
# narrow enough not to swallow a genuinely different one the same afternoon.
DEDUPE_WINDOW = timedelta(hours=2)

# Why an email produced no proposal. Recorded so the web UI can explain a silence, which is
# otherwise indistinguishable from the pipeline not having run.
SKIP_NO_COMMITMENT = "nessun impegno nell'email"
SKIP_NO_DATE = "impegno dichiarato ma data non risolvibile"
SKIP_LOW_CONFIDENCE = "confidenza troppo bassa"
SKIP_ALREADY_ON_CALENDAR = "già presente in calendario"
SKIP_DUPLICATE = "proposta già esistente per questa email"
SKIP_PAST = "l'impegno è già passato"


@dataclass(slots=True)
class Extraction:
    """What extraction concluded about one email."""

    commitment: Commitment
    when: dates.ResolvedWhen | None
    model: str
    trace_id: str
    confidence: float
    skip_reason: str | None = None

    @property
    def proposable(self) -> bool:
        return self.skip_reason is None and self.when is not None


@dataclass(slots=True)
class ExtractResult:
    examined: int = 0
    proposed: int = 0
    failed: int = 0
    duration_ms: int = 0
    skipped: dict[str, int] = field(default_factory=dict)

    def note_skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def summary(self) -> str:
        if not self.examined:
            return "niente da esaminare"
        parts = [f"{self.examined} esaminate", f"{self.proposed} proposte"]
        if self.failed:
            parts.append(f"{self.failed} fallite")
        for reason, count in sorted(self.skipped.items(), key=lambda kv: -kv[1]):
            parts.append(f"{count} scartate ({reason})")
        return ", ".join(parts) + f" in {self.duration_ms / 1000:.1f}s"


def _effective_confidence(commitment: Commitment, when: dates.ResolvedWhen) -> float:
    """Blend the model's own confidence with how the date was arrived at.

    Agreement between the quoted phrase and the model's ISO guess is independent evidence
    that it read the email rather than pattern-matched it, so it earns a bonus. A date that
    exists only as the model's guess, with no phrase to verify against, is penalised —
    that is the configuration most likely to be a hallucinated date.
    """
    score = commitment.confidence
    if when.source == "agreed":
        score = min(1.0, score + 0.1)
    elif when.source == "model":
        score *= 0.7
    return round(score, 3)


def extract_one(
    *,
    sender_name: str,
    sender_addr: str,
    subject: str,
    body: str,
    received_at: str,
) -> Extraction:
    """Run extraction for a single email and decide whether it is worth proposing."""
    settings = get_settings()
    reference = parse_iso(received_at)
    if reference is None:
        raise ValueError(f"received_at non interpretabile: {received_at!r}")

    llm = get_llm()
    result = llm.structured(
        registry.EXTRACT_COMMITMENT,
        Commitment,
        prompts.extract_user(sender_name, sender_addr, subject, body, format_it(reference)),
        system=prompts.EXTRACT_SYSTEM,
    )
    commitment = result.value

    if not commitment.has_commitment or commitment.kind == "nessuno":
        return Extraction(
            commitment=commitment,
            when=None,
            model=result.model,
            trace_id=result.trace_id,
            confidence=commitment.confidence,
            skip_reason=SKIP_NO_COMMITMENT,
        )

    when = dates.resolve(
        commitment.date_phrase,
        commitment.start_iso,
        reference,
        context=f"{subject} {body[:500]}",
        all_day_hint=True if commitment.all_day else None,
    )

    if when is None:
        # The model claimed a commitment but quoted nothing resolvable. This is the most
        # common failure and it is not worth a second model call: with no usable phrase
        # there is nothing to re-resolve, and inventing a date here is exactly the outcome
        # the design exists to prevent.
        logger.info(
            "Impegno dichiarato ma data non risolvibile: phrase=%r iso=%r",
            commitment.date_phrase,
            commitment.start_iso,
        )
        return Extraction(
            commitment=commitment,
            when=None,
            model=result.model,
            trace_id=result.trace_id,
            confidence=commitment.confidence,
            skip_reason=SKIP_NO_DATE,
        )

    # Plausibility inside dates.resolve() is judged against the *email's* receipt time,
    # which is right for working out what the email meant. Whether it is still worth acting
    # on is a separate question, judged against now.
    #
    # This matters on any backfill. A year-old email announcing a renewal on 15 November
    # 2025 resolves perfectly correctly to 15 November 2025 — and proposing a calendar entry
    # ten months in the past is noise. Observed on the real mailbox: all three of the first
    # proposals it ever made were for dates that had already gone.
    end = parse_iso(when.end_ts)
    if end is not None and end < now_utc():
        return Extraction(
            commitment=commitment,
            when=when,
            model=result.model,
            trace_id=result.trace_id,
            confidence=commitment.confidence,
            skip_reason=SKIP_PAST,
        )

    confidence = _effective_confidence(commitment, when)
    skip = None if confidence >= settings.proposal_confidence_floor else SKIP_LOW_CONFIDENCE

    return Extraction(
        commitment=commitment,
        when=when,
        model=result.model,
        trace_id=result.trace_id,
        confidence=confidence,
        skip_reason=skip,
    )


def _title_for(extraction: Extraction, subject: str) -> str:
    title = (extraction.commitment.title or "").strip()
    # Models sometimes echo the subject line despite being told not to; a subject is still
    # better than an empty calendar entry.
    return title or (subject or "Impegno").strip()[:120]


def _reasoning_for(extraction: Extraction, sender_name: str, sender_addr: str) -> str:
    """Amy's explanation, assembled from facts rather than generated.

    Generated justifications were dropped from triage for inventing details; the same
    applies here. Every clause below is something the pipeline actually knows.
    """
    when = extraction.when
    assert when is not None
    who = sender_name or sender_addr or "un mittente"
    start = parse_iso(when.start_ts)
    parts = [f"{who} indica {format_it(start, with_time=not when.all_day)}"]
    if when.phrase:
        parts.append(f'dal testo "{when.phrase}"')
    if when.source == "agreed":
        parts.append("data confermata due volte")
    elif when.source == "model":
        parts.append("data stimata, nessuna frase esplicita")
    if when.note:
        parts.append(when.note)
    return "; ".join(parts)


def propose(
    *,
    email_id: str,
    subject: str,
    sender_name: str,
    sender_addr: str,
    extraction: Extraction,
) -> int | None:
    """Turn a usable extraction into a proposal row, unless the calendar already has it."""
    when = extraction.when
    if when is None:
        return None

    start = parse_iso(when.start_ts)
    assert start is not None
    title = _title_for(extraction, subject)

    existing = repo.find_similar_event(
        iso_utc(start - DEDUPE_WINDOW), iso_utc(start + DEDUPE_WINDOW), title
    )
    if existing is not None:
        logger.info(
            "Impegno già in calendario come %r (%s); nessuna proposta",
            existing["summary"],
            existing["id"],
        )
        return None

    payload = {
        "kind": extraction.commitment.kind,
        "title": title,
        "start_ts": when.start_ts,
        "end_ts": when.end_ts,
        "all_day": when.all_day,
        "location": extraction.commitment.location,
        "date_phrase": when.phrase,
        "date_source": when.source,
    }

    return repo.create_proposal(
        kind="calendar_event",
        source_type="email",
        source_id=email_id,
        payload=payload,
        reasoning=_reasoning_for(extraction, sender_name, sender_addr),
        evidence_quote=extraction.commitment.evidence,
        confidence=extraction.confidence,
        trace_id=extraction.trace_id,
        # One proposal per source email: being asked twice about the same message is worse
        # than being asked late, and the unique index makes this impossible rather than
        # unlikely.
        dedupe_key=f"email:{email_id}",
    )


def run_extraction(*, limit: int | None = None, dry_run: bool = False) -> ExtractResult:
    """Process the queue of important emails that have not been examined yet."""
    settings = get_settings()
    started = time.perf_counter()
    result = ExtractResult()

    pending = repo.emails_awaiting_extraction(limit or settings.triage_batch_size)
    if not pending:
        return result

    logger.info("Estrazione: %d email importanti da esaminare", len(pending))

    for row in pending:
        result.examined += 1
        try:
            extraction = extract_one(
                sender_name=row["from_name"] or "",
                sender_addr=row["from_addr"] or "",
                subject=row["subject"] or "",
                body=row["body"] or "",
                received_at=row["received_at"],
            )
        except (LLMError, ValueError) as exc:
            result.failed += 1
            logger.warning("Estrazione fallita per %s: %s", row["id"], exc)
            continue

        if dry_run:
            if extraction.skip_reason:
                result.note_skip(extraction.skip_reason)
            else:
                result.proposed += 1
            continue

        if extraction.skip_reason:
            result.note_skip(extraction.skip_reason)
            repo.mark_email_extracted(row["id"], extraction.trace_id)
            continue

        proposal_id = propose(
            email_id=row["id"],
            subject=row["subject"] or "",
            sender_name=row["from_name"] or "",
            sender_addr=row["from_addr"] or "",
            extraction=extraction,
        )
        # Mark examined either way: a duplicate or an already-calendared event is a
        # finished decision, not something to retry every five minutes.
        repo.mark_email_extracted(row["id"], extraction.trace_id)

        if proposal_id is None:
            result.note_skip(SKIP_ALREADY_ON_CALENDAR)
        else:
            result.proposed += 1
            logger.info("Proposta %d creata da email %s", proposal_id, row["id"])

    result.duration_ms = int((time.perf_counter() - started) * 1000)
    logger.info("Estrazione: %s", result.summary())
    return result
