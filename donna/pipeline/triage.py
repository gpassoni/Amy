"""Email triage: classify, then label.

Runs on the worker model on CPU, so it never competes with the conversational model for
VRAM and a backlog can grind away while Donna stays responsive in chat.

Ordering note: the classification is written to the database *before* the Gmail label is
applied. If labelling fails (quota, network), the verdict is still recorded and
`label_applied` stays 0, so the work is not repeated — only the labelling is retried.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from donna.config import get_settings
from donna.google import gmail
from donna.llm import registry
from donna.llm.client import LLMError, get_llm
from donna.pipeline import prompts
from donna.pipeline.schemas import Classification
from donna.store import repo

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TriageResult:
    classified: int = 0
    labelled: int = 0
    failed: int = 0
    duration_ms: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    by_signal: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        if not self.classified and not self.failed:
            return "niente da classificare"
        spread = ", ".join(f"{k}: {v}" for k, v in sorted(self.by_category.items()))
        parts = [f"{self.classified} classificate ({spread})"]
        if self.labelled:
            parts.append(f"{self.labelled} etichettate su Gmail")
        if self.failed:
            parts.append(f"{self.failed} fallite")
        return ", ".join(parts) + f" in {self.duration_ms / 1000:.1f}s"


def classify_one(
    *, sender_name: str, sender_addr: str, subject: str, body: str
) -> tuple[Classification, str, str]:
    """Classify a single email. Returns (verdict, model, trace_id)."""
    llm = get_llm()
    result = llm.structured(
        registry.CLASSIFY_EMAIL,
        Classification,
        prompts.classify_user(sender_name, sender_addr, subject, body),
        system=prompts.CLASSIFY_SYSTEM,
    )
    return result.value, result.model, result.trace_id


def run_triage(
    *, limit: int | None = None, dry_run: bool = False, apply_labels: bool = True
) -> TriageResult:
    """Classify the pending queue.

    dry_run still calls the model — the point is to inspect its judgement without writing
    anything to the database or to Gmail.
    """
    settings = get_settings()
    started = time.perf_counter()
    result = TriageResult()

    pending = repo.untriaged_emails(limit or settings.triage_batch_size)
    if not pending:
        return result

    logger.info("Triage: %d email da classificare", len(pending))

    for row in pending:
        try:
            verdict, model, trace_id = classify_one(
                sender_name=row["from_name"] or "",
                sender_addr=row["from_addr"] or "",
                subject=row["subject"] or "",
                body=row["body"] or "",
            )
        except LLMError as exc:
            result.failed += 1
            logger.warning("Triage failed for %s: %s", row["id"], exc)
            continue

        result.classified += 1
        result.by_category[verdict.category] = result.by_category.get(verdict.category, 0) + 1
        result.by_signal[verdict.signal] = result.by_signal.get(verdict.signal, 0) + 1

        if dry_run:
            logger.info(
                "[dry-run] %s -> %s (%.2f) %s",
                (row["subject"] or "")[:50],
                verdict.category,
                verdict.confidence,
                verdict.reason,
            )
            continue

        labelled = False
        # Record first, label second: a labelling failure must not cost us the verdict.
        repo.set_email_category(
            row["id"],
            category=verdict.category,
            confidence=verdict.confidence,
            reason=verdict.reason,
            model=model,
            trace_id=trace_id,
            label_applied=False,
            signal=verdict.signal,
        )
        if apply_labels:
            labelled = gmail.apply_category_label(row["id"], verdict.category)
            if labelled:
                result.labelled += 1
                repo.set_email_category(
                    row["id"],
                    category=verdict.category,
                    confidence=verdict.confidence,
                    reason=verdict.reason,
                    model=model,
                    trace_id=trace_id,
                    label_applied=True,
                    signal=verdict.signal,
                )

    result.duration_ms = int((time.perf_counter() - started) * 1000)
    logger.info("Triage: %s", result.summary())
    return result


def reclassify(email_id: str, category: str, *, note: str | None = None) -> bool:
    """Override a verdict, recording the correction as training data.

    This is the feedback loop: every disagreement between Donna and the user becomes a row
    the eval harness scores against and a future fine-tune can learn from.
    """
    row = repo.get_email(email_id)
    if row is None:
        return False

    repo.record_feedback(
        kind="reclassify",
        trace_id=row["classify_trace_id"],
        email_id=email_id,
        original_output=row["category"],
        corrected_output=category,
        note=note,
    )
    repo.set_email_category(
        email_id,
        category=category,
        confidence=1.0,  # a human said so
        reason=note or "corretto manualmente",
        model="human",
        trace_id=row["classify_trace_id"],
        label_applied=gmail.apply_category_label(email_id, category),
    )
    return True
