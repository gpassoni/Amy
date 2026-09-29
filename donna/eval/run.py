"""Evaluation harness.

Scores triage and extraction against donna/eval/fixtures.py. Two reasons this exists
rather than being a one-off script:

  * prompt changes need a regression number, not an impression. Every prompt edit in this
    project so far has fixed one case and broken another.
  * it exports the same (input, expected, produced) triples to JSONL, which is the dataset
    a LoRA fine-tune would need later.

Model calls make it slow and mildly non-deterministic, so it lives here rather than in the
pytest suite, which stays fast and offline.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from donna.eval.fixtures import CASES, REFERENCE, EmailCase
from donna.llm.client import LLMError
from donna.pipeline import extract, triage
from donna.timeutil import iso_utc, parse_iso, to_local


@dataclass(slots=True)
class CaseOutcome:
    name: str
    passed: bool
    detail: str
    latency_ms: int = 0
    produced: dict = field(default_factory=dict)


@dataclass(slots=True)
class SuiteResult:
    suite: str
    outcomes: list[CaseOutcome] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for o in self.outcomes if o.passed)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def mean_latency_ms(self) -> int:
        if not self.outcomes:
            return 0
        return sum(o.latency_ms for o in self.outcomes) // len(self.outcomes)


def _local(ts: str) -> str:
    dt = parse_iso(ts)
    return f"{to_local(dt):%Y-%m-%d %H:%M}" if dt else "?"


def run_triage_suite(cases: list[EmailCase] | None = None) -> SuiteResult:
    suite = SuiteResult(suite="triage")
    for case in cases or CASES:
        started = time.perf_counter()
        try:
            verdict, model, _ = triage.classify_one(
                sender_name=case.sender_name,
                sender_addr=case.sender_addr,
                subject=case.subject,
                body=case.body,
            )
        except LLMError as exc:
            suite.outcomes.append(CaseOutcome(case.name, False, f"errore: {exc}"))
            continue

        elapsed = int((time.perf_counter() - started) * 1000)
        ok = verdict.signal == case.expect_signal
        detail = (
            f"{verdict.signal} -> {verdict.category}"
            if ok
            else f"atteso {case.expect_signal}, ottenuto {verdict.signal}"
        )
        suite.outcomes.append(
            CaseOutcome(
                case.name,
                ok,
                detail,
                elapsed,
                {"signal": verdict.signal, "category": verdict.category,
                 "confidence": verdict.confidence},
            )
        )
    return suite


def run_extraction_suite(cases: list[EmailCase] | None = None) -> SuiteResult:
    suite = SuiteResult(suite="extraction")
    received = iso_utc(REFERENCE)

    for case in cases or CASES:
        started = time.perf_counter()
        try:
            result = extract.extract_one(
                sender_name=case.sender_name,
                sender_addr=case.sender_addr,
                subject=case.subject,
                body=case.body,
                received_at=received,
            )
        except (LLMError, ValueError) as exc:
            suite.outcomes.append(CaseOutcome(case.name, False, f"errore: {exc}"))
            continue

        elapsed = int((time.perf_counter() - started) * 1000)
        produced: dict = {
            "has_commitment": result.commitment.has_commitment,
            "kind": result.commitment.kind,
            "title": result.commitment.title,
            "date_phrase": result.commitment.date_phrase,
            "confidence": result.confidence,
            "skip_reason": result.skip_reason,
        }

        if not case.expect_commitment:
            # Expected nothing. Anything that would reach the user is a false positive, and
            # those are the expensive mistake: a wrong proposal trains the user to ignore
            # proposals.
            ok = not result.proposable
            detail = "nessuna proposta, corretto" if ok else (
                f"FALSO POSITIVO: {result.commitment.title!r} il "
                f"{_local(result.when.start_ts) if result.when else '?'}"
            )
            suite.outcomes.append(CaseOutcome(case.name, ok, detail, elapsed, produced))
            continue

        if not result.proposable:
            suite.outcomes.append(
                CaseOutcome(
                    case.name,
                    False,
                    f"MANCATO: {result.skip_reason} (phrase={result.commitment.date_phrase!r})",
                    elapsed,
                    produced,
                )
            )
            continue

        assert result.when is not None
        produced["start"] = _local(result.when.start_ts)
        produced["all_day"] = result.when.all_day
        produced["date_source"] = result.when.source

        expected = case.expect_start or ""
        got = produced["start"]
        # An all-day expectation is written as a bare date.
        date_only = len(expected) == 10
        date_ok = got.startswith(expected) if date_only else got == expected

        title = (result.commitment.title or "").lower()
        title_ok = (
            case.expect_title_contains is None
            or case.expect_title_contains.lower() in title
        )

        ok = date_ok and title_ok
        problems = []
        if not date_ok:
            problems.append(f"data: atteso {expected}, ottenuto {got}")
        if not title_ok:
            problems.append(f"titolo: atteso contenga {case.expect_title_contains!r}, ottenuto {title!r}")
        detail = f"{got} · {result.commitment.title}" if ok else "; ".join(problems)

        suite.outcomes.append(CaseOutcome(case.name, ok, detail, elapsed, produced))

    return suite


def export_jsonl(suites: list[SuiteResult], path: Path) -> int:
    """Write every case outcome as JSONL, for later fine-tuning or offline analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for suite in suites:
            for outcome in suite.outcomes:
                handle.write(
                    json.dumps({"suite": suite.suite, **asdict(outcome)}, ensure_ascii=False) + "\n"
                )
                written += 1
    return written
