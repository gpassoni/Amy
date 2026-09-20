"""Call tracing.

Every LLM call lands in the traces table. This is not observability for its own sake: the
web UI has to be able to answer "why did she suggest this?", which means the exact prompt,
the exact raw response, which model produced it and how long it took must be recoverable
after the fact. Proposals and emails store a trace_id pointing here.

Tracing never breaks a call: if the database write fails, the failure is logged and the
call proceeds.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from donna.store.db import get_db
from donna.timeutil import now_utc

logger = logging.getLogger(__name__)


def new_trace_id() -> str:
    return uuid.uuid4().hex


@dataclass(slots=True)
class Trace:
    task: str
    model: str
    device: str
    id: str = field(default_factory=new_trace_id)
    parent_id: str | None = None
    system_prompt: str | None = None
    prompt: str | None = None
    output: str | None = None
    schema_name: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: int | None = None
    load_ms: int | None = None
    ok: bool = True
    error: str | None = None

    def save(self) -> str:
        try:
            get_db().execute(
                "INSERT OR REPLACE INTO traces (id, parent_id, task, model, device,"
                " system_prompt, prompt, output, schema_name, tokens_in, tokens_out,"
                " latency_ms, load_ms, ok, error, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    self.id,
                    self.parent_id,
                    self.task,
                    self.model,
                    self.device,
                    self.system_prompt,
                    self.prompt,
                    self.output,
                    self.schema_name,
                    self.tokens_in,
                    self.tokens_out,
                    self.latency_ms,
                    self.load_ms,
                    1 if self.ok else 0,
                    self.error,
                    now_utc().isoformat(timespec="seconds"),
                ),
            )
        except Exception:  # tracing must never take down a call
            logger.exception("Could not persist trace %s (task=%s)", self.id, self.task)
        return self.id


def dump_messages(messages: list[dict[str, Any]]) -> str:
    try:
        return json.dumps(messages, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(messages)
