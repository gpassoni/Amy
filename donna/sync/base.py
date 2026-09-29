"""Shared shape for sync results.

Every sync reports the same counters so the CLI, the scheduler log and the web dashboard
can render any of them identically.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SyncResult:
    resource: str
    mode: str = "incremental"  # incremental | full
    created: int = 0
    updated: int = 0
    deleted: int = 0
    skipped: int = 0
    duration_ms: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def touched(self) -> int:
        return self.created + self.updated + self.deleted

    def summary(self) -> str:
        if self.error:
            return f"{self.resource}: ERRORE — {self.error}"
        parts = [f"{self.resource} ({self.mode})"]
        if self.created:
            parts.append(f"{self.created} nuovi")
        if self.updated:
            parts.append(f"{self.updated} aggiornati")
        if self.deleted:
            parts.append(f"{self.deleted} rimossi")
        if self.skipped:
            parts.append(f"{self.skipped} saltati")
        if self.touched == 0:
            parts.append("nessuna novità")
        return ", ".join(parts) + f" in {self.duration_ms / 1000:.1f}s"
