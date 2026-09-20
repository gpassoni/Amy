"""Semantic memory: durable facts about the user.

Three layers of memory exist in Donna, and this is only the third:

  1. the mirror — her calendar, inbox and tasks, which is structured state
  2. the conversation — the last few turns plus a rolling summary
  3. facts — things that are true about the user and are not in any Google API:
     "I train Tuesday and Thursday evenings", "I hate meetings before 10"

Retrieval is brute-force cosine similarity in Python. With a few hundred facts that costs
under a millisecond, and sqlite-vec would be a painful Windows/conda dependency bought for
nothing. If the table ever reaches tens of thousands of rows, that is the upgrade.

Facts marked `always_on` bypass retrieval entirely — a standing constraint should not have to
be semantically similar to the question to apply.
"""
from __future__ import annotations

import logging
import struct

from donna.llm import registry
from donna.llm.client import LLMError, get_llm
from donna.pipeline.schemas import ExtractedFacts
from donna.store.db import get_db
from donna.timeutil import iso_utc, now_utc

logger = logging.getLogger(__name__)

# Below this, a "match" is noise. mxbai-embed-large scores unrelated Italian sentences
# around 0.54, so the floor has to sit meaningfully above that.
SIMILARITY_FLOOR = 0.62

EXTRACT_SYSTEM = """Estrai SOLO fatti durevoli su Gabriele dal messaggio.

Un fatto durevole vale ancora fra sei mesi: abitudini, preferenze, vincoli, relazioni,
luoghi, ruoli.

SÌ: "vado in palestra il martedì e il giovedì sera", "non voglio riunioni prima delle 10",
    "mia sorella si chiama Chiara", "lavoro da remoto il venerdì"
NO: "cosa ho domani", "grazie", "sposta l'appuntamento" — sono richieste, non fatti.
NO: eventi singoli già in calendario.

Scrivi ogni fatto in terza persona, breve e autonomo. Se non c'è nessun fatto durevole,
restituisci una lista vuota. È il caso più comune."""


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / ((na**0.5) * (nb**0.5))


def remember(
    text: str,
    *,
    kind: str = "preference",
    source: str | None = None,
    confidence: float = 0.8,
    always_on: bool = False,
) -> int | None:
    """Store a fact with its embedding. Returns None if it is already known.

    Embedding failure is not fatal: the fact is stored without a vector, so it is still
    visible in the web UI and still exportable, it just cannot be recalled semantically until
    re-embedded.
    """
    text = text.strip()
    if not text:
        return None

    db = get_db()
    existing = db.query_one("SELECT id FROM facts WHERE text = ?", (text,))
    if existing is not None:
        db.execute(
            "UPDATE facts SET last_seen_at = ? WHERE id = ?", (iso_utc(now_utc()), existing["id"])
        )
        return None

    embedding = None
    model = None
    try:
        embedding = _pack(get_llm().embed_one(text))
        model = registry.get_spec(registry.EMBED).model
    except LLMError:
        logger.warning("Could not embed fact %r; stored without a vector", text[:60])

    now = iso_utc(now_utc())
    cursor = db.execute(
        "INSERT INTO facts (text, kind, source, confidence, always_on, embedding,"
        " embedding_model, created_at, last_seen_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (text, kind, source, confidence, int(always_on), embedding, model, now, now),
    )
    logger.info("Nuovo fatto ricordato: %s", text[:80])
    return cursor.lastrowid


def forget(fact_id: int) -> bool:
    return get_db().execute("DELETE FROM facts WHERE id = ?", (fact_id,)).rowcount > 0


def all_facts(limit: int = 200) -> list:
    return get_db().query(
        "SELECT * FROM facts WHERE superseded_by IS NULL ORDER BY always_on DESC, id DESC LIMIT ?",
        (limit,),
    )


def recall(query: str | None, *, limit: int = 8) -> list[str]:
    """Always-on facts first, then the closest matches to `query`."""
    db = get_db()
    rows = db.query(
        "SELECT id, text, always_on, embedding FROM facts WHERE superseded_by IS NULL"
    )
    if not rows:
        return []

    standing = [r["text"] for r in rows if r["always_on"]]
    remaining = limit - len(standing)
    if remaining <= 0 or not query or not query.strip():
        return standing[:limit]

    candidates = [r for r in rows if not r["always_on"] and r["embedding"]]
    if not candidates:
        return standing

    try:
        target = get_llm().embed_one(query)
    except LLMError:
        logger.warning("Could not embed the query; returning standing facts only")
        return standing

    scored = [(_cosine(target, _unpack(r["embedding"])), r["text"]) for r in candidates]
    scored.sort(reverse=True)
    matched = [text for score, text in scored[:remaining] if score >= SIMILARITY_FLOOR]
    return standing + matched


def learn_from(message: str, *, source: str | None = None) -> list[str]:
    """Extract durable facts from something the user said, and store them.

    Runs after a reply is sent, not before, so it never adds latency to the turn the user is
    waiting on.
    """
    if len(message.strip()) < 12:
        return []

    try:
        result = get_llm().structured(
            registry.EXTRACT_FACTS, ExtractedFacts, message, system=EXTRACT_SYSTEM
        )
    except LLMError:
        logger.warning("Fact extraction failed", exc_info=True)
        return []

    stored: list[str] = []
    for fact in result.value.facts:
        if remember(fact, source=source, confidence=0.7) is not None:
            stored.append(fact)
    return stored
