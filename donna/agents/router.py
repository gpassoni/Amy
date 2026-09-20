"""Intent routing.

Two stages, cheapest first.

**A deterministic pre-router**, which handles the phrasings that actually recur — "cosa ho
oggi", "/briefing", "accetta 3" — with regexes. These cost microseconds and are exactly
right, where the model costs a second and is occasionally wrong. On a local setup the LLM hop
is the whole latency budget for a short question, so skipping it is not a micro-optimisation.

**The model**, for everything else, constrained to the intent enum by the output schema so an
invented intent is unrepresentable.

When the model is unsure, routing falls back to `chat`, which holds every tool. Degrading to
the general agent is always safe; degrading to a narrow one is not.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from donna.llm import registry
from donna.llm.client import LLMError, get_llm
from donna.pipeline.schemas import RouterDecision

logger = logging.getLogger(__name__)

# Below this the model's own answer is not trusted and `chat` takes it.
CONFIDENCE_FLOOR = 0.45

ROUTER_SYSTEM = """Classifica l'intento del messaggio. Una sola etichetta.

schedule_query   vuole sapere cosa ha in programma, quando è libero, quando è qualcosa
schedule_mutate  vuole creare, spostare o cancellare un impegno
inbox_query      vuole sapere della posta: cosa è arrivato, da chi, riassunti
task_query       vuole sapere le cose da fare
task_mutate      vuole aggiungere o completare una cosa da fare
proposal_action  risponde a una proposta che Donna ha fatto (accetta, rifiuta, chiede perché)
briefing         vuole il punto della situazione: la giornata, la settimana, "come siamo"
smalltalk        chiacchiera, ringraziamenti, domande su di lei, nulla di operativo
multi            più cose insieme che ricadono in categorie diverse

ESEMPI:
"cosa ho domani pomeriggio" -> schedule_query
"quando sono libero due ore questa settimana" -> schedule_query
"metti palestra domani alle 19" -> schedule_mutate
"sposta il dentista a venerdì" -> schedule_mutate
"ci sono email importanti?" -> inbox_query
"cosa mi ha scritto Marco" -> inbox_query
"cosa devo fare" -> task_query
"ricordami di chiamare il meccanico" -> task_mutate
"segna fatto lampadine" -> task_mutate
"sì metti in calendario la 3" -> proposal_action
"perché me l'hai proposto" -> proposal_action
"come va oggi" -> briefing
"fammi il punto della settimana" -> briefing
"grazie sei la migliore" -> smalltalk
"sposta il dentista e segna fatto le lampadine" -> multi

Rispondi SOLO con il JSON."""


@dataclass(slots=True)
class Route:
    intent: str
    confidence: float
    via: str          # "regex" | "model" | "fallback"
    trace_id: str | None = None


# Ordered: the first match wins, so put the specific before the general.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "proposal_action",
        re.compile(
            r"\b(accett\w*|conferm\w*|rifiut\w*|scart\w*|ignor\w*)\s*(la\s*)?(proposta\s*)?#?\d+\b"
            r"|\bproposte?\b|\bperché me l'hai proposto\b",
            re.I,
        ),
    ),
    (
        "briefing",
        re.compile(
            r"^\s*/?(briefing|punto|riepilogo|situazione)\b"
            r"|\b(fammi|dammi) il punto\b|\bcome (va|siamo)\b|\briassunto della (giornata|settimana)\b",
            re.I,
        ),
    ),
    (
        "schedule_query",
        re.compile(
            r"\bcosa (ho|c'è|abbiamo)\b.*\b(oggi|domani|dopodomani|settimana|lunedì|martedì|"
            r"mercoledì|giovedì|venerdì|sabato|domenica)\b"
            r"|\b(sono|sarò) liber[oa]\b|\bquando (sono|posso)\b|\bche impegni\b"
            r"|\bprossimi impegni\b|\bagenda\b",
            re.I,
        ),
    ),
    (
        "task_query",
        re.compile(r"\bcosa devo fare\b|\b(mie )?task\b|\bcose da fare\b|\blista (della )?spesa\b", re.I),
    ),
    (
        "inbox_query",
        re.compile(r"\b(email|mail|posta|messaggi)\b", re.I),
    ),
    (
        "smalltalk",
        re.compile(r"^\s*(grazie|ciao|buongiorno|buonasera|sei (la migliore|un genio)|ok)\b\W*$", re.I),
    ),
]

# Commands are unambiguous by construction.
_COMMANDS = {
    "/briefing": "briefing",
    "/proposte": "proposal_action",
    "/agenda": "schedule_query",
    "/posta": "inbox_query",
    "/task": "task_query",
}


def pre_route(message: str) -> Route | None:
    """Deterministic routing for the phrasings that recur. None means ask the model."""
    text = message.strip()
    if not text:
        return None

    command = text.split()[0].lower()
    if command in _COMMANDS:
        return Route(_COMMANDS[command], 1.0, "regex")

    # A mutation verb plus a time expression is a calendar write, and it is worth catching
    # before the generic schedule_query pattern below.
    if re.search(r"\b(metti|aggiungi|segna|prenota|fissa|sposta|cancella|elimina)\b", text, re.I):
        if re.search(r"\b(ricordami|task|da fare|lista)\b", text, re.I):
            return Route("task_mutate", 0.9, "regex")
        if re.search(
            r"\b(alle|ore|domani|oggi|dopodomani|luned|marted|mercoled|gioved|venerd|sabato|domenica|\d{1,2}[:/]\d{2})\b",
            text,
            re.I,
        ):
            return Route("schedule_mutate", 0.9, "regex")

    for intent, pattern in _PATTERNS:
        if pattern.search(text):
            return Route(intent, 0.9, "regex")

    return None


def route(message: str) -> Route:
    """Pick an intent for a message."""
    quick = pre_route(message)
    if quick is not None:
        logger.info("Router (regex): %r -> %s", message[:60], quick.intent)
        return quick

    try:
        result = get_llm().structured(
            registry.ROUTE, RouterDecision, message, system=ROUTER_SYSTEM
        )
    except LLMError as exc:
        logger.warning("Router failed (%s); falling back to chat", exc)
        return Route("smalltalk", 0.0, "fallback")

    decision = result.value
    if decision.confidence < CONFIDENCE_FLOOR:
        logger.info(
            "Router unsure (%s at %.2f); falling back to chat", decision.intent, decision.confidence
        )
        return Route("multi", decision.confidence, "fallback", result.trace_id)

    logger.info(
        "Router (model): %r -> %s (%.2f, %d ms)",
        message[:60],
        decision.intent,
        decision.confidence,
        result.latency_ms,
    )
    return Route(decision.intent, decision.confidence, "model", result.trace_id)
