"""Answering the question before the model is asked it.

Some questions have an answer that is pure computation over the mirror, and asking a model to
*decide to compute it* is a bet that does not pay. The free-time question is the clearest case.
Told twice, in increasingly explicit prompt text, to call `trova_slot_liberi` for availability
questions, the 9b instead answered from imagination — and once the calendar had real events in
it, the invented slots were confidently wrong:

    "Lunedì 21: dalle 14:30 alle 16:30 (dopo il colloquio HR)"   <- no such gap existed

A wrong calendar answer costs more than a slow one. So when a message looks like a free-time
question, the slots are computed here, in code, and handed to the model as fact. The model's
job shrinks to phrasing, which is what it is good at.

This is the same division as everywhere else in Donna: the model reads and writes language,
the code does arithmetic and owns the truth.
"""
from __future__ import annotations

import logging
import re

from donna.agents import tools as toolkit

logger = logging.getLogger(__name__)

# "quando sono libero", "trovami due ore", "ho tempo giovedì", "quando posso"
_AVAILABILITY = re.compile(
    r"\b(quando\s+(?:sono|sarò|posso|riesco|ho))\b"
    r"|\b(?:sono|sarò)\s+liber[oa]\b"
    r"|\btrovami\b|\bho\s+tempo\b|\bspazio\s+(?:in|per)\b|\bslot\b"
    r"|\bquando\s+incastr\w+\b|\bdove\s+(?:la|lo)\s+metto\b",
    re.I,
)

# Duration, spelled or numeric: "due ore", "un'ora", "30 minuti", "mezz'ora", "1h".
_WORD_HOURS = {
    "un": 1, "una": 1, "un'": 1, "due": 2, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6,
}
_DURATION = re.compile(
    r"\b(un'?|una|due|tre|quattro|cinque|sei|\d{1,2})\s*(or[ae]|h\b|minut[oi]|min\b)", re.I
)
_HALF_HOUR = re.compile(r"\bmezz'?ora\b", re.I)

# Horizon words, so "questa settimana" and "oggi" do not both search ten days.
_HORIZONS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"\boggi\b|\bstasera\b|\bstamattina\b", re.I), 1),
    (re.compile(r"\bdomani\b", re.I), 2),
    (re.compile(r"\bdopodomani\b", re.I), 3),
    (re.compile(r"\bquesta settimana\b|\bin settimana\b", re.I), 7),
    (re.compile(r"\bprossima settimana\b", re.I), 14),
    (re.compile(r"\bquesto mese\b|\bnelle prossime settimane\b", re.I), 30),
]

DEFAULT_DURATION_MINUTES = 60
DEFAULT_HORIZON_DAYS = 7


def wants_availability(message: str) -> bool:
    return bool(_AVAILABILITY.search(message or ""))


def parse_duration_minutes(message: str) -> int:
    if _HALF_HOUR.search(message or ""):
        return 30
    match = _DURATION.search(message or "")
    if not match:
        return DEFAULT_DURATION_MINUTES
    amount_raw, unit = match.group(1).lower(), match.group(2).lower()
    amount = _WORD_HOURS.get(amount_raw.rstrip("'"), 0)
    if not amount:
        try:
            amount = int(amount_raw)
        except ValueError:
            return DEFAULT_DURATION_MINUTES
    return amount if unit.startswith(("minut", "min")) else amount * 60


def parse_horizon_days(message: str) -> int:
    for pattern, days in _HORIZONS:
        if pattern.search(message or ""):
            return days
    return DEFAULT_HORIZON_DAYS


def for_message(message: str) -> str | None:
    """A context block of precomputed facts, or None when nothing applies."""
    if not wants_availability(message):
        return None

    minutes = parse_duration_minutes(message)
    days = parse_horizon_days(message)
    try:
        slots = toolkit.trova_slot_liberi(durata_minuti=minutes, entro_giorni=days)
    except Exception:
        # A prefetch failure must not fail the turn; the model still has the world state.
        logger.warning("Prefetch degli slot liberi fallito", exc_info=True)
        return None

    logger.info("Prefetch slot: %d min entro %d giorni", minutes, days)
    return (
        f"## Slot liberi calcolati adesso (durata richiesta: {minutes} minuti, "
        f"prossimi {days} giorni)\n"
        "Questi sono calcolati sul calendario reale. Usa ESATTAMENTE questi orari, non "
        "inventarne altri e non aggiungere spiegazioni su cosa c'è prima o dopo.\n\n"
        f"{slots}"
    )
