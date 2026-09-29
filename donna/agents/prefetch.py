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
from dataclasses import dataclass
from datetime import timedelta

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


# Day references worth resolving explicitly, with their offset from today.
_DAY_REFS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"\bdopodomani\b", re.I), 2),
    (re.compile(r"\bdomani\b|\bdomattina\b", re.I), 1),
    (re.compile(r"\boggi\b|\bstasera\b|\bstamattina\b|\bstanotte\b", re.I), 0),
]
_WEEKDAY_REF = re.compile(
    r"\b(lunedì|lunedi|martedì|martedi|mercoledì|mercoledi|giovedì|giovedi|venerdì|venerdi|"
    r"sabato|domenica)\b",
    re.I,
)
_WEEKDAY_INDEX = {
    "lunedì": 0, "lunedi": 0, "martedì": 1, "martedi": 1, "mercoledì": 2, "mercoledi": 2,
    "giovedì": 3, "giovedi": 3, "venerdì": 4, "venerdi": 4, "sabato": 5, "domenica": 6,
}


# More than this and the block stops being a fact sheet and becomes the whole week again.
MAX_REFERENCED_DAYS = 4


def referenced_days(message: str) -> list[int]:
    """Offsets in days of every day the message names, in the order they are mentioned.

    Every one, not the first: «martedì dopo il lavoro… e giovedì in piscina» is about two days,
    and stating only Tuesday's agenda leaves Thursday to be guessed — which is exactly the
    failure the agenda block exists to prevent.
    """
    text = message or ""
    found: list[tuple[int, int]] = []          # (position in the message, offset)
    for pattern, offset in _DAY_REFS:
        found.extend((m.start(), offset) for m in pattern.finditer(text))
    if _WEEKDAY_REF.search(text):
        from donna.timeutil import now_local

        weekday_today = now_local().weekday()
        for m in _WEEKDAY_REF.finditer(text):
            ahead = (_WEEKDAY_INDEX[m.group(1).lower()] - weekday_today) % 7
            found.append((m.start(), ahead or 7))

    offsets: list[int] = []
    for _, offset in sorted(found):
        if offset not in offsets:
            offsets.append(offset)
    return offsets[:MAX_REFERENCED_DAYS]


def referenced_day(message: str) -> int | None:
    """The first day the message names, if any."""
    days = referenced_days(message)
    return days[0] if days else None


def day_agenda(offset: int) -> str:
    """That day's schedule, stated plainly.

    Exists because absence is hard for the model to read. Asked to schedule something
    "tomorrow after work" on a day with nothing on it, it asserted that work ended at 16:30 —
    borrowed from the other days — even with "lunedì 21: niente in programma" sitting in the
    week list. A list is something it skims; a dedicated block about the day in question is
    something it answers from.
    """
    from datetime import date as _date

    from donna.store import repo
    from donna.timeutil import day_bounds_utc, format_it, now_local, parse_iso, to_local

    day = _date.today() + timedelta(days=offset)
    start, end = day_bounds_utc(day)
    rows = repo.events_between(start, end)
    label = format_it(now_local() + timedelta(days=offset), with_time=False)

    if not rows:
        return (
            f"## Il giorno di cui sta parlando: {label}\n"
            f"In calendario non c'è NIENTE quel giorno. Nessun lavoro, nessun impegno.\n"
            "Se la sua richiesta presuppone un impegno che quel giorno non esiste "
            "(per esempio \"dopo il lavoro\"), diglielo invece di inventare un orario."
        )

    lines = []
    for row in rows:
        begin, finish = parse_iso(row["start_ts"]), parse_iso(row["end_ts"])
        if row["all_day"]:
            lines.append(f"- {row['summary']} (tutto il giorno)")
        elif begin and finish:
            lines.append(
                f"- {to_local(begin):%H:%M}–{to_local(finish):%H:%M}: {row['summary']}"
            )
    last_end = max(
        (parse_iso(r["end_ts"]) for r in rows if r["end_ts"] and not r["all_day"]),
        default=None,
    )
    tail = (
        f"\nL'ultimo impegno di quel giorno finisce alle {to_local(last_end):%H:%M}."
        if last_end
        else ""
    )
    return f"## Il giorno di cui sta parlando: {label}\n" + "\n".join(lines) + tail


@dataclass(slots=True)
class Prefetched:
    """What was computed, and a plain-language list of what that was.

    The labels are shown in the dashboard, so it is always possible to see exactly which facts
    were handed to the model instead of a bare "precomputed" flag.
    """

    text: str
    parts: list[str]


def for_message(message: str) -> Prefetched | None:
    """A context block of facts read from the calendar in code, or None when nothing applies."""
    blocks: list[str] = []
    parts: list[str] = []

    for offset in referenced_days(message):
        try:
            blocks.append(day_agenda(offset))
            from donna.timeutil import format_it, now_local

            parts.append(f"agenda di {format_it(now_local() + timedelta(days=offset), with_time=False)}")
        except Exception:
            logger.warning("Prefetch dell'agenda del giorno fallito", exc_info=True)

    def _done() -> Prefetched | None:
        return Prefetched("\n\n".join(blocks), parts) if blocks else None

    if not wants_availability(message):
        return _done()

    minutes = parse_duration_minutes(message)
    days = parse_horizon_days(message)
    try:
        slots = toolkit.trova_slot_liberi(durata_minuti=minutes, entro_giorni=days)
    except Exception:
        # A prefetch failure must not fail the turn; the model still has the world state.
        logger.warning("Prefetch degli slot liberi fallito", exc_info=True)
        return _done()

    logger.info("Prefetch slot: %d min entro %d giorni", minutes, days)
    blocks.append(
        f"## Slot liberi calcolati adesso (durata richiesta: {minutes} minuti, "
        f"prossimi {days} giorni)\n"
        "Questi sono calcolati sul calendario reale. Usa ESATTAMENTE questi orari, non "
        "inventarne altri e non aggiungere spiegazioni su cosa c'è prima o dopo.\n\n"
        f"{slots}"
    )
    parts.append(f"slot liberi da {minutes} min nei prossimi {days} giorni")
    return _done()
