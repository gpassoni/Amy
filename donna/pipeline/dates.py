"""Resolving dates out of text.

The rule this module enforces: **the model extracts, the code decides**.

A 2B model asked to compute "martedì prossimo alle 15" into an ISO timestamp will produce
something plausible and often wrong — it has no reliable notion of today's date, weekday
arithmetic, or DST. So the extractor is asked for two things instead: the literal phrase it
saw, and its own guess. This module re-resolves the phrase with dateparser against the
email's own receipt time, and treats the model's guess only as a cross-check.

When the two agree, confidence goes up. When they disagree, the code wins.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import dateparser

from donna.config import get_settings
from donna.timeutil import iso_utc, local_zone, parse_iso, to_local

logger = logging.getLogger(__name__)

# A resolved date this far from the email is almost certainly a misparse (a phone number
# read as a year, a footer date, a "since 1998" in a signature).
MAX_PAST = timedelta(days=2)
MAX_FUTURE = timedelta(days=550)

DEFAULT_DURATION_MINUTES = 60

# Phrases that indicate a whole-day commitment rather than an appointment.
_ALL_DAY_HINTS = re.compile(
    r"\b(tutto il giorno|giornata intera|all day|ferie|vacanz\w+|compleann\w+)\b", re.I
)

# Explicit durations Donna can honour: "per due ore", "di 30 minuti", "dalle 9 alle 11".
_DURATION_RE = re.compile(
    r"\b(?:per|di)\s+(?:(un|una|due|tre|quattro)|(\d{1,2}))\s*(or[ae]|minut[oi])\b", re.I
)
_RANGE_RE = re.compile(
    r"\bdalle\s+(\d{1,2})(?:[:.](\d{2}))?\s+alle\s+(\d{1,2})(?:[:.](\d{2}))?", re.I
)
_WORD_NUMBERS = {"un": 1, "una": 1, "due": 2, "tre": 3, "quattro": 4}


@dataclass(slots=True)
class ResolvedWhen:
    start_ts: str
    end_ts: str
    all_day: bool
    # How the start was arrived at, for the "why" view and for scoring the extractor.
    source: str  # phrase | model | agreed
    agreed: bool
    phrase: str | None = None
    note: str | None = None


# --- Italian phrase normalisation --------------------------------------------
# dateparser needs a clean date expression. Real extracted phrases carry filler that makes
# it fail outright or, worse, misparse. All of these were observed:
#
#   "24/09/2026 ore 15:00"            -> 14 December 18:00   (the word "ore" derails it)
#   "il 15 dicembre"                  -> None                (leading article)
#   "lunedì 28 settembre per due ore" -> None                (trailing duration)
#   "venerdì dalle 14 alle 16"        -> None                (range)
#   "martedì prossimo alle 15"        -> None                ("prossimo")
#   "dopodomani"                      -> None                (not in its vocabulary)
#
# So the phrase is normalised, and the constructs dateparser cannot do are resolved here
# arithmetically instead. This is the same principle as the module docstring: the model
# extracts, the code decides.

_WEEKDAYS_IT = {
    "lunedì": 0, "lunedi": 0, "martedì": 1, "martedi": 1, "mercoledì": 2, "mercoledi": 2,
    "giovedì": 3, "giovedi": 3, "venerdì": 4, "venerdi": 4, "sabato": 5, "domenica": 6,
}
_RELATIVE_DAYS = {
    "oggi": 0, "stasera": 0, "stamattina": 0, "stamane": 0,
    "domani": 1, "domattina": 1,
    "dopodomani": 2,
    "ieri": -1,
}

_LEADING_FILLER = re.compile(
    r"^\s*(?:in\s+data|entro\s+il|entro|il|lo|la|l'|nel|nella|del|dello|della|di|a|per|su)\s+",
    re.I,
)
_TRAILING_DURATION = re.compile(r"\s*\b(?:per|di)\s+(?:un|una|due|tre|quattro|\d{1,2})\s*(?:or[ae]|minut[oi])\b.*$", re.I)
_ALL_DAY_PREFIX = re.compile(r"^\s*(?:tutto il giorno|per tutta la giornata|giornata intera)\s*(?:di|del|dello|della)?\s*", re.I)
_TIME_RE = re.compile(r"\b(?:alle|ore|h|at)?\s*(\d{1,2})(?:[:.](\d{2}))?\s*(?:alle|-)?", re.I)
_EXPLICIT_TIME_RE = re.compile(r"\b(?:alle|ore|h)\s*(\d{1,2})(?:[:.](\d{2}))?|\b(\d{1,2})[:.](\d{2})\b", re.I)
_NEXT_RE = re.compile(r"\b(prossim[oa]|che viene|venturo)\b", re.I)
_LAST_RE = re.compile(r"\b(scors[oa]|passat[oa])\b", re.I)


# Any way a clock time can be written, so it can be removed once read. Leaving it in is
# what makes dateparser fail on "24 settembre alle 15" and misparse "24/09/2026 ore 15:00".
_TIME_EXPR_RE = re.compile(
    r"\b(?:dalle|alle|ore|h|at)\s*\d{1,2}(?:[:.]\d{2})?|\b\d{1,2}[:.]\d{2}\b", re.I
)


def _extract_time(phrase: str) -> tuple[int, int] | None:
    """The stated clock time, if any.

    A range is checked first and yields its *start*: "dalle 14 alle 16" begins at 14. Left
    to the general pattern it matched 16, because \\balle does not match inside "dalle" and
    so the first time was skipped and the second one taken.
    """
    span = _RANGE_RE.search(phrase)
    if span:
        hour, minute = int(span.group(1)), int(span.group(2) or 0)
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour, minute

    match = _EXPLICIT_TIME_RE.search(phrase)
    if not match:
        return None
    hour = match.group(1) or match.group(3)
    minute = match.group(2) or match.group(4)
    if hour is None:
        return None
    h, m = int(hour), int(minute or 0)
    return (h, m) if 0 <= h <= 23 and 0 <= m <= 59 else None


def _strip_time(text: str) -> str:
    """Remove clock times, leaving only the date expression."""
    return re.sub(r"\s+", " ", _TIME_EXPR_RE.sub(" ", text)).strip(" ,.;:-")


# Abbreviated month names, which real email is full of ("15 nov 2025", "3 ott"). dateparser
# does not resolve these in Italian, so the phrase fell through to the model's own ISO guess
# — the least trustworthy path, and the one the whole design exists to avoid relying on.
_MONTH_ABBREV = {
    "gen": "gennaio", "genn": "gennaio", "feb": "febbraio", "febb": "febbraio",
    "mar": "marzo", "apr": "aprile", "mag": "maggio", "giu": "giugno",
    "lug": "luglio", "ago": "agosto", "set": "settembre", "sett": "settembre",
    "ott": "ottobre", "nov": "novembre", "dic": "dicembre",
}
_MONTH_ABBREV_RE = re.compile(
    r"\b(" + "|".join(sorted(_MONTH_ABBREV, key=len, reverse=True)) + r")\.?\b", re.I
)


def _expand_months(text: str) -> str:
    return _MONTH_ABBREV_RE.sub(lambda m: _MONTH_ABBREV[m.group(1).lower()], text)


def _clean_for_parser(phrase: str) -> str:
    """Strip everything that is not a date, leaving a bare expression for dateparser.

    The time has already been read by _extract_time and is reapplied afterwards, so it is
    removed here rather than left to confuse the parser.
    """
    text = _ALL_DAY_PREFIX.sub("", phrase.strip())
    text = _TRAILING_DURATION.sub("", text)
    text = _NEXT_RE.sub("", text)
    text = _LAST_RE.sub("", text)
    text = _strip_time(text)
    text = _expand_months(text)
    while True:
        stripped = _LEADING_FILLER.sub("", text)
        if stripped == text:
            break
        text = stripped
    return re.sub(r"\s+", " ", text).strip(" ,.;:-")


def _resolve_day_arithmetically(phrase: str, reference: datetime) -> datetime | None:
    """Handle the Italian relative-day and weekday forms dateparser cannot.

    Weekday resolution deliberately looks forward: an email saying "venerdì" means the
    coming Friday, and "venerdì prossimo" means the one after that when today is already
    in that week.
    """
    lowered = phrase.lower()
    base = to_local(reference)

    for word, offset in _RELATIVE_DAYS.items():
        if re.search(rf"\b{word}\b", lowered):
            return base + timedelta(days=offset)

    for name, weekday in _WEEKDAYS_IT.items():
        if not re.search(rf"\b{name}\b", lowered):
            continue
        # A day-of-month in the same phrase is more specific, so let dateparser own it —
        # but only a real one. The clock time has to come out first, or "martedì alle 15"
        # looks like it carries a numeric day and falls through to a parser that fails.
        if re.search(r"\b\d{1,2}\b", _strip_time(lowered)):
            return None
        ahead = (weekday - base.weekday()) % 7
        if ahead == 0:
            ahead = 7  # "martedì" said on a Tuesday means the next one
        if _NEXT_RE.search(lowered) and ahead < 7:
            ahead += 7
        if _LAST_RE.search(lowered):
            return base - timedelta(days=(base.weekday() - weekday) % 7 or 7)
        return base + timedelta(days=ahead)

    return None


def _parse_phrase(phrase: str, reference: datetime) -> datetime | None:
    """Resolve a phrase to a local datetime.

    Day and time are resolved separately, because they fail independently: dateparser is
    good at "3 ottobre" and bad at "dopodomani", while a clock time is a trivial regex.
    """
    if not phrase or not phrase.strip():
        return None

    time_of_day = _extract_time(phrase)
    day = _resolve_day_arithmetically(phrase, reference)

    if day is None:
        cleaned = _clean_for_parser(phrase)
        if not cleaned:
            return None
        parsed = dateparser.parse(
            cleaned,
            languages=["it", "en"],
            settings={
                # Anchored to the email's own receipt time, not to now, so reprocessing an
                # old message still yields the date it meant when it was written.
                "RELATIVE_BASE": to_local(reference).replace(tzinfo=None),
                "PREFER_DATES_FROM": "future",
                "TIMEZONE": get_settings().calendar_timezone,
                "RETURN_AS_TIMEZONE_AWARE": True,
                "DATE_ORDER": "DMY",  # 24/09 is 24 September, never 9 April
            },
        )
        if parsed is None:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=local_zone())
        day = to_local(parsed)
        # dateparser already found a time; only override when we found an explicit one.
        if time_of_day is None:
            return day

    hour, minute = time_of_day if time_of_day else (0, 0)
    return day.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _plausible(candidate: datetime, reference: datetime) -> bool:
    return (reference - MAX_PAST) <= candidate <= (reference + MAX_FUTURE)


def _explicit_duration(text: str) -> int | None:
    """Minutes, if the text states a duration or an explicit time range."""
    span = _RANGE_RE.search(text)
    if span:
        start_h, start_m, end_h, end_m = span.groups()
        start = int(start_h) * 60 + int(start_m or 0)
        end = int(end_h) * 60 + int(end_m or 0)
        if end > start:
            return end - start

    match = _DURATION_RE.search(text)
    if match:
        word, digits, unit = match.groups()
        amount = _WORD_NUMBERS.get((word or "").lower(), 0) or int(digits or 0)
        if amount:
            return amount * 60 if unit.lower().startswith("or") else amount

    return None


def _has_explicit_time(phrase: str, resolved: datetime) -> bool:
    """Whether a clock time was actually stated, rather than defaulted by the parser.

    dateparser fills midnight when no time is given, so a bare date resolving to 00:00 is
    almost always "that day", not "at midnight".
    """
    if re.search(r"\d{1,2}[:.]\d{2}", phrase):
        return True
    if re.search(r"\b(?:alle|ore|at)\s+\d{1,2}\b", phrase, re.I):
        return True
    local = to_local(resolved)
    return not (local.hour == 0 and local.minute == 0)


def resolve(
    phrase: str | None,
    model_iso: str | None,
    reference: datetime,
    *,
    context: str = "",
    all_day_hint: bool | None = None,
) -> ResolvedWhen | None:
    """Turn an extracted phrase (and the model's guess) into a concrete window.

    Returns None when neither source yields a plausible date — which is the correct
    outcome for the many emails that mention no appointment at all.
    """
    from_phrase = _parse_phrase(phrase or "", reference)
    from_model = parse_iso(model_iso)

    if from_phrase is not None and not _plausible(from_phrase, reference):
        logger.debug("Discarding implausible phrase date %s from %r", from_phrase, phrase)
        from_phrase = None
    if from_model is not None and not _plausible(from_model, reference):
        logger.debug("Discarding implausible model date %s", from_model)
        from_model = None

    if from_phrase is None and from_model is None:
        return None

    agreed = False
    if from_phrase is not None and from_model is not None:
        # Within an hour counts as agreement: the model often rounds or drops minutes.
        agreed = abs((from_phrase - from_model).total_seconds()) <= 3600

    if from_phrase is not None:
        start = from_phrase
        source = "agreed" if agreed else "phrase"
    else:
        start = from_model
        source = "model"

    haystack = f"{phrase or ''} {context}"
    all_day = (
        all_day_hint
        if all_day_hint is not None
        else bool(_ALL_DAY_HINTS.search(haystack)) or not _has_explicit_time(phrase or "", start)
    )

    if all_day:
        local_start = to_local(start).replace(hour=0, minute=0, second=0, microsecond=0)
        return ResolvedWhen(
            start_ts=iso_utc(local_start),
            end_ts=iso_utc(local_start + timedelta(days=1)),
            all_day=True,
            source=source,
            agreed=agreed,
            phrase=phrase,
            note=None if agreed or from_model is None else "il modello proponeva un altro orario",
        )

    minutes = _explicit_duration(haystack) or DEFAULT_DURATION_MINUTES
    return ResolvedWhen(
        start_ts=iso_utc(start),
        end_ts=iso_utc(start + timedelta(minutes=minutes)),
        all_day=False,
        source=source,
        agreed=agreed,
        phrase=phrase,
        note=None if agreed or from_model is None else "il modello proponeva un altro orario",
    )
