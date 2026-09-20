"""When the model will not call the tool, stop asking it to.

Tool calling is the weakest link in this stack. Asked to put a motorbike wash in the calendar,
the 9b produced a perfectly reasonable proposal *in prose* and called nothing — three times
running, including once immediately after being told "you did not call the tool, call it now",
to which it replied «Hai ragione, mi sono dimenticata di usare lo strumento» and then did not
use the strumento.

So the task gets changed rather than the prompt. Deciding to emit a tool call is unreliable;
filling a JSON schema is not, because the schema is compiled into a sampling grammar and the
model cannot produce anything else. This module asks the same question as a structured
extraction and builds the proposal in code.

It runs only after the tool path has already been given its chance, so the normal case is
unaffected — this is the floor, not the plan.
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, Field

from donna.llm import registry
from donna.llm.client import LLMError, get_llm
from donna.pipeline.schemas import Confidence
from donna.store import repo
from donna.timeutil import format_range_it, iso_utc, now_local, parse_iso

logger = logging.getLogger(__name__)

EXTRACT_SYSTEM = """Dalla richiesta dell'utente ricava l'impegno da mettere in calendario.

Ti do la data e l'ora attuali e gli impegni del giorno di cui parla. Usa QUELLI per risolvere
espressioni come "domani", "dopo il lavoro", "in serata". Non inventare orari: se dice "dopo
il lavoro", usa l'ora di fine del lavoro che vedi nei dati.

Compila i campi in quest'ordine:
1. "ragionamento": una riga su come hai ricavato l'orario.
2. "titolo": breve, da calendario ("Lavaggio moto", "Dentista").
3. "inizio_iso" e "fine_iso": formato YYYY-MM-DDTHH:MM, ora locale.
4. "promemoria_minuti": se chiede un avviso, quanti minuti prima. Altrimenti 0.
5. "dati_sufficienti": false SOLO se manca qualcosa che non puoi dedurre in nessun modo.
6. "domanda": se dati_sufficienti è false, la singola domanda da fargli.

Rispondi SOLO con il JSON."""


class ScheduleRequest(BaseModel):
    """What the user wants put in the calendar.

    Field order is generation order (the schema becomes a sampling grammar), so the reasoning
    comes first and the verdict last — the same lesson as the email extractor, where asking
    for the conclusion first made the model deny evidence it went on to quote.
    """

    ragionamento: str = Field(max_length=200)
    titolo: str = Field(max_length=120)
    inizio_iso: str | None = Field(default=None, description="YYYY-MM-DDTHH:MM locale")
    fine_iso: str | None = Field(default=None, description="YYYY-MM-DDTHH:MM locale")
    promemoria_minuti: int = 0
    dati_sufficienti: bool = True
    domanda: str | None = Field(default=None, max_length=200)
    confidenza: Confidence = 0.8


class Outcome(BaseModel):
    proposal_id: int | None = None
    text: str
    trace_id: str | None = None


def propose_from_request(
    message: str, *, context: str, parent_trace_id: str | None = None
) -> Outcome | None:
    """Extract the intended event and create the proposal. None if it cannot be done.

    `context` is the same world-state block the agent saw, so "after work" resolves against
    the real calendar rather than against the model's memory of a typical week.
    """
    try:
        result = get_llm().structured(
            registry.SCHEDULE,
            ScheduleRequest,
            f"{context}\n\n---\n\nRichiesta: {message}",
            system=EXTRACT_SYSTEM,
            parent_trace_id=parent_trace_id,
        )
    except LLMError:
        logger.warning("Fallback strutturato fallito", exc_info=True)
        return None

    request = result.value

    if not request.dati_sufficienti:
        # A real ambiguity: her question stands, and nothing is invented to paper over it.
        return Outcome(
            text=request.domanda or "Mi manca un dato per prepararla: puoi dirmi l'orario?",
            trace_id=result.trace_id,
        )

    start = parse_iso(request.inizio_iso)
    if start is None:
        logger.info("Fallback senza data usabile: %r", request.inizio_iso)
        return None

    # Sanity, in code: an event before now, or more than a year out, is a misparse rather than
    # an intention. Better to say nothing than to file a proposal for last Tuesday.
    if start < now_local() - timedelta(minutes=5) or start > now_local() + timedelta(days=400):
        logger.info("Fallback con data implausibile: %s", start)
        return None

    end = parse_iso(request.fine_iso) or (start + timedelta(hours=1))
    if end <= start:
        end = start + timedelta(hours=1)

    payload = {
        "kind": "appuntamento",
        "title": request.titolo,
        "start_ts": iso_utc(start),
        "end_ts": iso_utc(end),
        "all_day": False,
        "location": None,
    }
    if request.promemoria_minuti:
        payload["reminder_minutes"] = int(request.promemoria_minuti)

    proposal_id = repo.create_proposal(
        kind="calendar_event",
        source_type="conversation",
        source_id=None,
        payload=payload,
        reasoning=request.ragionamento or "Me l'hai chiesto tu in chat.",
        confidence=request.confidenza,
        trace_id=result.trace_id,
    )
    if proposal_id is None:
        return Outcome(text="Ne avevo già preparata una identica.", trace_id=result.trace_id)

    reminder = (
        f", con promemoria {request.promemoria_minuti} minuti prima"
        if request.promemoria_minuti
        else ""
    )
    logger.info("Proposta %d creata dal fallback strutturato", proposal_id)
    return Outcome(
        proposal_id=proposal_id,
        text=(
            f"Fatto: {request.titolo} — {format_range_it(start, end)}{reminder}. "
            f"È la proposta #{proposal_id}, confermala e va in calendario."
        ),
        trace_id=result.trace_id,
    )
