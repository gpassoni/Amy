"""When the model will not call the tool, stop asking it to.

Tool calling is the weakest link in this stack. Asked to put a motorbike wash in the calendar,
the 9b produced a perfectly reasonable proposal *in prose* and called nothing — three times
running, including once immediately after being told "you did not call the tool, call it now",
to which it replied «Hai ragione, mi sono dimenticata di usare lo strumento» and then did not
use the strumento.

So the task gets changed rather than the prompt. Deciding to emit a tool call is unreliable;
filling a JSON schema is not, because the schema is compiled into a sampling grammar and the
model cannot produce anything else. This module asks the same question as a structured
extraction and builds the proposals in code.

**One request can hold several changes.** The first version of this schema described a single
event, so «una corsa domenica pomeriggio, e anche uno slot di burocrazia la mattina» produced
one proposal and a confident «Fatto:» — the model had understood both (its own reasoning field
said so) and the schema left no room for the second. It is a list now, and three things stop a
part of the request from vanishing quietly:

  * every action carries the fragment of the message it comes from, checked in code against the
    message — an action with no basis in what he wrote is dropped, not filed;
  * the model first counts how many distinct things were asked, and a shortfall is reported;
  * the reply is written here, from what was actually created, so it cannot claim more than
    exists and it names anything that was left out.

It also handles moving and deleting existing events, which the single-event schema could only
misread as "create a new event called 'Aggiornamento turno'".
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, Field

from donna.agents import calendar_actions
from donna.agents import tools as toolkit
from donna.llm import registry
from donna.llm.client import LLMError, get_llm
from donna.timeutil import now_local, parse_iso

logger = logging.getLogger(__name__)

MAX_ACTIONS = 6

# How many days of events, with their ids, the model is shown for moves and deletes.
EVENT_LOOKAHEAD_DAYS = 21

# Share of an action's quoted fragment that must be found in the message. Not exact matching:
# a small model copies with slips (punctuation, an article), and rejecting those would throw
# away real requests. A fragment that is mostly absent, though, is an invention.
MIN_QUOTE_OVERLAP = 0.6

EXTRACT_SYSTEM = """Dalla richiesta dell'utente ricava TUTTE le modifiche al calendario che chiede.
Una richiesta può contenerne più di una: crea un'azione per ciascuna, senza fonderle e senza
tralasciarne.

Ti do la data e l'ora attuali, gli impegni del giorno di cui parla e l'elenco degli eventi
con il loro id. Usa QUELLI per risolvere espressioni come "domani", "dopo il lavoro", "in
serata". Non inventare orari: se dice "dopo il lavoro", usa l'ora di fine del lavoro che vedi.

Prima di tutto:
1. "numero_richieste": quante modifiche distinte chiede il messaggio (conta ogni impegno da
   creare, spostare o cancellare).

Poi, per ogni modifica, un'azione con i campi in quest'ordine:
2. "frase": la parte del messaggio da cui nasce, copiata alla lettera.
3. "tipo": "crea" per un impegno nuovo; "sposta" per cambiare l'orario di un impegno che
   ESISTE GIÀ (anche solo l'ora di fine: "il turno finisce alle 15"); "elimina" per cancellarne uno.
4. "cosa_cambia": solo per "sposta". "solo_fine" se cambia solo l'ora di fine ("il turno
   finisce alle 15"); "solo_inizio" se cambia solo l'ora di inizio ("il turno inizia alle 9");
   "intero" se l'impegno si sposta tutto, con la stessa durata ("sposta il dentista alle 18").
   Per "crea" ed "elimina": "niente".
5. "ragionamento": UNA frase brevissima (massimo 100 caratteri) su come hai ricavato l'orario.
6. "titolo": breve, da calendario ("Lavaggio moto", "Dentista"). Usa le SUE parole: se dice
   "il dentista", il titolo è "Dentista", non un nome più specifico che hai visto altrove.
7. "id_evento": solo per sposta ed elimina: l'id dell'evento dall'elenco. Non inventarlo.
8. "inizio_iso" e "fine_iso": formato YYYY-MM-DDTHH:MM, ora locale. Per "sposta" indica solo
   ciò che cambia; ciò che non nomina lascialo vuoto.
9. "promemoria_minuti": se chiede un avviso, quanti minuti prima. Altrimenti 0.
10. "dati_sufficienti": false SOLO se manca qualcosa che non puoi dedurre in nessun modo.
11. "domanda": se dati_sufficienti è false, la singola domanda da fargli.

Compila TUTTI i campi di ogni azione; usa null dove non si applica.
Se il messaggio non chiede nessuna modifica al calendario, "azioni" è vuoto.
Rispondi SOLO con il JSON."""


class ScheduleAction(BaseModel):
    """One change to the calendar.

    Field order is generation order (the schema becomes a sampling grammar): the quoted
    evidence and the kind of change come first, so everything after is conditioned on them —
    the same lesson as the email extractor, where asking for the verdict first made the model
    deny evidence it went on to quote.

    **Every field is required, on purpose.** A field with a default is optional in the schema,
    and an optional field is one the grammar lets the model skip. The first version of this
    class defaulted everything except `frase`, and on a real message the model wrote a long
    `ragionamento`, hit its length cap, and closed the object — the title and both times were
    silently filled in as empty and two perfectly understood requests came back as «non ho una
    data utilizzabile». Required-but-nullable forces it to say something for each field.
    """

    frase: str = Field(max_length=300)
    tipo: Literal["crea", "sposta", "elimina"]
    # Which part of an existing event changes. Classifying is something the model does well;
    # deriving the untouched half of the time range from that is arithmetic, and is left to
    # code. It could not be done by prompt: «il turno inizia alle 9» came back, twice, with the
    # end shifted an hour along with the start, even with a worked example in front of it.
    cosa_cambia: Literal["intero", "solo_inizio", "solo_fine", "niente"]
    ragionamento: str = Field(max_length=140)
    titolo: str = Field(max_length=120)
    id_evento: str | None
    inizio_iso: str | None = Field(description="YYYY-MM-DDTHH:MM locale")
    fine_iso: str | None = Field(description="YYYY-MM-DDTHH:MM locale")
    promemoria_minuti: int
    dati_sufficienti: bool
    domanda: str | None = Field(max_length=200)


# The plan does not ask the model to rate itself: it was never a good signal, and a field the
# model must fill is tokens it must generate. Proposals from a request he typed get this.
PLAN_CONFIDENCE = 0.8


class SchedulePlan(BaseModel):
    numero_richieste: int = Field(ge=0, le=20)
    azioni: list[ScheduleAction] = Field(max_length=MAX_ACTIONS)


class Outcome(BaseModel):
    text: str
    made: list[tuple[int, str]] = Field(default_factory=list)   # (proposal id, tool name)
    trace_id: str | None = None

    @property
    def proposal_ids(self) -> list[int]:
        return [pid for pid, _ in self.made]

    @property
    def proposal_id(self) -> int | None:
        return self.made[0][0] if self.made else None


@dataclass(slots=True)
class _Tally:
    made: list[tuple[int, str, str]] = field(default_factory=list)   # (id, description, tool)
    questions: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)              # «frase» — why


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", (text or "").casefold()))


def is_grounded(quote: str, message: str) -> bool:
    """Whether the quoted fragment really comes from the message."""
    words = _tokens(quote)
    if not words:
        return False
    return len(words & _tokens(message)) / len(words) >= MIN_QUOTE_OVERLAP


def _events_block() -> str:
    """Upcoming events with their ids, the only place a move or delete can name its target."""
    try:
        listing = toolkit.elenca_eventi(da_giorni=0, a_giorni=EVENT_LOOKAHEAD_DAYS)
    except Exception:
        logger.warning("Elenco eventi per il piano non disponibile", exc_info=True)
        return ""
    return f"## Eventi in calendario, con id (per spostare o eliminare)\n{listing}"


def _short(quote: str) -> str:
    quote = " ".join((quote or "").split())
    return quote if len(quote) <= 70 else quote[:67] + "…"


def _apply(action: ScheduleAction, message: str, tally: _Tally, trace_id: str | None) -> None:
    label = _short(action.frase)

    if not is_grounded(action.frase, message):
        # The model produced an action that does not correspond to anything he wrote.
        logger.warning("Azione scartata, frase non presente nel messaggio: %r", action.frase)
        tally.skipped.append(f"«{label}» — non l'ho ritrovata nel tuo messaggio")
        return

    if not action.dati_sufficienti:
        tally.questions.append(
            action.domanda or f"Per «{label}» mi manca un dato: puoi dirmi l'orario?"
        )
        return

    start = parse_iso(action.inizio_iso) if action.inizio_iso else None
    end = parse_iso(action.fine_iso) if action.fine_iso else None
    reasoning = action.ragionamento or "Me l'hai chiesto tu in chat."

    try:
        if action.tipo == "crea":
            if start is None:
                tally.skipped.append(f"«{label}» — non ho una data utilizzabile")
                return
            # Sanity, in code: an event before now, or over a year out, is a misparse rather
            # than an intention. Better to say so than to file a proposal for last Tuesday.
            if start < now_local() - timedelta(minutes=5) or start > now_local() + timedelta(days=400):
                logger.info("Azione con data implausibile: %s", start)
                tally.skipped.append(f"«{label}» — la data che ne ricavo ({start:%d/%m/%Y}) non ha senso")
                return
            if not action.titolo.strip():
                tally.skipped.append(f"«{label}» — non ho capito come chiamarlo")
                return
            if end is None or end <= start:
                end = start + timedelta(hours=1)
            prepared = calendar_actions.propose_create(
                title=action.titolo.strip(),
                start=start,
                end=end,
                reminder_minutes=action.promemoria_minuti or None,
                reasoning=reasoning,
                confidence=PLAN_CONFIDENCE,
                trace_id=trace_id,
            )
        elif action.tipo == "sposta":
            if start is None and end is None:
                tally.skipped.append(f"«{label}» — non ho capito a che ora spostarlo")
                return
            if action.cosa_cambia == "solo_inizio" and start is None:
                tally.skipped.append(f"«{label}» — non ho capito il nuovo orario di inizio")
                return
            if action.cosa_cambia == "solo_fine" and end is None:
                tally.skipped.append(f"«{label}» — non ho capito il nuovo orario di fine")
                return
            prepared = calendar_actions.propose_move(
                event_id=action.id_evento or "",
                # "solo_fine" ignores a start the model volunteered anyway, and "solo_inizio"
                # ignores an end: the untouched half comes from the event, not from the model.
                start=None if action.cosa_cambia == "solo_fine" else start,
                end=None if action.cosa_cambia == "solo_inizio" else end,
                keep_end=action.cosa_cambia == "solo_inizio",
                reasoning=reasoning,
                confidence=PLAN_CONFIDENCE,
                trace_id=trace_id,
            )
        else:
            prepared = calendar_actions.propose_delete(
                event_id=action.id_evento or "",
                reasoning=reasoning,
                confidence=PLAN_CONFIDENCE,
                trace_id=trace_id,
            )
    except calendar_actions.ActionError as exc:
        tally.skipped.append(f"«{label}» — {exc}")
        return

    if prepared.proposal_id is None:
        tally.skipped.append(f"«{label}» — ne avevo già preparata una identica")
        return
    tally.made.append((prepared.proposal_id, prepared.description, prepared.tool))


def _compose(tally: _Tally, *, asked: int) -> str:
    """The reply, built from what exists. It cannot claim more than was created."""
    lines: list[str] = []

    if tally.made:
        noun = "una proposta" if len(tally.made) == 1 else f"{len(tally.made)} proposte"
        lines.append(f"Ho preparato {noun}:")
        lines.extend(f"- #{pid} {desc}" for pid, desc, _ in tally.made)
        lines.append(
            "Non è ancora in calendario: confermala."
            if len(tally.made) == 1
            else "Nessuna è ancora in calendario: confermale una per una."
        )

    if tally.questions:
        if lines:
            lines.append("")
        lines.append("Mi serve un chiarimento:" if len(tally.questions) == 1 else "Mi servono dei chiarimenti:")
        lines.extend(f"- {q}" for q in tally.questions)

    if tally.skipped:
        if lines:
            lines.append("")
        lines.append("Non ho preparato:")
        lines.extend(f"- {s}" for s in tally.skipped)

    handled = len(tally.made) + len(tally.questions) + len(tally.skipped)
    if asked > handled:
        lines.append("")
        lines.append(
            f"Attenzione: nel messaggio ho contato {asked} richieste ma ne ho trattate {handled}. "
            "Ricontrolla che non ne manchi qualcuna."
        )

    return "\n".join(lines)


def propose_from_request(
    message: str, *, context: str, parent_trace_id: str | None = None
) -> Outcome | None:
    """Extract every requested calendar change and create a proposal for each.

    None means nothing actionable came out of it (or the model failed), and the caller should
    fall back to the ordinary conversational path.

    `context` is the same world-state block the agent saw, so "after work" resolves against
    the real calendar rather than against the model's memory of a typical week.
    """
    prompt = "\n\n".join(part for part in (context, _events_block(), f"---\n\nRichiesta: {message}") if part)
    try:
        result = get_llm().structured(
            registry.SCHEDULE,
            SchedulePlan,
            prompt,
            system=EXTRACT_SYSTEM,
            parent_trace_id=parent_trace_id,
        )
    except LLMError:
        logger.warning("Piano strutturato fallito", exc_info=True)
        return None

    plan = result.value
    if not plan.azioni:
        logger.info("Piano senza azioni per %r", message[:60])
        return None

    tally = _Tally()
    for action in plan.azioni:
        _apply(action, message, tally, result.trace_id)

    logger.info(
        "Piano: %d richieste contate, %d azioni -> %d proposte, %d domande, %d scartate",
        plan.numero_richieste, len(plan.azioni), len(tally.made), len(tally.questions), len(tally.skipped),
    )
    return Outcome(
        text=_compose(tally, asked=plan.numero_richieste),
        made=[(pid, tool) for pid, _, tool in tally.made],
        trace_id=result.trace_id,
    )
