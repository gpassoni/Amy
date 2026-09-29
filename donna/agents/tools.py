"""Tools the agents can call.

Two rules shape this module.

**Tools read from the mirror, not from Google.** A question about the week is answered from
SQLite in microseconds. Only actions go out to the network. This is what makes conversation
feel instant on local hardware.

**A tool that mutates is either an explicit instruction or a proposal.** When the user says
"book the gym tomorrow at seven", that is an instruction and it executes. When Donna infers
something, it becomes a proposal. The approval gate exists for her guesses, not for his
orders — so `crea_evento` writes directly and is only ever reachable from a user turn.

Tool descriptions are prompt text. They are in Italian, phrased as instructions to the model,
and deliberately explicit about when *not* to use each one — a local model over-calls tools
far more often than it under-calls them.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from donna.agents import calendar_actions
from donna.google import tasks as gtasks
from donna.pipeline import resolve as resolve_pipeline
from donna.store import repo
from donna.timeutil import (
    day_bounds_utc,
    format_it,
    format_range_it,
    humanize_duration,
    iso_utc,
    now_local,
    now_utc,
    parse_iso,
    to_local,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[..., str]
    # A mutating tool changes the world; the loop reports these back to the user explicitly
    # rather than letting the model paraphrase what it did.
    mutating: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required or []}


# ================================================================== calendar (read)
def elenca_eventi(da_giorni: int = 0, a_giorni: int = 7) -> str:
    start_day = date.today() + timedelta(days=max(0, da_giorni))
    span = max(1, a_giorni - da_giorni + 1)
    start, end = day_bounds_utc(start_day, days=span)
    rows = repo.events_between(start, end)
    if not rows:
        return f"Nessun evento dal {start_day:%d/%m} per {span} giorni."

    lines = []
    for row in rows:
        begin, finish = parse_iso(row["start_ts"]), parse_iso(row["end_ts"])
        when = (
            format_it(begin, with_time=False) + " (tutto il giorno)"
            if row["all_day"]
            else format_range_it(begin, finish)
        )
        line = f"[{row['id']}] {when} — {row['summary']}"
        if row["location"]:
            line += f" @ {row['location']}"
        lines.append(line)
    return "\n".join(lines)


def trova_slot_liberi(durata_minuti: int = 60, entro_giorni: int = 7) -> str:
    """Free slots inside working hours, computed in code.

    Deliberately not a model job: it is interval arithmetic, and a model doing it produces
    confident nonsense like offering a slot inside an existing meeting.
    """
    from donna.context.builder import DAY_END_HOUR, DAY_START_HOUR

    found: list[str] = []
    today = date.today()

    for offset in range(max(1, entro_giorni)):
        day = today + timedelta(days=offset)
        start, end = day_bounds_utc(day)
        events = [r for r in repo.events_between(start, end) if not r["all_day"]]

        window_start = now_local().replace(hour=DAY_START_HOUR, minute=0, second=0, microsecond=0)
        window_start += timedelta(days=offset)
        window_end = window_start.replace(hour=DAY_END_HOUR, minute=0)
        if offset == 0:
            window_start = max(window_start, now_local())

        cursor = window_start
        busy = sorted(
            (to_local(parse_iso(r["start_ts"])), to_local(parse_iso(r["end_ts"])))
            for r in events
            if parse_iso(r["start_ts"]) and parse_iso(r["end_ts"])
        )
        for begin, finish in busy:
            if (begin - cursor).total_seconds() / 60 >= durata_minuti:
                found.append(f"{format_it(cursor)}–{begin:%H:%M}")
            cursor = max(cursor, finish)
        if (window_end - cursor).total_seconds() / 60 >= durata_minuti:
            found.append(f"{format_it(cursor)}–{window_end:%H:%M}")

        if len(found) >= 6:
            break

    if not found:
        return f"Nessuno slot da {humanize_duration(durata_minuti)} nei prossimi {entro_giorni} giorni."
    return "Slot liberi:\n" + "\n".join(f"- {s}" for s in found[:6])


# ================================================================== calendar (write)
def proponi_evento(
    titolo: str,
    inizio: str,
    fine: str | None = None,
    luogo: str | None = None,
    promemoria_minuti: int | None = None,
) -> str:
    """Propose an event rather than creating one.

    Originally this wrote straight to the calendar, on the reasoning that an explicit
    instruction is not a guess and does not need approving. That was wrong in practice, for
    two reasons the user found immediately:

      * Asked to schedule a motorbike wash after work, Donna replied "ti ho inserito" and
        had called no tool at all. Nothing existed, and the claim was indistinguishable from
        a real one. A proposal cannot be faked the same way: it is a row, and the interfaces
        render rows, not prose.
      * Even a correct write is worth seeing before it lands, because the model gets the
        *details* wrong — the time, the duration, what "after work" means — far more often
        than it gets the intent wrong.

    So everything goes through the same gate now, whether Donna inferred it or was told.
    """
    start = parse_iso(inizio)
    if start is None:
        return f"Non ho capito la data di inizio: {inizio!r}. Serve il formato YYYY-MM-DDTHH:MM."
    finish = parse_iso(fine) if fine else start + timedelta(hours=1)

    prepared = calendar_actions.propose_create(
        title=titolo, start=start, end=finish, location=luogo, reminder_minutes=promemoria_minuti
    )
    if prepared.proposal_id is None:
        return "Ne avevo già preparata una identica."
    return (
        f"PROPOSTA #{prepared.proposal_id}: {prepared.description}. "
        "Non è ancora in calendario: serve la conferma."
    )


def sposta_evento(
    id_evento: str, nuovo_inizio: str | None = None, nuova_fine: str | None = None
) -> str:
    start = parse_iso(nuovo_inizio) if nuovo_inizio else None
    finish = parse_iso(nuova_fine) if nuova_fine else None
    if nuovo_inizio and start is None:
        return f"Non ho capito la nuova data: {nuovo_inizio!r}."
    if nuova_fine and finish is None:
        return f"Non ho capito la nuova ora di fine: {nuova_fine!r}."
    if start is None and finish is None:
        return "Serve almeno un nuovo inizio o una nuova fine."
    try:
        prepared = calendar_actions.propose_move(event_id=id_evento, start=start, end=finish)
    except calendar_actions.ActionError as exc:
        return f"Non posso spostarlo: {exc}."
    if prepared.proposal_id is None:
        return "Ne avevo già preparata una identica."
    return f"PROPOSTA #{prepared.proposal_id}: {prepared.description}. Non è ancora cambiato niente: serve la conferma."


def elimina_evento(id_evento: str) -> str:
    try:
        prepared = calendar_actions.propose_delete(event_id=id_evento)
    except calendar_actions.ActionError as exc:
        return f"Non posso eliminarlo: {exc}."
    if prepared.proposal_id is None:
        return "Ne avevo già preparata una identica."
    return f"PROPOSTA #{prepared.proposal_id}: {prepared.description}. Non è ancora cambiato niente: serve la conferma."


# ================================================================== inbox
def cerca_email(query: str, quante: int = 5) -> str:
    rows = repo.search_emails(query, limit=quante)
    if not rows:
        return f"Nessuna email trovata per {query!r}."
    lines = []
    for row in rows:
        received = parse_iso(row["received_at"])
        lines.append(
            f"[{row['category'] or '?'}] {format_it(received, with_time=False) if received else '?'} — "
            f"{row['from_name'] or row['from_addr']}: {(row['subject'] or '')[:70]}"
        )
    return "\n".join(lines)


def email_per_categoria(categoria: str = "importante", giorni: int = 7, quante: int = 8) -> str:
    if categoria not in repo.CATEGORIES:
        return f"Categoria sconosciuta: {categoria!r}. Usa: {', '.join(repo.CATEGORIES)}."
    since = iso_utc(now_utc() - timedelta(days=max(1, giorni)))
    rows = repo.emails_by_category(categoria, since=since, limit=quante)
    if not rows:
        return f"Nessuna email {categoria} negli ultimi {giorni} giorni."
    lines = []
    for row in rows:
        received = parse_iso(row["received_at"])
        flag = " (non letta)" if row["is_unread"] else ""
        lines.append(
            f"{format_it(received, with_time=False) if received else '?'} — "
            f"{row['from_name'] or row['from_addr']}: {(row['subject'] or '')[:70]}{flag}"
        )
    return "\n".join(lines)


def leggi_email(id_email: str) -> str:
    row = repo.get_email(id_email)
    if row is None:
        return f"Email {id_email} non trovata."
    return (
        f"Da: {row['from_name']} <{row['from_addr']}>\n"
        f"Oggetto: {row['subject']}\n"
        f"Categoria: {row['category']} ({row['category_reason']})\n\n"
        f"{(row['body'] or '')[:1500]}"
    )


# ================================================================== tasks
def elenca_task(quante: int = 10) -> str:
    rows = repo.open_tasks(limit=quante)
    if not rows:
        return "Nessuna task aperta."
    overdue = {r["id"] for r in repo.overdue_tasks()}
    lines = []
    for row in rows:
        due = parse_iso(row["due_ts"])
        line = f"[{row['id']}] {row['title']}"
        if due is not None:
            line += f" — entro {format_it(due, with_time=False)}"
        if row["id"] in overdue:
            line += " (IN RITARDO)"
        lines.append(line)
    return "\n".join(lines)


def aggiungi_task(titolo: str, scadenza: str | None = None, note: str | None = None) -> str:
    due = parse_iso(scadenza) if scadenza else None
    created = gtasks.create_task(titolo, notes=note, due_iso=iso_utc(due) if due else None)
    when = f" (entro {format_it(due, with_time=False)})" if due else ""
    return f"Aggiunta: {titolo}{when}." if created else "Non sono riuscita ad aggiungerla."


def completa_task(riferimento: str) -> str:
    row = repo.find_task(riferimento)
    if row is None:
        return f"Nessuna task aperta corrisponde a {riferimento!r}."
    gtasks.complete_task(row["id"])
    return f"Fatto: {row['title']} segnata come completata."


# ================================================================== proposals
def elenca_proposte() -> str:
    rows = repo.pending_proposals(limit=10)
    if not rows:
        return "Nessuna proposta in attesa."
    lines = []
    for row in rows:
        lines.append(f"#{row['id']} {resolve_pipeline.describe(row)} — {row['reasoning'] or ''}")
    return "\n".join(lines)


def accetta_proposta(id_proposta: int) -> str:
    try:
        result = resolve_pipeline.accept(int(id_proposta), via="chat")
    except resolve_pipeline.ProposalError as exc:
        return f"Non ho potuto accettarla: {exc}"
    return f"Fatto: {result.message}"


def rifiuta_proposta(id_proposta: int, motivo: str | None = None) -> str:
    try:
        result = resolve_pipeline.reject(int(id_proposta), via="chat", note=motivo)
    except resolve_pipeline.ProposalError as exc:
        return f"Non ho potuto rifiutarla: {exc}"
    return f"Scartata: {result.message}"


# ================================================================== memory
def ricorda(fatto: str) -> str:
    from donna.context import memory

    stored = memory.remember(fatto, source="chat", confidence=0.95)
    return "Me lo ricorderò." if stored else "Lo sapevo già."


# ================================================================== registry
TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "elenca_eventi",
            "Elenca gli impegni in calendario. da_giorni=0 è oggi, 1 domani. Usalo solo se "
            "ti serve più dettaglio di quanto c'è già nel contesto.",
            _obj(
                {
                    "da_giorni": {"type": "integer", "description": "0 = oggi, 1 = domani"},
                    "a_giorni": {"type": "integer", "description": "ultimo giorno da includere"},
                }
            ),
            elenca_eventi,
        ),
        Tool(
            "trova_slot_liberi",
            "Trova spazi liberi in calendario di almeno una certa durata. Usalo quando "
            "chiede quando può fare qualcosa, o dove infilare un impegno.",
            _obj(
                {
                    "durata_minuti": {"type": "integer"},
                    "entro_giorni": {"type": "integer"},
                }
            ),
            trova_slot_liberi,
        ),
        Tool(
            "proponi_evento",
            "Prepara un impegno da mettere in calendario. NON lo mette: crea una proposta che "
            "lui conferma con un tocco. Devi chiamare questo strumento: se scrivi soltanto che "
            "l'hai fatto, non è stato fatto niente. Date in formato YYYY-MM-DDTHH:MM.",
            _obj(
                {
                    "titolo": {"type": "string"},
                    "inizio": {"type": "string", "description": "YYYY-MM-DDTHH:MM"},
                    "fine": {"type": "string", "description": "YYYY-MM-DDTHH:MM, opzionale"},
                    "luogo": {"type": "string"},
                    "promemoria_minuti": {
                        "type": "integer",
                        "description": "minuti di anticipo per l'avviso, se ne chiede uno",
                    },
                },
                ["titolo", "inizio"],
            ),
            proponi_evento,
            mutating=True,
        ),
        Tool(
            "sposta_evento",
            "Prepara lo spostamento di un evento esistente, o il cambio della sua ora di fine: "
            "crea una proposta che lui conferma. Serve l'id, che trovi con elenca_eventi. Se "
            "cambia solo la fine, passa solo nuova_fine.",
            _obj(
                {
                    "id_evento": {"type": "string"},
                    "nuovo_inizio": {"type": "string", "description": "YYYY-MM-DDTHH:MM"},
                    "nuova_fine": {"type": "string", "description": "YYYY-MM-DDTHH:MM"},
                },
                ["id_evento"],
            ),
            sposta_evento,
            mutating=True,
        ),
        Tool(
            "elimina_evento",
            "Prepara la cancellazione di un evento: crea una proposta che lui conferma. Serve l'id.",
            _obj({"id_evento": {"type": "string"}}, ["id_evento"]),
            elimina_evento,
            mutating=True,
        ),
        Tool(
            "cerca_email",
            "Cerca fra le email per parola chiave, mittente o argomento.",
            _obj({"query": {"type": "string"}, "quante": {"type": "integer"}}, ["query"]),
            cerca_email,
        ),
        Tool(
            "email_per_categoria",
            "Elenca le email di una categoria: importante, da_leggere, inutile.",
            _obj(
                {
                    "categoria": {"type": "string", "enum": list(repo.CATEGORIES)},
                    "giorni": {"type": "integer"},
                    "quante": {"type": "integer"},
                }
            ),
            email_per_categoria,
        ),
        Tool(
            "leggi_email",
            "Leggi il testo completo di una email di cui conosci l'id.",
            _obj({"id_email": {"type": "string"}}, ["id_email"]),
            leggi_email,
        ),
        Tool(
            "elenca_task",
            "Elenca le cose da fare aperte.",
            _obj({"quante": {"type": "integer"}}),
            elenca_task,
        ),
        Tool(
            "aggiungi_task",
            "Aggiunge una cosa da fare. La scadenza è opzionale, formato YYYY-MM-DD.",
            _obj(
                {
                    "titolo": {"type": "string"},
                    "scadenza": {"type": "string", "description": "YYYY-MM-DD, opzionale"},
                    "note": {"type": "string"},
                },
                ["titolo"],
            ),
            aggiungi_task,
            mutating=True,
        ),
        Tool(
            "completa_task",
            "Segna una task come fatta. Accetta l'id o una parola del titolo.",
            _obj({"riferimento": {"type": "string"}}, ["riferimento"]),
            completa_task,
            mutating=True,
        ),
        Tool(
            "elenca_proposte",
            "Elenca le proposte in attesa di risposta, con il motivo di ognuna.",
            _obj({}),
            elenca_proposte,
        ),
        Tool(
            "accetta_proposta",
            "Accetta una proposta e mettila in calendario. Serve il numero della proposta.",
            _obj({"id_proposta": {"type": "integer"}}, ["id_proposta"]),
            accetta_proposta,
            mutating=True,
        ),
        Tool(
            "rifiuta_proposta",
            "Scarta una proposta. Se dice perché, passalo in motivo: serve per imparare.",
            _obj(
                {"id_proposta": {"type": "integer"}, "motivo": {"type": "string"}}, ["id_proposta"]
            ),
            rifiuta_proposta,
            mutating=True,
        ),
        Tool(
            "ricorda",
            "Memorizza un fatto durevole su di lui (abitudini, preferenze, vincoli). "
            "Non usarlo per impegni singoli: quelli vanno in calendario.",
            _obj({"fatto": {"type": "string"}}, ["fatto"]),
            ricorda,
            mutating=True,
        ),
    ]
}


def schemas_for(names: list[str]) -> list[dict[str, Any]]:
    return [TOOLS[name].schema() for name in names if name in TOOLS]


def execute(name: str, arguments: dict[str, Any] | str) -> str:
    """Run a tool call from the model, tolerating the ways it gets arguments wrong.

    A local model will occasionally send arguments as a JSON string, pass an unknown keyword,
    or omit a required one. None of those should raise out of the agent loop — the model gets
    told what went wrong and can correct itself on the next turn, which it usually does.
    """
    tool = TOOLS.get(name)
    if tool is None:
        return f"Strumento sconosciuto: {name}. Disponibili: {', '.join(sorted(TOOLS))}."

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except ValueError:
            return f"Argomenti non validi per {name}: {arguments!r}"
    if not isinstance(arguments, dict):
        return f"Argomenti non validi per {name}: attesi un oggetto."

    # Drop unknown keys rather than failing on them.
    allowed = set(tool.parameters.get("properties", {}))
    cleaned = {k: v for k, v in arguments.items() if k in allowed and v is not None}
    missing = [k for k in tool.parameters.get("required", []) if k not in cleaned]
    if missing:
        return f"Mancano argomenti obbligatori per {name}: {', '.join(missing)}."

    try:
        logger.info("Tool %s(%s)", name, cleaned)
        return tool.run(**cleaned)
    except Exception as exc:  # surfaced to the model, not to the user
        logger.warning("Tool %s failed: %s", name, exc, exc_info=True)
        return f"Lo strumento {name} è fallito: {exc}"
