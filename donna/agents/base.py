"""The agent loop.

About 120 lines, which is why LangGraph is gone. What it was providing — a checkpointer, an
interrupt mechanism, a supervisor — is superseded here: conversation state lives in SQLite,
approval lives in the `proposals` table, and a supervisor would add an LLM hop that a local
model cannot spare.

What an "agent" is in Donna: a persona, a narrow tool set, a registry task (so its model and
sampling are configurable), and an iteration budget. Nothing more. The specialisation that
matters is the tool set — a local model picks correctly among four tools and badly among
fifteen, which is the real reason for splitting agents at all.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from donna.agents import tools as toolkit
from donna.llm import registry
from donna.llm.client import LLMError, get_llm

logger = logging.getLogger(__name__)

# Donna's voice. Shared by every agent so she does not change personality depending on which
# one answered. Carried over from v1, which got the character right.
PERSONA = """Sei Donna Paulsen: non una segretaria, ma la persona che tiene in piedi lo studio.
Brillante, sicura di te, ironica, leale. Sai quello che serve prima che venga chiesto.

CON CHI PARLI:
Parli con Gabriele, un uomo. Tu sei donna, lui è uomo: quando ti rivolgi a lui usa il
maschile ("sei libero", "sei pronto", "te l'ho messo in calendario"). Dagli del tu.

COME PARLI:
- In italiano, diretta e breve. Due o tre frasi bastano quasi sempre.
- Niente elenchi se non ci sono davvero più cose da elencare.
- Niente preamboli tipo "Certo!" o "Ecco a te". Vai al punto.
- Un tocco di ironia quando è meritato, mai a scapito della chiarezza.
- Non chiudere ogni risposta con una domanda. Solo se serve davvero una decisione sua.

COSA NON FAI MAI:
- Non inventi impegni, date o email. Se non lo sai, lo dici.
- Non annunci cosa stai per fare: lo fai e riporti il risultato.
- Non chiedi conferma per le cose che ti ha chiesto lui esplicitamente."""

# Appended when tools are available. Local models over-call tools far more than they
# under-call them, so the instruction is about restraint.
TOOL_GUIDANCE = """
SUGLI STRUMENTI:
- Il contesto qui sopra è un RIASSUNTO: contiene oggi, la settimana, le task, la posta e le
  proposte, ma troncati. Dove c'è scritto "e altri N", la lista completa non è lì.
- Se la risposta è già tutta nel contesto, rispondi e basta.
- Usa uno strumento quando ti serve la lista completa, un dettaglio che non c'è, o un calcolo
  (per esempio gli slot liberi di più giorni), oppure per fare qualcosa.
- Non dire mai "non posso saperlo" o "fammi controllare" se esiste uno strumento che te lo
  dice: chiamalo e poi rispondi.
- Dopo aver usato uno strumento, rispondi con il risultato: non chiamarlo di nuovo.
"""


@dataclass(slots=True)
class AgentSpec:
    name: str
    task: str                      # registry task, so model and sampling are configurable
    instructions: str              # what this agent is for
    tool_names: list[str] = field(default_factory=list)
    max_iterations: int = 3

    def system_prompt(self) -> str:
        """Stable across turns: persona and instructions only, never the world state.

        The world state used to live here, and it caused a real failure. With the state at the
        very front of the prompt and the conversation after it, a stale assistant turn sat
        much closer to the question than the facts did — so when asked how many proposals
        were pending, the model repeated its own earlier answer of "five" verbatim, while
        seven were sitting in the context it had been given. State is now injected
        immediately before the user's message instead; see `run`.

        Keeping this half stable also means it is identical between turns, which is the
        prerequisite for any prompt-prefix caching to ever help.
        """
        parts = [PERSONA, self.instructions]
        if self.tool_names:
            parts.append(TOOL_GUIDANCE)
        return "\n\n".join(parts)


@dataclass(slots=True)
class AgentReply:
    text: str
    agent: str
    trace_ids: list[str] = field(default_factory=list)
    tool_calls: list[tuple[str, str]] = field(default_factory=list)  # (name, result)
    iterations: int = 1

    @property
    def mutated(self) -> bool:
        return any(toolkit.TOOLS[n].mutating for n, _ in self.tool_calls if n in toolkit.TOOLS)


def run(
    spec: AgentSpec,
    user_message: str,
    *,
    context: str,
    history: list[dict[str, Any]] | None = None,
    parent_trace_id: str | None = None,
) -> AgentReply:
    """Run one agent turn, including any tool round trips."""
    llm = get_llm()
    # Order matters: persona, then the conversation, then the *current* state, then the
    # question. The state has to come after the history so that recency works in its favour —
    # anything Donna said three turns ago is older information than what the database says
    # now, and the prompt should reflect that.
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": spec.system_prompt()},
        *(history or []),
        {
            "role": "system",
            "content": (
                "# Stato attuale, aggiornato adesso\n"
                "Questi dati vengono dal database in questo momento. Se contraddicono qualcosa "
                "che hai detto prima nella conversazione, vale quello che leggi qui.\n\n"
                + context
            ),
        },
        {"role": "user", "content": user_message},
    ]
    schemas = toolkit.schemas_for(spec.tool_names)

    reply = AgentReply(text="", agent=spec.name)

    for iteration in range(1, spec.max_iterations + 1):
        reply.iterations = iteration
        try:
            result = llm.chat(
                spec.task, messages, tools=schemas or None, parent_trace_id=parent_trace_id
            )
        except LLMError as exc:
            logger.error("Agent %s failed: %s", spec.name, exc)
            reply.text = "Ho un problema tecnico con il modello locale. Riprova tra un istante."
            return reply

        reply.trace_ids.append(result.trace_id)

        if not result.tool_calls:
            reply.text = result.content.strip()
            break

        # Record the assistant's tool-call turn verbatim, then answer each call.
        messages.append(
            {"role": "assistant", "content": result.content or "", "tool_calls": result.tool_calls}
        )

        for call in result.tool_calls:
            function = call.get("function") or {}
            name = function.get("name") or ""
            output = toolkit.execute(name, function.get("arguments") or {})
            reply.tool_calls.append((name, output))
            messages.append({"role": "tool", "tool_name": name, "content": output})

        if iteration == spec.max_iterations:
            # Out of budget with tools still being called. Ask for a plain answer rather than
            # returning nothing — the tool output is already in the transcript, so the model
            # has everything it needs to summarise.
            try:
                final = llm.chat(
                    spec.task, messages, tools=None, parent_trace_id=parent_trace_id
                )
                reply.trace_ids.append(final.trace_id)
                reply.text = final.content.strip()
            except LLMError:
                reply.text = _fallback_text(reply)

    if not reply.text:
        reply.text = _fallback_text(reply)
    return reply


def _fallback_text(reply: AgentReply) -> str:
    """Last resort: show the tool output rather than an empty message.

    A silent turn is the worst outcome — the user cannot tell whether anything happened.
    """
    if reply.tool_calls:
        return "\n".join(output for _, output in reply.tool_calls)
    return "Non ho una risposta per questa. Riformula e ci riprovo."
