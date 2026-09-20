"""One turn, end to end.

    message -> router -> context -> agent -> reply, persisted

Two LLM hops at most: the router (often zero, when a regex matched) and the agent. That
budget is the reason there is no supervisor and no agent-calling-agent — on local hardware a
third hop is the difference between a conversation and a wait.

Conversation memory lives in SQLite, not in a framework's checkpointer. This also fixes v1's
bug: it read history out of a LangGraph checkpointer and passed it *back* into the same
thread, so the graph appended it to state and history doubled every turn while the sliding
window never actually slid.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from donna.agents import roster, router
from donna.agents.base import AgentReply, run as run_agent
from donna.context import builder
from donna.store import repo
from donna.timeutil import iso_utc, now_utc

logger = logging.getLogger(__name__)

# Turns of history replayed to the model. Small on purpose: the world-state block carries the
# facts, so history only needs to carry the thread of the conversation.
HISTORY_TURNS = 8


@dataclass(slots=True)
class TurnResult:
    text: str
    intent: str
    agent: str
    route_via: str
    confidence: float
    trace_ids: list[str] = field(default_factory=list)
    tool_calls: list[tuple[str, str]] = field(default_factory=list)
    latency_ms: int = 0


def handle(
    message: str,
    *,
    channel: str = "cli",
    chat_id: str = "local",
    learn: bool = True,
) -> TurnResult:
    """Handle one user message."""
    import time

    started = time.perf_counter()

    decision = router.route(message)
    spec = roster.for_intent(decision.intent)

    # The query steers fact recall only; the rest of the world state is unconditional.
    context = builder.build(query=message).render()
    history = repo.recent_turns(channel, chat_id, limit=HISTORY_TURNS)

    reply: AgentReply = run_agent(spec, message, context=context, history=history)

    repo.append_turn(channel, chat_id, role="user", content=message)
    repo.append_turn(
        channel,
        chat_id,
        role="assistant",
        content=reply.text,
        agent=reply.agent,
        intent=decision.intent,
        trace_id=reply.trace_ids[-1] if reply.trace_ids else None,
    )

    result = TurnResult(
        text=reply.text,
        intent=decision.intent,
        agent=reply.agent,
        route_via=decision.via,
        confidence=decision.confidence,
        trace_ids=([decision.trace_id] if decision.trace_id else []) + reply.trace_ids,
        tool_calls=reply.tool_calls,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )

    if learn:
        _learn_quietly(message, channel=channel, chat_id=chat_id)

    logger.info(
        "Turno %s/%s: intent=%s (%s) agent=%s tools=%s %d ms",
        channel,
        chat_id,
        decision.intent,
        decision.via,
        reply.agent,
        [n for n, _ in reply.tool_calls] or "-",
        result.latency_ms,
    )
    return result


def _learn_quietly(message: str, *, channel: str, chat_id: str) -> None:
    """Harvest durable facts from what the user said.

    Runs after the reply is composed, and never raises: memory is a bonus, and a failure here
    must not cost the user their answer. (It is still synchronous — moving it off the turn is
    a later refinement, and at ~1 s on the resident model it is not yet worth the complexity.)
    """
    try:
        from donna.context import memory

        learned = memory.learn_from(message, source=f"{channel}:{chat_id}")
        if learned:
            logger.info("Imparato: %s", "; ".join(learned))
    except Exception:
        logger.warning("Fact extraction failed", exc_info=True)


def reset(channel: str, chat_id: str) -> int:
    """Forget the conversation, keeping the facts.

    Deliberately asymmetric: "start over" is about the thread, not about unlearning that he
    trains on Tuesdays.
    """
    return repo.clear_turns(channel, chat_id)


def briefing(*, channel: str = "cli", chat_id: str = "local") -> TurnResult:
    """The proactive summary, used by /briefing and by the scheduled morning message."""
    return handle("Fammi il punto della situazione.", channel=channel, chat_id=chat_id, learn=False)
