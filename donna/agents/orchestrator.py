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

from donna.agents import fallback, prefetch, roster, router
from donna.agents.base import AgentReply
from donna.agents.base import run as run_agent
from donna.context import builder
from donna.store import activity, repo

logger = logging.getLogger(__name__)

# Turns of history replayed to the model. Small on purpose: the world-state block carries the
# facts, so history only needs to carry the thread of the conversation.
HISTORY_TURNS = 8

# How much of a past *assistant* turn to replay.
#
# Stale answers in history turned out to be dangerously authoritative for this model. Twice it
# reproduced an earlier reply verbatim instead of reading fresh state: once claiming five
# pending proposals when seven were in front of it, and once repeating a list of invented
# calendar slots — complete with "(dopo il colloquio HR)" — even after the correct slots had
# been computed and handed to it as fact.
#
# Moving state after the history helped and was not enough. The deeper issue is that a long,
# specific, fluent previous answer *looks* more like an answer than a data block does. So
# assistant turns are replayed truncated: enough to keep the thread ("we were discussing
# slots"), not enough to re-assert stale specifics as if they were current. User turns are
# never truncated — what he said is not something we get to summarise.
ASSISTANT_HISTORY_CHARS = 220

# Intents that are a request to change something rather than to look something up.
MUTATION_INTENTS = {"schedule_mutate", "task_mutate", "proposal_action"}

# Intents that only look something up. The user asked for no action, so a reply that says «ho
# preparato…» is referring to something that already exists, not lying about this turn.
QUERY_INTENTS = {"schedule_query", "inbox_query", "task_query", "briefing"}

# Intents whose changes are extracted as a structured plan rather than left to the model
# choosing to emit tool calls. See donna/agents/fallback.py.
STRUCTURED_INTENTS = {"schedule_mutate"}


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
    # Links a reply back to its activity row, so any interface can offer "what happened here".
    activity_id: str | None = None


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

    # The whole turn is one activity row, written before any work starts, so the dashboard can
    # show which agent is busy *while* it is busy rather than only after the fact.
    with activity.record(
        activity.TURN, "router", channel=channel, chat_id=chat_id, summary=message[:120]
    ) as run:
        decision = router.route(message)
        spec = roster.for_intent(decision.intent)

        # The actor becomes the agent as soon as routing has chosen one.
        run.actor = spec.name
        run.note(
            intent=decision.intent,
            route_via=decision.via,
            route_confidence=round(decision.confidence, 2),
            agent=spec.name,
        )
        activity.reassign(run)

        # The query steers fact recall only; the rest of the world state is unconditional.
        context = builder.build(query=message).render()

        # Some answers are arithmetic, and the model cannot be trusted to decide to compute
        # them — it invented calendar gaps instead. Those are computed here and appended as
        # fact. See donna/agents/prefetch.py.
        precomputed = prefetch.for_message(message)
        if precomputed:
            context = f"{context}\n\n{precomputed.text}"
            # What was handed over, spelled out, so the dashboard can show it verbatim.
            run.note(prefetched=precomputed.parts, prefetched_text=precomputed.text)

        history = _history_for_model(repo.recent_turns(channel, chat_id, limit=HISTORY_TURNS))

        expects_mutation = decision.intent in MUTATION_INTENTS
        has_plan = decision.intent in STRUCTURED_INTENTS

        reply: AgentReply | None = None
        used_plan = False

        # A calendar change goes through the structured plan *first*, and the agent loop is not
        # allowed to author proposals at all for it.
        #
        # It used to be the other way round: the agent loop ran, and the structured path was
        # only a fallback for when no tool had been called. That left a hole exactly where it
        # hurt — asked for two changes, the model called the tool once, "something was done",
        # the fallback stayed asleep, and the second change vanished with no trace. Deciding
        # to emit tool calls is the unreliable step (see donna/agents/fallback.py), so it is
        # not the step a request with several parts should depend on.
        if has_plan:
            outcome = fallback.propose_from_request(
                message, context=context, parent_trace_id=run.id
            )
            if outcome is not None:
                used_plan = True
                reply = AgentReply(
                    text=outcome.text,
                    agent=spec.name,
                    trace_ids=[outcome.trace_id] if outcome.trace_id else [],
                    tool_calls=[(tool, f"PROPOSTA #{pid}") for pid, tool in outcome.made],
                )

        # No plan came out of it (a question that only looked like a change, or the model
        # failed): the ordinary conversational path answers.
        if reply is None:
            reply = run_agent(
                spec,
                message,
                context=context,
                history=history,
                parent_trace_id=run.id,
                expect_mutation=expects_mutation and not has_plan,
                guard_claims=decision.intent not in QUERY_INTENTS,
            )

        run.note(
            tools=[name for name, _ in reply.tool_calls],
            iterations=reply.iterations,
            reply_chars=len(reply.text),
            expected_mutation=expects_mutation,
            mutated=reply.mutated,
            claimed_without_acting=reply.claimed_without_acting,
            structured_plan=used_plan,
        )
        run.describe(f"{decision.intent} → {spec.name}: {message[:80]}")

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
        activity_id=run.id,
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


def _history_for_model(turns: list[dict[str, str]]) -> list[dict[str, str]]:
    """Trim past assistant turns before replaying them. See ASSISTANT_HISTORY_CHARS.

    The full text stays in the database — this only shapes what the model is shown, so the
    dashboard and the Telegram transcript are unaffected.
    """
    trimmed: list[dict[str, str]] = []
    for turn in turns:
        content = turn.get("content") or ""
        if turn.get("role") == "assistant" and len(content) > ASSISTANT_HISTORY_CHARS:
            content = content[:ASSISTANT_HISTORY_CHARS].rstrip() + " […]"
        trimmed.append({"role": turn["role"], "content": content})
    return trimmed


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
