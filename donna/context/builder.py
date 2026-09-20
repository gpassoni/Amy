"""The world-state block.

This is the part that makes Donna feel like she knows your life, and it matters more than
the agent topology. v1 could only see whatever its last tool call happened to return, so it
had no way to reason about "your week" — every answer was assembled from scratch out of one
API response.

Now every agent call opens with a compact picture of the actual state: what is on today, what
is coming, what is overdue, what is sitting in the inbox, and what Donna has already
suggested. Agents then use tools to go *deeper* than this block, not to discover it.

Budgeted on purpose. The block is rebuilt for every turn and prompt-eval is the dominant
cost of a local model, so it is written to be dense rather than complete: counts instead of
lists where a count will do, and hard caps on every section.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

from donna.pipeline.resolve import describe as describe_proposal
from donna.store import repo
from donna.timeutil import (
    day_bounds_utc,
    format_it,
    humanize_duration,
    iso_utc,
    now_local,
    now_utc,
    parse_iso,
    to_local,
    weekday_name_it,
)

logger = logging.getLogger(__name__)

# Caps, so a busy week cannot crowd out the rest of the prompt.
MAX_TODAY = 12
MAX_WEEK = 14
MAX_TASKS = 8
MAX_IMPORTANT_MAIL = 5
# Generous on purpose: pending proposals are few (they get resolved), they are one line
# each, and they are the most actionable thing in the context. Truncating them at 5 made
# the agent state confidently that there were five when there were seven — it did not act
# on the "e altre 2" hint, so the hint is not a mechanism to rely on.
MAX_PROPOSALS = 12
MAX_FACTS = 8

# A gap shorter than this is not usable time, so it is not reported as free.
MIN_GAP_MINUTES = 45
# The working window used when talking about free time.
DAY_START_HOUR = 8
DAY_END_HOUR = 22


@dataclass(slots=True)
class WorldState:
    """Assembled context, with the pieces kept separate so interfaces can reuse them."""

    now_line: str
    today: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    week: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    mail: list[str] = field(default_factory=list)
    proposals: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)

    def render(self) -> str:
        parts = [f"## Adesso\n{self.now_line}"]

        parts.append(
            "## Oggi\n" + ("\n".join(self.today) if self.today else "Nessun impegno in calendario.")
        )
        if self.gaps:
            parts.append("## Tempo libero oggi\n" + "\n".join(self.gaps))
        parts.append(
            "## Prossimi 7 giorni\n"
            + ("\n".join(self.week) if self.week else "Niente in programma.")
        )
        parts.append(
            "## Cose da fare\n" + ("\n".join(self.tasks) if self.tasks else "Nessuna task aperta.")
        )
        if self.mail:
            parts.append("## Posta\n" + "\n".join(self.mail))
        if self.proposals:
            parts.append(
                "## Proposte che hai già fatto e sono in attesa\n" + "\n".join(self.proposals)
            )
        if self.facts:
            parts.append("## Cose che sai di lui\n" + "\n".join(self.facts))

        return "\n\n".join(parts)


def _event_line(row, *, with_day: bool = False) -> str:
    start, end = parse_iso(row["start_ts"]), parse_iso(row["end_ts"])
    if start is None:
        return f"- {row['summary']}"
    if row["all_day"]:
        stamp = format_it(start, with_time=False) if with_day else "tutto il giorno"
        return f"- {stamp}: {row['summary']}"
    local = to_local(start)
    if with_day:
        head = f"{weekday_name_it(start)} {local.day} alle {local:%H:%M}"
    else:
        head = f"{local:%H:%M}"
        if end is not None:
            head += f"–{to_local(end):%H:%M}"
    line = f"- {head}: {row['summary']}"
    if row["location"]:
        line += f" ({row['location'][:40]})"
    return line


def _free_gaps(events: list, day: date) -> list[str]:
    """Usable gaps between today's events, within working hours.

    Computed here rather than asked of the model: it is arithmetic over a sorted list, and a
    model doing it would get it subtly wrong while sounding confident.
    """
    zone_now = now_local()
    start_of_day = zone_now.replace(hour=DAY_START_HOUR, minute=0, second=0, microsecond=0)
    end_of_day = zone_now.replace(hour=DAY_END_HOUR, minute=0, second=0, microsecond=0)

    # Only look forward: a gap that has already passed is not free time.
    cursor = max(start_of_day, zone_now) if day == zone_now.date() else start_of_day

    busy: list[tuple] = []
    for row in events:
        if row["all_day"]:
            continue
        start, end = parse_iso(row["start_ts"]), parse_iso(row["end_ts"])
        if start is None:
            continue
        busy.append((to_local(start), to_local(end) if end else to_local(start)))
    busy.sort()

    gaps: list[str] = []
    for start, end in busy:
        if start > cursor:
            minutes = int((start - cursor).total_seconds() // 60)
            if minutes >= MIN_GAP_MINUTES:
                gaps.append(f"- {cursor:%H:%M}–{start:%H:%M} ({humanize_duration(minutes)})")
        cursor = max(cursor, end)

    if cursor < end_of_day:
        minutes = int((end_of_day - cursor).total_seconds() // 60)
        if minutes >= MIN_GAP_MINUTES:
            gaps.append(f"- {cursor:%H:%M}–{end_of_day:%H:%M} ({humanize_duration(minutes)})")

    return gaps


def build(*, query: str | None = None, include_facts: bool = True) -> WorldState:
    """Assemble the current world state.

    `query` steers semantic fact recall only; every other section is unconditional, because
    Donna should not need to be asked the right way to know what is on today.
    """
    today = date.today()
    now = now_local()
    state = WorldState(
        now_line=f"{format_it(now)} ({now:%d/%m/%Y})",
    )

    today_start, today_end = day_bounds_utc(today)
    _, week_end = day_bounds_utc(today, days=8)

    today_events = repo.events_between(today_start, today_end)
    state.today = [_event_line(r) for r in today_events[:MAX_TODAY]]
    state.gaps = _free_gaps(today_events, today)

    week_events = repo.events_between(today_end, week_end)
    state.week = [_event_line(r, with_day=True) for r in week_events[:MAX_WEEK]]
    if len(week_events) > MAX_WEEK:
        state.week.append(
            f"- (e altri {len(week_events) - MAX_WEEK}: usa elenca_eventi per la lista completa)"
        )

    overdue = {r["id"] for r in repo.overdue_tasks()}
    for row in repo.open_tasks(limit=MAX_TASKS):
        due = parse_iso(row["due_ts"])
        line = f"- {row['title']}"
        if due is not None:
            line += f" (entro {format_it(due, with_time=False)})"
        if row["id"] in overdue:
            line += " — IN RITARDO"
        state.tasks.append(line)

    state.mail = _mail_lines()

    # Truncation has to be visible. Without the "e altre N" line the agent read five and
    # confidently told the user there were five, when there were seven.
    pending = repo.pending_proposals(limit=MAX_PROPOSALS + 1)
    for row in pending[:MAX_PROPOSALS]:
        state.proposals.append(f"- #{row['id']} {describe_proposal(row)}")
    if len(pending) > MAX_PROPOSALS:
        total = repo.pending_proposal_count()
        state.proposals.append(
            f"- (e altre {total - MAX_PROPOSALS}: usa elenca_proposte per la lista completa)"
        )

    if include_facts:
        state.facts = _fact_lines(query)

    return state


def _mail_lines() -> list[str]:
    """Inbox state: counts, plus the important unread ones by name.

    A week's window rather than everything: what matters is what is live, and a year of
    archive would only add tokens.
    """
    since = iso_utc(now_utc() - timedelta(days=7))
    counts = repo.category_counts(since=since)
    if not counts:
        return []

    total = sum(counts.values())
    spread = ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in sorted(counts.items(), key=lambda kv: -kv[1]))
    lines = [f"Ultimi 7 giorni: {total} email ({spread})."]

    important = [
        r for r in repo.emails_by_category("importante", since=since, limit=MAX_IMPORTANT_MAIL * 2)
        if r["is_unread"]
    ][:MAX_IMPORTANT_MAIL]
    if important:
        lines.append("Importanti non lette:")
        for row in important:
            lines.append(f"- {row['from_name'] or row['from_addr']}: {(row['subject'] or '')[:70]}")
    return lines


def _fact_lines(query: str | None) -> list[str]:
    """Always-on facts, plus the ones most relevant to this turn.

    Recall failing is not worth failing a turn over — she just loses some memory for that
    message — so any error here is swallowed with a log line.
    """
    try:
        from donna.context import memory

        return [f"- {text}" for text in memory.recall(query, limit=MAX_FACTS)]
    except Exception:
        logger.warning("Fact recall failed; continuing without memory", exc_info=True)
        return []


def render(*, query: str | None = None) -> str:
    return build(query=query).render()
