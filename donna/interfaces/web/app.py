"""Local web dashboard.

Purpose, in order of importance:

  1. **See what Donna is doing.** Which agent is working right now, on what, with which tools,
     and — by drilling into the traces — the exact prompt that produced an answer. The
     `activity` table exists for this.
  2. **Review proposals properly.** Accept, reject, or *edit and accept*. The edit path matters
     most: it is the only interaction that records what the right answer was, rather than just
     that she was wrong.
  3. **Show the training set accumulating**, so the feedback loop is visible rather than
     theoretical.

Server-rendered Jinja plus htmx: no build step, no npm, no client state to keep in sync. Bound
to 127.0.0.1 by default because there is no authentication — this reads a real mailbox.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from donna.agents import orchestrator, roster
from donna.config import get_settings
from donna.eval import dataset
from donna.llm import registry
from donna.pipeline import resolve, triage
from donna.store import activity, repo
from donna.store.db import get_db
from donna.timeutil import (
    day_bounds_utc,
    format_it,
    format_range_it,
    iso_utc,
    now_local,
    now_utc,
    parse_iso,
    to_local,
)

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
CHANNEL = "web"
CHAT_ID = "dashboard"

app = FastAPI(title="Donna", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=str(HERE / "templates"))


# ---------------------------------------------------------------- template helpers
def _when(ts: str | None, *, with_time: bool = True) -> str:
    parsed = parse_iso(ts)
    return format_it(parsed, with_time=with_time) if parsed else "—"


def _clock(ts: str | None) -> str:
    parsed = parse_iso(ts)
    return f"{to_local(parsed):%H:%M:%S}" if parsed else "—"


def _ago(ts: str | None) -> str:
    parsed = parse_iso(ts)
    if parsed is None:
        return "—"
    seconds = int((now_utc() - parsed).total_seconds())
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}min"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}g"


def _localinput(ts: str | None) -> str:
    """Format for <input type="datetime-local">, which wants naive local time.

    Stored timestamps are UTC, and handing a UTC string straight to the input would show the
    user a time an hour or two off from the one they are correcting — which is exactly the
    field they are most likely to be fixing.
    """
    parsed = parse_iso(ts)
    return f"{to_local(parsed):%Y-%m-%dT%H:%M}" if parsed else ""


templates.env.filters["when"] = _when
templates.env.filters["clock"] = _clock
templates.env.filters["ago"] = _ago
templates.env.filters["localinput"] = _localinput
templates.env.globals["detail_of"] = activity.detail_of
templates.env.globals["payload_of"] = repo.proposal_payload
templates.env.globals["describe_proposal"] = resolve.describe


def _page(request: Request, template: str, **context: Any) -> HTMLResponse:
    running = activity.running()
    return templates.TemplateResponse(
        request,
        template,
        {
            "running": running,
            "pending_count": repo.pending_proposal_count(),
            "nav_now": format_it(now_local()),
            **context,
        },
    )


# ---------------------------------------------------------------- dashboard
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    today = date.today()
    today_start, today_end = day_bounds_utc(today)
    _, week_end = day_bounds_utc(today, days=8)
    since = iso_utc(now_utc() - timedelta(days=7))

    return _page(
        request,
        "dashboard.html",
        today_events=repo.events_between(today_start, today_end),
        week_events=repo.events_between(today_end, week_end)[:10],
        proposals=repo.pending_proposals(limit=6),
        tasks=repo.open_tasks(limit=6),
        overdue={r["id"] for r in repo.overdue_tasks()},
        categories=repo.category_counts(since=since),
        recent_runs=activity.recent(8),
        feedback=repo.feedback_counts(),
    )


# ---------------------------------------------------------------- agents / activity
@app.get("/agents", response_class=HTMLResponse)
def agents(request: Request) -> HTMLResponse:
    """The "what is going on" view: the roster, what is live, and what just ran."""
    specs = []
    for spec in roster.ALL:
        model = registry.get_spec(spec.task)
        specs.append(
            {
                "name": spec.name,
                "task": spec.task,
                "model": model.model,
                "device": model.device,
                "temperature": model.temperature,
                "tools": spec.tool_names,
                "max_iterations": spec.max_iterations,
            }
        )

    stats = {r["actor"]: r for r in activity.actor_stats()}
    return _page(
        request,
        "agents.html",
        specs=specs,
        stats=stats,
        recent_runs=activity.recent(25),
        intents=roster.BY_INTENT,
    )


@app.get("/activity/live", response_class=HTMLResponse)
def activity_live(request: Request) -> HTMLResponse:
    """Polled fragment: the live strip plus the latest runs. htmx swaps this in place."""
    return templates.TemplateResponse(
        request,
        "_live.html",
        {"running": activity.running(), "recent_runs": activity.recent(12)},
    )


@app.get("/activity/{activity_id}", response_class=HTMLResponse)
def activity_detail(request: Request, activity_id: str) -> HTMLResponse:
    row = activity.get(activity_id)
    if row is None:
        return _page(request, "missing.html", what=f"attività {activity_id}")
    return _page(
        request,
        "activity_detail.html",
        run=row,
        traces=activity.traces_for(activity_id),
    )


@app.get("/api/activity")
def api_activity() -> JSONResponse:
    """For anything that wants to watch Donna without scraping HTML."""
    return JSONResponse(
        {
            "running": [
                {
                    "id": r["id"],
                    "kind": r["kind"],
                    "actor": r["actor"],
                    "summary": r["summary"],
                    "started_at": r["started_at"],
                    "detail": activity.detail_of(r),
                }
                for r in activity.running()
            ],
            "recent": [
                {
                    "id": r["id"],
                    "kind": r["kind"],
                    "actor": r["actor"],
                    "status": r["status"],
                    "duration_ms": r["duration_ms"],
                    "summary": r["summary"],
                    "detail": activity.detail_of(r),
                }
                for r in activity.recent(20)
            ],
        }
    )


# ---------------------------------------------------------------- proposals
@app.get("/proposals", response_class=HTMLResponse)
def proposals(request: Request) -> HTMLResponse:
    pending = repo.pending_proposals(limit=50)
    enriched = [
        {
            "row": row,
            "payload": repo.proposal_payload(row),
            "email": repo.get_email(row["source_id"]) if row["source_type"] == "email" else None,
        }
        for row in pending
    ]
    return _page(
        request,
        "proposals.html",
        proposals=enriched,
        history=repo.resolved_proposals(limit=15),
    )


@app.post("/proposals/{proposal_id}/accept")
def accept_proposal(proposal_id: int) -> RedirectResponse:
    try:
        resolve.accept(proposal_id, via="web")
    except resolve.ProposalError as exc:
        logger.warning("Accept da web fallito per %s: %s", proposal_id, exc)
    return RedirectResponse("/proposals", status_code=303)


@app.post("/proposals/{proposal_id}/reject")
def reject_proposal(proposal_id: int, note: str = Form("")) -> RedirectResponse:
    try:
        resolve.reject(proposal_id, via="web", note=note or None)
    except resolve.ProposalError as exc:
        logger.warning("Reject da web fallito per %s: %s", proposal_id, exc)
    return RedirectResponse("/proposals", status_code=303)


@app.post("/proposals/{proposal_id}/edit")
def edit_proposal(
    proposal_id: int,
    title: str = Form(...),
    start_local: str = Form(...),
    end_local: str = Form(""),
    location: str = Form(""),
    all_day: str = Form(""),
) -> RedirectResponse:
    """Edit and accept.

    The most valuable interaction in the whole dashboard: it records what the right answer was,
    which is the only feedback kind that can supervise a fine-tune rather than merely rank.
    """
    start = parse_iso(start_local)
    end = parse_iso(end_local) if end_local else None
    if start is None:
        logger.warning("Edit da web con data non valida: %r", start_local)
        return RedirectResponse("/proposals", status_code=303)

    try:
        resolve.edit_and_accept(
            proposal_id,
            title=title.strip() or None,
            start_ts=iso_utc(start),
            end_ts=iso_utc(end) if end else None,
            location=location.strip() or None,
            all_day=bool(all_day),
            via="web",
        )
    except resolve.ProposalError as exc:
        logger.warning("Edit da web fallito per %s: %s", proposal_id, exc)
    return RedirectResponse("/proposals", status_code=303)


# ---------------------------------------------------------------- inbox
@app.get("/inbox", response_class=HTMLResponse)
def inbox(request: Request, category: str = "importante", days: int = 14) -> HTMLResponse:
    since = iso_utc(now_utc() - timedelta(days=days))
    rows = (
        repo.emails_by_category(category, since=since, limit=60)
        if category in repo.CATEGORIES
        else repo.recent_emails(limit=60)
    )
    return _page(
        request,
        "inbox.html",
        emails=rows,
        category=category,
        days=days,
        categories=repo.category_counts(since=since),
        signals=repo.signal_counts(),
    )


@app.post("/inbox/{email_id}/reclassify")
def reclassify(email_id: str, category: str = Form(...), note: str = Form("")) -> RedirectResponse:
    triage.reclassify(email_id, category, note=note or None)
    return RedirectResponse("/inbox", status_code=303)


# ---------------------------------------------------------------- chat
@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request) -> HTMLResponse:
    return _page(
        request,
        "chat.html",
        turns=repo.conversation(CHANNEL, CHAT_ID, limit=40),
    )


@app.post("/chat")
def chat_send(message: str = Form(...)) -> RedirectResponse:
    text = message.strip()
    if text:
        try:
            orchestrator.handle(text, channel=CHANNEL, chat_id=CHAT_ID)
        except Exception:
            logger.exception("Turno web fallito")
    return RedirectResponse("/chat", status_code=303)


@app.post("/chat/reset")
def chat_reset() -> RedirectResponse:
    orchestrator.reset(CHANNEL, CHAT_ID)
    return RedirectResponse("/chat", status_code=303)


# ---------------------------------------------------------------- memory
@app.get("/memory", response_class=HTMLResponse)
def memory_page(request: Request) -> HTMLResponse:
    from donna.context import memory

    return _page(request, "memory.html", facts=memory.all_facts())


@app.post("/memory/add")
def memory_add(text: str = Form(...), always_on: str = Form("")) -> RedirectResponse:
    from donna.context import memory

    if text.strip():
        memory.remember(
            text.strip(), source="web", confidence=1.0, always_on=bool(always_on)
        )
    return RedirectResponse("/memory", status_code=303)


@app.post("/memory/{fact_id}/forget")
def memory_forget(fact_id: int) -> RedirectResponse:
    from donna.context import memory

    memory.forget(fact_id)
    return RedirectResponse("/memory", status_code=303)


# ---------------------------------------------------------------- training data
@app.get("/training", response_class=HTMLResponse)
def training(request: Request) -> HTMLResponse:
    """The feedback loop, made visible.

    Shows the examples that have accumulated and what each one can be used for, so it is
    obvious whether the dataset is growing and which kind is scarce.
    """
    examples = list(dataset.iter_examples(limit=200))
    return _page(
        request,
        "training.html",
        examples=examples,
        counts=repo.feedback_counts(),
        by_kind={
            kind: sum(1 for e in examples if e.kind == kind)
            for kind in (dataset.TRIAGE, dataset.EXTRACTION, dataset.PROPOSAL)
        },
    )


@app.get("/training/export")
def training_export(kind: str = "") -> FileResponse:
    settings = get_settings()
    target = settings.db_path.parent / "datasets" / f"feedback_{kind or 'all'}.jsonl"
    written = dataset.export_jsonl(target, kind=kind or None)
    logger.info("Export dataset: %d esempi", written)
    return FileResponse(target, filename=target.name, media_type="application/x-ndjson")


# ---------------------------------------------------------------- traces
@app.get("/traces", response_class=HTMLResponse)
def traces(request: Request, task: str = "") -> HTMLResponse:
    db = get_db()
    if task:
        rows = db.query(
            "SELECT * FROM traces WHERE task = ? ORDER BY created_at DESC LIMIT 60", (task,)
        )
    else:
        rows = db.query("SELECT * FROM traces ORDER BY created_at DESC LIMIT 60")
    per_task = db.query(
        "SELECT task, count(*) n, avg(latency_ms) avg_ms, sum(tokens_in) tin,"
        " sum(tokens_out) tout FROM traces GROUP BY task ORDER BY n DESC"
    )
    return _page(request, "traces.html", traces=rows, per_task=per_task, task=task)


@app.get("/traces/{trace_id}", response_class=HTMLResponse)
def trace_detail(request: Request, trace_id: str) -> HTMLResponse:
    row = get_db().query_one("SELECT * FROM traces WHERE id = ?", (trace_id,))
    if row is None:
        return _page(request, "missing.html", what=f"trace {trace_id}")
    return _page(request, "trace_detail.html", trace=row)


# ---------------------------------------------------------------- health
@app.get("/health")
def health() -> JSONResponse:
    db = get_db()
    return JSONResponse(
        {
            "ok": True,
            "emails": db.scalar("SELECT count(*) FROM emails", default=0),
            "events": db.scalar("SELECT count(*) FROM events", default=0),
            "pending_proposals": repo.pending_proposal_count(),
            "running": len(activity.running()),
        }
    )
