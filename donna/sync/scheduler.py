"""Background scheduling.

APScheduler's BackgroundScheduler rather than AsyncIOScheduler: the Google clients and the
Ollama client are both synchronous and blocking, and running them on the Telegram event
loop would stall message handling for the duration of a sync.

`max_instances=1` and `coalesce=True` on every job, because a sync that overruns its
interval must not stack up behind itself — on a cold model load a triage pass can take
minutes, and three of them running concurrently would fight over the same CPU runner.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from donna.config import get_settings
from donna.sync.base import SyncResult
from donna.sync.calendar_sync import sync_calendar
from donna.sync.gmail_sync import sync_gmail
from donna.sync.tasks_sync import sync_tasks

logger = logging.getLogger(__name__)


def run_all_syncs(*, full_gmail: bool = False) -> list[SyncResult]:
    """Every source, in order. Gmail first: triage depends on it."""
    return [sync_gmail(full=full_gmail), sync_calendar(), sync_tasks()]


def run_cycle(*, full_gmail: bool = False) -> dict[str, object]:
    """One complete pass: sync, triage, extract, expire.

    Sequential and in this order because each step consumes the previous one's output, and
    because they share a single GPU-resident model — running them concurrently would just
    queue requests against the same runner while making the failure modes harder to read.

    Imported lazily so that donna.sync does not depend on donna.pipeline; the dependency
    runs one way, from pipeline to store.
    """
    from donna.config import get_settings
    from donna.pipeline.extract import run_extraction
    from donna.pipeline.resolve import expire_stale
    from donna.pipeline.triage import run_triage

    settings = get_settings()
    syncs = run_all_syncs(full_gmail=full_gmail)
    triaged = run_triage()
    extracted = run_extraction()
    expired = expire_stale(settings.proposal_expiry_days)

    logger.info(
        "Ciclo completo: %s | %s | %s | %d proposte scadute",
        "; ".join(s.summary() for s in syncs),
        triaged.summary(),
        extracted.summary(),
        expired,
    )
    return {
        "syncs": syncs,
        "triage": triaged,
        "extraction": extracted,
        "expired": expired,
    }


def build_scheduler(
    extra_jobs: dict[str, Callable[[], object]] | None = None,
    *,
    cycle_job: Callable[[], object] | None = None,
) -> BackgroundScheduler:
    """Wire the recurring jobs. Caller starts and stops it.

    extra_jobs lets later phases (triage, proposal expiry, notifications) attach without
    this module importing them, keeping the dependency direction one-way.
    """
    settings = get_settings()
    scheduler = BackgroundScheduler(
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 300},
        timezone=settings.calendar_timezone,
    )

    # One job for the whole cycle rather than one per stage: they are strictly sequential
    # and share a single model, so separate jobs would only create the illusion of
    # parallelism while making overlap possible.
    # `cycle_job` lets the composition root wrap the cycle (to push notifications after it)
    # without this module importing an interface. Defaults to the pipeline alone.
    scheduler.add_job(
        cycle_job or run_cycle,
        trigger=IntervalTrigger(minutes=settings.sync_interval_minutes),
        id="cycle",
        name="Sync, triage, estrazione",
    )

    for job_id, func in (extra_jobs or {}).items():
        scheduler.add_job(
            func,
            trigger=IntervalTrigger(minutes=settings.sync_interval_minutes),
            id=job_id,
            name=job_id,
        )

    return scheduler
