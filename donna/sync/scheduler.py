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
from typing import Callable

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


def build_scheduler(extra_jobs: dict[str, Callable[[], object]] | None = None) -> BackgroundScheduler:
    """Wire the recurring jobs. Caller starts and stops it.

    extra_jobs lets later phases (triage, proposal expiry, notifications) attach without
    this module importing them, keeping the dependency direction one-way.
    """
    settings = get_settings()
    scheduler = BackgroundScheduler(
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 300},
        timezone=settings.calendar_timezone,
    )

    scheduler.add_job(
        run_all_syncs,
        trigger=IntervalTrigger(minutes=settings.sync_interval_minutes),
        id="sync_all",
        name="Sincronizzazione Gmail, Calendar, Tasks",
        next_run_time=None,  # the caller triggers the first pass explicitly
    )

    for job_id, func in (extra_jobs or {}).items():
        scheduler.add_job(
            func,
            trigger=IntervalTrigger(minutes=settings.sync_interval_minutes),
            id=job_id,
            name=job_id,
        )

    return scheduler
