"""Composition root: the one place that wires everything together and starts it.

Threading model, because it is the only genuinely tricky part:

  * the Telegram bot owns the main thread and an asyncio event loop
  * APScheduler runs the pipeline on background worker threads
  * every LLM and Google call is blocking, so it must never touch the bot's loop — handlers
    push that work to `asyncio.to_thread`
  * the notifier goes the other way, marshalling sends from a worker thread back onto the
    loop with `run_coroutine_threadsafe`

The pipeline cycle deliberately runs after the loop exists, so the notifier has a loop to
hand proposals to.
"""
from __future__ import annotations

import logging
from typing import Any

from donna.config import get_settings
from donna.logging_setup import setup_logging
from donna.store.db import get_db

logger = logging.getLogger(__name__)


def _start_web() -> None:
    """Serve the dashboard on its own thread.

    A separate uvicorn loop rather than sharing the bot's: the bot owns its loop and mixing a
    server into it couples two lifecycles that have no reason to be coupled. `daemon=True` means
    the web thread cannot keep the process alive after the bot stops.

    Bound to 127.0.0.1 by default. There is no authentication and the dashboard shows a real
    mailbox, so it should not be reachable from the network without a deliberate decision.
    """
    import threading

    import uvicorn

    settings = get_settings()

    def serve() -> None:
        config = uvicorn.Config(
            "donna.interfaces.web.app:app",
            host=settings.web_host,
            port=settings.web_port,
            log_level="warning",
            access_log=False,
        )
        uvicorn.Server(config).run()

    threading.Thread(target=serve, name="donna-web", daemon=True).start()
    logger.info(
        "Dashboard su http://%s:%d", settings.web_host, settings.web_port
    )


def run() -> int:
    """Start Donna: migrations, scheduler, bot, dashboard. Blocks until interrupted."""
    setup_logging()
    settings = get_settings()

    applied = get_db().migrate()
    if applied:
        logger.info("Migrazioni applicate: %s", ", ".join(applied))

    # A row left 'running' is the fingerprint of a crash mid-turn; clear those so the live view
    # does not show phantom work forever.
    from donna.store import activity

    activity.sweep_stale()

    from donna.interfaces.telegram.bot import build as build_bot
    from donna.interfaces.telegram.notifier import Notifier
    from donna.sync.scheduler import build_scheduler, run_cycle

    app = build_bot()
    scheduler: Any = None
    notifier: Notifier | None = None

    async def _on_start(application) -> None:
        """Once the loop is running, start the background work."""
        nonlocal scheduler, notifier
        import asyncio

        from donna.interfaces.telegram.bot import register_commands

        await register_commands(application)

        loop = asyncio.get_running_loop()
        notifier = Notifier(application.bot, loop)

        def cycle_then_notify() -> None:
            """One pipeline pass, then push whatever it produced.

            Coupled on purpose: a proposal created and not sent is invisible, and the point of
            the whole pipeline is that it reaches the user without being asked.
            """
            try:
                run_cycle()
            except Exception:
                logger.exception("Ciclo fallito")
            try:
                if notifier is not None:
                    notifier.push_pending()
            except Exception:
                logger.exception("Notifica fallita")

        scheduler = build_scheduler(cycle_job=cycle_then_notify)
        scheduler.start()
        logger.info(
            "Scheduler avviato: ciclo ogni %d minuti", settings.sync_interval_minutes
        )

        _start_web()

    async def _on_stop(_application) -> None:
        if scheduler is not None and scheduler.running:
            scheduler.shutdown(wait=False)
            logger.info("Scheduler fermato")

    app.post_init = _on_start
    app.post_shutdown = _on_stop

    logger.info("Donna in ascolto su Telegram")
    app.run_polling(drop_pending_updates=True)
    return 0
