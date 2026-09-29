"""Logging configuration.

This lived inside services/google_auth.py, which meant importing an auth module had the
side effect of reconfiguring logging for the whole process. It is now explicit and
idempotent: call setup_logging() once from an entrypoint.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from donna.config import get_settings

_configured = False

# Third-party loggers that are chatty at INFO.
_NOISY = (
    "httpx",
    "httpcore",
    "googleapiclient.discovery_cache",
    "googleapiclient.discovery",
    "apscheduler.executors.default",
    "telegram.ext.Application",
    "hpack",
)


def setup_logging(level: str | None = None, *, force: bool = False) -> None:
    global _configured
    if _configured and not force:
        return

    settings = get_settings()
    resolved = (level or settings.log_level).upper()

    root = logging.getLogger()
    root.setLevel(resolved)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        settings.log_file,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True
