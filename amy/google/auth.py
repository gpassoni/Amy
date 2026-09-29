"""Google API client access.

The Singleton + thread-local pattern here is carried over from v1's
services/google_auth.py, which got it right: one OAuth session shared across every thread,
with per-thread service objects because googleapiclient clients are not thread-safe.

Two changes from v1:
  * logging.basicConfig no longer runs at import. Importing an auth module should not
    reconfigure logging for the whole process; amy.logging_setup owns that now.
  * paths come from config instead of module-level os.getenv, so tests and alternate
    profiles work without editing the module.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from amy.config import get_settings

logger = logging.getLogger(__name__)

# gmail.modify allows reading and relabelling, but not sending. Amy has no reason to
# send mail, and keeping send out of the grant means a bug cannot email anyone.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/tasks",
]


class GoogleServiceManager:
    _instance: GoogleServiceManager | None = None
    _lock = threading.Lock()
    _creds: Credentials | None = None
    _thread_local = threading.local()

    def __new__(cls) -> GoogleServiceManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def credentials(self, *, interactive: bool = True) -> Credentials:
        """Load, refresh, or acquire credentials. Thread-safe.

        `interactive=False` refuses to open a browser — background jobs should fail loudly
        rather than silently block on a consent screen nobody is watching.
        """
        with self._lock:
            if self._creds and self._creds.valid:
                return self._creds

            settings = get_settings()
            creds: Credentials | None = None

            if settings.google_token_file.exists():
                creds = Credentials.from_authorized_user_file(
                    str(settings.google_token_file), SCOPES
                )

            if creds and creds.expired and creds.refresh_token:
                try:
                    logger.info("Refreshing Google token")
                    creds.refresh(Request())
                except Exception:
                    logger.warning("Token refresh failed; re-authentication needed", exc_info=True)
                    creds = None

            if not creds or not creds.valid:
                if not interactive:
                    raise RuntimeError(
                        "Credenziali Google non valide e nessuna sessione interattiva "
                        "disponibile. Esegui `python -m amy auth` per autorizzare."
                    )
                if not settings.google_credentials_file.exists():
                    raise FileNotFoundError(
                        f"File credenziali Google mancante: {settings.google_credentials_file}"
                    )
                logger.info("Starting Google OAuth flow")
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(settings.google_credentials_file), SCOPES
                )
                creds = flow.run_local_server(port=0)

            settings.google_token_file.write_text(creds.to_json(), encoding="utf-8")
            self._creds = creds
            return creds

    def service(self, name: str, version: str, *, interactive: bool = True) -> Any:
        if not hasattr(self._thread_local, "services"):
            self._thread_local.services = {}

        key = f"{name}_{version}"
        if key not in self._thread_local.services:
            creds = self.credentials(interactive=interactive)
            # static_discovery avoids the on-disk discovery cache, which deadlocks on
            # Windows when several threads build clients at once.
            self._thread_local.services[key] = build(
                name, version, credentials=creds, static_discovery=True
            )
            logger.debug(
                "Built Google service %s v%s for thread %s", name, version, threading.get_ident()
            )
        return self._thread_local.services[key]


def get_service(name: str, version: str, *, interactive: bool = True) -> Any:
    return GoogleServiceManager().service(name, version, interactive=interactive)


def gmail(*, interactive: bool = True) -> Any:
    return get_service("gmail", "v1", interactive=interactive)


def calendar(*, interactive: bool = True) -> Any:
    return get_service("calendar", "v3", interactive=interactive)


def tasks(*, interactive: bool = True) -> Any:
    return get_service("tasks", "v1", interactive=interactive)


def is_authorized() -> bool:
    """Whether a usable token exists, without triggering a consent flow."""
    try:
        GoogleServiceManager().credentials(interactive=False)
        return True
    except Exception:
        return False
