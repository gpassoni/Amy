from __future__ import annotations

import pytest

from donna.config import get_settings
from donna.store.db import Database, reset_db_for_tests


@pytest.fixture(autouse=True)
def _fixed_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the zone so date assertions do not depend on the developer's machine."""
    monkeypatch.setenv("CALENDAR_TIMEZONE", "Europe/Rome")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def db(tmp_path) -> Database:
    """A migrated, throwaway database. Never touches the real donna.db."""
    database = reset_db_for_tests(tmp_path / "test.db")
    database.migrate()
    return database
