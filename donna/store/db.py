"""SQLite access layer.

One connection per thread, following the same reasoning as the existing
GoogleServiceManager: APScheduler workers, the Telegram event loop and FastAPI all touch
the database, and sqlite3 connections are not safe to share across threads.

WAL mode is on, so background sync writing does not block the web UI reading.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from donna.config import get_settings

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._local = threading.local()
        self._migrate_lock = threading.Lock()

    # ------------------------------------------------------------------ connection
    @property
    def conn(self) -> sqlite3.Connection:
        existing = getattr(self._local, "conn", None)
        if existing is not None:
            return existing

        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------ queries
    def query(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    def scalar(self, sql: str, params: Sequence[Any] | dict[str, Any] = (), default: Any = None) -> Any:
        row = self.query_one(sql, params)
        return row[0] if row is not None else default

    def execute(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, seq: Iterable[Sequence[Any]]) -> sqlite3.Cursor:
        return self.conn.executemany(sql, seq)

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """Explicit transaction. isolation_level=None means we drive BEGIN ourselves."""
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except Exception:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    # ------------------------------------------------------------------ migrations
    def migrate(self) -> list[str]:
        """Apply pending .sql migrations in filename order. Returns the ones applied."""
        with self._migrate_lock:
            conn = self.conn
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " name TEXT PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            done = {r["name"] for r in conn.execute("SELECT name FROM schema_migrations")}

            applied: list[str] = []
            for sql_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
                if sql_file.name in done:
                    continue
                logger.info("Applying migration %s", sql_file.name)
                # executescript() implicitly commits any open transaction before it runs,
                # so the transaction has to live inside the script rather than around it.
                # This keeps each migration all-or-nothing, bookkeeping row included.
                name = sql_file.name.replace("'", "''")
                script = (
                    "BEGIN IMMEDIATE;\n"
                    f"{sql_file.read_text(encoding='utf-8')}\n"
                    f"INSERT INTO schema_migrations(name) VALUES ('{name}');\n"
                    "COMMIT;"
                )
                try:
                    conn.executescript(script)
                except Exception:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                    raise
                applied.append(sql_file.name)
            return applied

    # ------------------------------------------------------------------ helpers
    def table_names(self) -> list[str]:
        rows = self.query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            " ORDER BY name"
        )
        return [r["name"] for r in rows]


_db: Database | None = None
_db_lock = threading.Lock()


def get_db() -> Database:
    """Process-wide Database handle (thread-safe; connections are still per-thread)."""
    global _db
    if _db is None:
        with _db_lock:
            if _db is None:
                _db = Database(get_settings().db_path)
    return _db


def reset_db_for_tests(path: Path | str) -> Database:
    global _db
    with _db_lock:
        _db = Database(path)
    return _db


def upsert(
    table: str,
    row: dict[str, Any],
    *,
    conflict: str = "id",
    preserve: Sequence[str] = (),
) -> str:
    """Build an INSERT .. ON CONFLICT DO UPDATE statement for the given columns.

    Returns SQL; caller supplies `row.values()` as params. Keeping this as a builder rather
    than executing means repos stay explicit about which columns they touch.

    `preserve` names columns that are written on insert but never overwritten on conflict —
    for columns owned by something other than the writer. A sync pass must not clobber
    provenance recorded by the approval flow, in the same way it must not clobber a triage
    verdict.
    """
    cols = list(row)
    placeholders = ", ".join("?" for _ in cols)
    skip = {conflict, *preserve}
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c not in skip)
    if not updates:
        return (
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT({conflict}) DO NOTHING"
        )
    return (
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT({conflict}) DO UPDATE SET {updates}"
    )
