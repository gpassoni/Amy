"""Installation state: small facts about this deployment rather than about the user.

Exists because of the Telegram chat id, which is genuinely awkward to place. It cannot come
from .env on first run — you do not know your own chat id until you have messaged the bot —
and it has to survive restarts. Making the first /start bind it and storing it here means
setup is "message the bot", with no file to edit.
"""

from __future__ import annotations

from donna.store.db import get_db
from donna.timeutil import iso_utc, now_utc

TELEGRAM_CHAT_ID = "telegram_chat_id"
LAST_BRIEFING_AT = "last_briefing_at"


def get_value(key: str) -> str | None:
    return get_db().scalar("SELECT value FROM app_state WHERE key = ?", (key,))


def set_value(key: str, value: str | None) -> None:
    get_db().execute(
        "INSERT INTO app_state (key, value, updated_at) VALUES (?,?,?)"
        " ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        (key, value, iso_utc(now_utc())),
    )


def get_int(key: str) -> int | None:
    raw = get_value(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def clear(key: str) -> bool:
    return get_db().execute("DELETE FROM app_state WHERE key = ?", (key,)).rowcount > 0
