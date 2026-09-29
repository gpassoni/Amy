"""Pushing proposals to Telegram without being asked.

This is what makes Amy proactive rather than pull-only, and it is the piece v1 had no way to
do: with no place to park a suggestion and no scheduler, nothing could ever happen unless the
user typed first.

The notifier runs on the background scheduler, which lives in a different thread from the bot's
event loop. Sending from a thread therefore has to be marshalled back onto the loop —
`asyncio.run_coroutine_threadsafe` — rather than called directly, which would either fail or
corrupt the loop's state.

`notified_at` is set only after a successful send, so a network failure means the proposal is
retried on the next tick rather than silently swallowed.
"""

from __future__ import annotations

import asyncio
import logging

from telegram import Bot
from telegram.error import TelegramError

from amy.config import get_settings
from amy.interfaces.telegram import cards, keyboards
from amy.store import repo, state

logger = logging.getLogger(__name__)

# Per tick. A backlog arriving all at once reads as spam, and the rest will follow shortly.
MAX_PER_TICK = 3


class Notifier:
    """Sends proposal cards from the scheduler thread into the bot's event loop."""

    def __init__(self, bot: Bot, loop: asyncio.AbstractEventLoop) -> None:
        self._bot = bot
        self._loop = loop

    @property
    def chat_id(self) -> int | None:
        return get_settings().telegram_chat_id or state.get_int(state.TELEGRAM_CHAT_ID)

    def push_pending(self) -> int:
        """Send unnotified proposals. Safe to call from any thread. Returns how many went out."""
        chat_id = self.chat_id
        if chat_id is None:
            # Normal before the first /start; there is nowhere to send yet.
            logger.debug("Nessuna chat Telegram associata: niente da notificare")
            return 0

        rows = repo.unnotified_proposals(MAX_PER_TICK)
        if not rows:
            return 0

        sent = 0
        for row in rows:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._bot.send_message(
                        chat_id=chat_id,
                        text=cards.proposal_card(row),
                        reply_markup=keyboards.proposal_keyboard(row["id"]),
                    ),
                    self._loop,
                )
                future.result(timeout=30)
            except (TelegramError, TimeoutError, RuntimeError) as exc:
                # Deliberately not marked as notified: the next tick retries it.
                logger.warning("Non ho potuto notificare la proposta %s: %s", row["id"], exc)
                continue

            repo.mark_proposal_notified(row["id"])
            sent += 1

        if sent:
            logger.info("Notificate %d proposte su Telegram", sent)
        return sent

    def send_text(self, text: str) -> bool:
        """Send a plain message, used by the morning briefing."""
        chat_id = self.chat_id
        if chat_id is None:
            return False
        try:
            for part in cards.split(text):
                future = asyncio.run_coroutine_threadsafe(
                    self._bot.send_message(chat_id=chat_id, text=part), self._loop
                )
                future.result(timeout=30)
        except (TelegramError, TimeoutError, RuntimeError) as exc:
            logger.warning("Non ho potuto inviare il messaggio: %s", exc)
            return False
        return True
