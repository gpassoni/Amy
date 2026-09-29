"""Building the Telegram application."""

from __future__ import annotations

import logging

from telegram import BotCommand
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from amy.config import get_settings
from amy.interfaces.telegram import handlers

logger = logging.getLogger(__name__)


def build() -> Application:
    settings = get_settings()
    if not settings.telegram_bot_token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN non impostato. Prendi un token da @BotFather e mettilo in .env."
        )

    app = ApplicationBuilder().token(settings.telegram_bot_token).build()

    app.add_handler(CommandHandler("start", handlers.start))
    app.add_handler(CommandHandler("reset", handlers.reset))
    app.add_handler(CommandHandler(["proposte", "proposals"], handlers.proposals))
    app.add_handler(CommandHandler(["briefing", "punto"], handlers.briefing))
    app.add_handler(CallbackQueryHandler(handlers.button))
    # Last, so it only catches what the command handlers did not.
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.message))
    app.add_error_handler(handlers.on_error)

    return app


# Shown in Telegram's own command menu, so the commands are discoverable without a help text.
COMMANDS = [
    BotCommand("briefing", "Il punto della situazione"),
    BotCommand("proposte", "Gli impegni che ti ho trovato e aspettano una risposta"),
    BotCommand("reset", "Dimentica la conversazione"),
    BotCommand("start", "Chi sono e cosa faccio"),
]


async def register_commands(app: Application) -> None:
    """Publish the command menu. A failure here is cosmetic, so it never blocks startup."""
    try:
        await app.bot.set_my_commands(COMMANDS)
        logger.info("Menu comandi Telegram registrato")
    except Exception:
        logger.warning("Non ho potuto registrare il menu comandi", exc_info=True)
