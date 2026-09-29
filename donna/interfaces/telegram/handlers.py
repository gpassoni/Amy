"""Telegram handlers.

Two things worth knowing about the shape of this module.

**Everything model-facing runs in a worker thread.** Inference and the Google clients are
blocking and synchronous, and running them on the event loop would freeze the bot — no typing
indicator, no other messages processed — for the several seconds a turn takes. `asyncio.to_thread`
keeps the loop free.

**Only the owner is obeyed.** The bot token is a bearer credential: anyone who learns the
bot's name can message it. Donna reads a real mailbox and writes to a real calendar, so every
handler checks the chat against the one she was bound to, and the first `/start` is what binds
her.
"""

from __future__ import annotations

import asyncio
import logging

from telegram import Update
from telegram.constants import ChatAction
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from donna.agents import orchestrator
from donna.config import get_settings
from donna.interfaces.telegram import cards, keyboards
from donna.pipeline import resolve
from donna.store import repo, state

logger = logging.getLogger(__name__)

CHANNEL = "telegram"

WELCOME = """Sono Donna. E so già perché sei qui.

Mi occupo io di posta, calendario e cose da fare. Leggo le email, capisco quali contano e
quando c'è dentro un impegno te lo propongo: un tocco e va in calendario.

Scrivimi come ti viene. Se vuoi i comandi: /briefing per il punto della situazione,
/proposte per quelle in attesa, /reset per farmi dimenticare la conversazione."""

NOT_AUTHORISED = "Non ci conosciamo, e io lavoro per una persona sola."


def _authorised(update: Update) -> bool:
    """Whether this chat is the one Donna belongs to.

    The first /start binds her when no chat is configured, which is how you set this up
    without having to look up your own chat id. Every later message from anyone else is
    ignored.
    """
    chat = update.effective_chat
    if chat is None:
        return False

    configured = get_settings().telegram_chat_id or state.get_int(state.TELEGRAM_CHAT_ID)
    if configured is None:
        state.set_value(state.TELEGRAM_CHAT_ID, str(chat.id))
        logger.info("Donna associata alla chat Telegram %s", chat.id)
        return True
    return chat.id == configured


async def _guard(update: Update) -> bool:
    if _authorised(update):
        return True
    logger.warning(
        "Messaggio ignorato da chat non autorizzata %s",
        update.effective_chat.id if update.effective_chat else "?",
    )
    if update.message is not None:
        await update.message.reply_text(NOT_AUTHORISED)
    elif update.callback_query is not None:
        await update.callback_query.answer(NOT_AUTHORISED, show_alert=True)
    return False


# ---------------------------------------------------------------- commands
async def start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    await update.message.reply_text(WELCOME)


async def reset(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    removed = await asyncio.to_thread(orchestrator.reset, CHANNEL, str(update.effective_chat.id))
    await update.message.reply_text(
        f"Fascicolo archiviato: {removed} messaggi dimenticati.\n"
        "Quello che so di te resta — solo la conversazione ricomincia."
    )


async def proposals(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """Re-send every pending proposal as its own card with buttons."""
    if not await _guard(update):
        return

    rows = await asyncio.to_thread(repo.pending_proposals, 10)
    if not rows:
        await update.message.reply_text("Niente in attesa. Sei in pari.")
        return

    for row in rows:
        await update.message.reply_text(
            cards.proposal_card(row), reply_markup=keyboards.proposal_keyboard(row["id"])
        )
        await asyncio.to_thread(repo.mark_proposal_notified, row["id"])


async def briefing(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    await update.effective_chat.send_action(ChatAction.TYPING)
    result = await asyncio.to_thread(
        orchestrator.briefing, channel=CHANNEL, chat_id=str(update.effective_chat.id)
    )
    for part in cards.split(result.text):
        await update.message.reply_text(part)


# ---------------------------------------------------------------- conversation
async def message(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    if update.message is None or not update.message.text:
        return

    chat_id = str(update.effective_chat.id)
    text = update.message.text

    # A local turn takes a couple of seconds, so show the indicator and refresh it while the
    # model works — Telegram expires it after ~5s and silence reads as a broken bot.
    typing = asyncio.create_task(_keep_typing(update))
    try:
        result = await asyncio.to_thread(
            orchestrator.handle, text, channel=CHANNEL, chat_id=chat_id
        )
    except Exception:
        logger.exception("Turno Telegram fallito")
        await update.message.reply_text(
            "Ho avuto un problema tecnico. Riprova — e se insiste, guarda i log."
        )
        return
    finally:
        typing.cancel()

    for part in cards.split(result.text):
        await update.message.reply_text(part)

    # If that turn created proposals (it can, via a tool), surface them with buttons rather
    # than leaving them to the next notifier tick.
    await _push_new_proposals(update)


async def _keep_typing(update: Update) -> None:
    try:
        while True:
            await update.effective_chat.send_action(ChatAction.TYPING)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass


async def _push_new_proposals(update: Update) -> None:
    rows = await asyncio.to_thread(repo.unnotified_proposals, 5)
    for row in rows:
        await update.message.reply_text(
            cards.proposal_card(row), reply_markup=keyboards.proposal_keyboard(row["id"])
        )
        await asyncio.to_thread(repo.mark_proposal_notified, row["id"])


# ---------------------------------------------------------------- buttons
async def button(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle a tap on a proposal card."""
    if not await _guard(update):
        return

    query = update.callback_query
    # Answer immediately: Telegram shows a spinner on the button until this returns, and the
    # work below can take a second or two.
    await query.answer()

    callback = keyboards.parse(query.data or "")
    if callback is None or callback.kind != keyboards.PROPOSAL:
        await query.edit_message_text("Questo pulsante non è più valido.")
        return

    row = await asyncio.to_thread(repo.get_proposal, callback.target)
    if row is None:
        await query.edit_message_text("Questa proposta non esiste più.")
        return

    if callback.action == keyboards.WHY:
        await _safe_edit(query, cards.why_card(row), keyboards.why_keyboard(row["id"]))
        return

    if callback.action == keyboards.LATER:
        # Left pending on purpose: it will come back in /proposte and in the briefing, and
        # will expire on its own if it is never answered.
        await _safe_edit(query, cards.proposal_card(row) + "\n\n⏳ La ritiro fuori più tardi.")
        return

    if callback.action == keyboards.ACCEPT:
        try:
            resolution = await asyncio.to_thread(resolve.accept, callback.target, via="telegram")
        except resolve.ProposalError as exc:
            await _safe_edit(
                query,
                f"{cards.proposal_card(row)}\n\n❌ {exc}",
                keyboards.proposal_keyboard(row["id"]),
            )
            return
        await _safe_edit(query, cards.resolved_card(row, resolution))
        return

    if callback.action == keyboards.REJECT:
        resolution = await asyncio.to_thread(resolve.reject, callback.target, via="telegram")
        await _safe_edit(query, cards.resolved_card(row, resolution))
        return

    await _safe_edit(query, "Azione non riconosciuta.")


async def _safe_edit(query, text: str, markup=None) -> None:
    """Edit a message, tolerating Telegram's "message is not modified" error.

    Tapping the same button twice produces identical text, and Telegram treats a no-op edit as
    a 400. That is not worth surfacing to the user.
    """
    try:
        await query.edit_message_text(text, reply_markup=markup)
    except BadRequest as exc:
        if "not modified" not in str(exc).lower():
            logger.warning("Impossibile aggiornare il messaggio: %s", exc)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Errore non gestito nel bot", exc_info=context.error)
