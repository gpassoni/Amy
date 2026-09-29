"""Telegram layer: callback encoding, cards, and the authorisation guard.

The guard gets the most attention here. A bot token is a bearer credential — anyone who finds
the bot can message it — and Amy reads a real mailbox and writes to a real calendar. A hole
in `_authorised` is not a cosmetic bug.
"""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from amy.config import get_settings
from amy.interfaces.telegram import cards, keyboards
from amy.interfaces.telegram.handlers import _authorised
from amy.store import repo, state
from amy.store.db import Database
from amy.timeutil import iso_utc, now_local, now_utc, to_utc


# ---------------------------------------------------------------- callback data
def test_callback_round_trip():
    original = keyboards.Callback(keyboards.PROPOSAL, keyboards.ACCEPT, 17)
    parsed = keyboards.parse(original.data)
    assert parsed == original


def test_callback_data_fits_telegrams_limit():
    # Telegram rejects callback_data over 64 bytes, and a rejected button is a dead button.
    data = keyboards.Callback(keyboards.PROPOSAL, keyboards.WHY, 999_999_999).data
    assert len(data.encode("utf-8")) <= 64


@pytest.mark.parametrize("bad", ["", "nonsense", "p:ok", "p:ok:abc", "p:ok:1:2", None, "::"])
def test_unparseable_callbacks_return_none(bad):
    # Old messages keep working buttons across restarts, so junk is expected input.
    assert keyboards.parse(bad) is None


def test_every_button_on_a_proposal_keyboard_parses():
    markup = keyboards.proposal_keyboard(42)
    buttons = [b for row in markup.inline_keyboard for b in row]
    assert len(buttons) == 4
    for button in buttons:
        parsed = keyboards.parse(button.callback_data)
        assert parsed is not None
        assert parsed.target == 42


def test_the_destructive_button_is_not_alone_on_its_row():
    # Accept and reject sit together on the top row; a mis-tap should be plausible either way
    # rather than the reject being the easiest thing to hit.
    markup = keyboards.proposal_keyboard(1)
    top = [keyboards.parse(b.callback_data).action for b in markup.inline_keyboard[0]]
    assert keyboards.ACCEPT in top and keyboards.REJECT in top


# ---------------------------------------------------------------- cards
def _proposal(db: Database, **payload_overrides) -> object:
    # Pinned to a known local time so assertions about the rendered clock time mean something.
    start = to_utc(
        now_local().replace(hour=15, minute=0, second=0, microsecond=0) + timedelta(days=2)
    )
    payload = {
        "title": "Igiene dentale",
        "start_ts": iso_utc(start),
        "end_ts": iso_utc(start + timedelta(hours=1)),
        "all_day": False,
        "location": "via Verdi 12",
        "date_phrase": "giovedì alle 15:00",
        "date_source": "phrase",
    }
    payload.update(payload_overrides)
    proposal_id = repo.create_proposal(
        kind="calendar_event",
        source_type="email",
        source_id="m1",
        payload=payload,
        reasoning="Studio Bianchi indica giovedì alle 15:00",
        evidence_quote="le confermiamo l'appuntamento di giovedì alle 15:00",
        confidence=0.95,
        trace_id="t1",
    )
    return repo.get_proposal(proposal_id)


def test_proposal_card_names_the_thing_and_the_time(db: Database):
    text = cards.proposal_card(_proposal(db))
    assert "Igiene dentale" in text
    assert "via Verdi 12" in text
    assert "15:00" in text


def test_proposal_card_shows_the_sender_when_the_email_is_mirrored(db: Database):
    now = iso_utc(now_utc())
    repo.upsert_email(
        id="m1",
        thread_id="t1",
        from_addr="studio@x.it",
        from_name="Studio Bianchi",
        to_addrs="me",
        subject="Conferma appuntamento",
        snippet="",
        body="",
        received_at=now,
        label_ids=[],
        is_unread=True,
    )
    text = cards.proposal_card(_proposal(db))
    assert "Studio Bianchi" in text
    assert "Conferma appuntamento" in text


def test_a_low_confidence_proposal_says_so(db: Database):
    start = now_utc() + timedelta(days=2)
    proposal_id = repo.create_proposal(
        kind="calendar_event",
        source_type="email",
        source_id="m9",
        payload={
            "title": "Forse qualcosa",
            "start_ts": iso_utc(start),
            "end_ts": iso_utc(start + timedelta(hours=1)),
            "all_day": False,
        },
        reasoning="stimata",
        evidence_quote=None,
        confidence=0.45,
        trace_id="t2",
    )
    text = cards.proposal_card(repo.get_proposal(proposal_id))
    # Presenting a shaky guess with the same certainty as a solid one is how you train
    # someone to stop reading the cards.
    assert "sicurissima" in text
    assert "45%" in text


def test_a_confident_proposal_does_not_hedge(db: Database):
    assert "sicurissima" not in cards.proposal_card(_proposal(db))


def test_an_all_day_proposal_reads_as_a_day_not_a_time(db: Database):
    start = now_utc() + timedelta(days=3)
    row = _proposal(
        db,
        all_day=True,
        title="Pagamento bollo",
        start_ts=iso_utc(start.replace(hour=0, minute=0, second=0, microsecond=0)),
        end_ts=iso_utc(
            start.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        ),
    )
    assert "tutto il giorno" in cards.proposal_card(row)


def test_why_card_leads_with_the_evidence(db: Database):
    text = cards.why_card(_proposal(db))
    assert "le confermiamo l'appuntamento" in text
    assert "giovedì alle 15:00" in text
    assert "ricavata dal testo" in text


def test_why_card_admits_when_a_date_was_only_estimated(db: Database):
    row = _proposal(db, date_source="model", date_phrase=None)
    assert "stimata da me" in cards.why_card(row)


def test_cards_stay_inside_telegrams_message_limit(db: Database):
    row = _proposal(db, title="X" * 500)
    assert len(cards.proposal_card(row)) <= cards.MAX_LEN


# ---------------------------------------------------------------- splitting
def test_short_text_is_not_split():
    assert cards.split("ciao") == ["ciao"]


def test_empty_text_still_produces_a_message():
    # An empty send is an API error; better a placeholder than a crash.
    assert cards.split("") == ["(nessuna risposta)"]


def test_long_text_splits_on_line_boundaries():
    text = "\n".join(f"riga numero {i}" for i in range(500))
    parts = cards.split(text, limit=200)
    assert len(parts) > 1
    assert all(len(p) <= 200 for p in parts)
    # No line should be cut in half.
    assert all(not p.startswith("umero") for p in parts)


def test_splitting_loses_nothing():
    text = "\n".join(f"riga {i}" for i in range(200))
    rejoined = " ".join(cards.split(text, limit=100)).replace("\n", " ")
    for i in (0, 99, 199):
        assert f"riga {i}" in rejoined


def test_a_single_unbroken_word_is_still_split():
    parts = cards.split("X" * 500, limit=100)
    assert len(parts) == 5


# ---------------------------------------------------------------- authorisation
def _update(chat_id: int) -> object:
    return SimpleNamespace(effective_chat=SimpleNamespace(id=chat_id))


def test_the_first_chat_binds_amy(db: Database):
    assert state.get_int(state.TELEGRAM_CHAT_ID) is None
    assert _authorised(_update(555)) is True
    assert state.get_int(state.TELEGRAM_CHAT_ID) == 555


def test_a_bound_chat_stays_authorised(db: Database):
    _authorised(_update(555))
    assert _authorised(_update(555)) is True


def test_any_other_chat_is_refused(db: Database):
    """The security case: a bot token is a bearer credential.

    Anyone who discovers the bot can message it, and Amy reads a real mailbox and writes to a
    real calendar. Binding to one chat is what stops a stranger from driving her.
    """
    _authorised(_update(555))
    assert _authorised(_update(999)) is False
    assert state.get_int(state.TELEGRAM_CHAT_ID) == 555  # not rebound


def test_an_explicitly_configured_chat_id_wins(db: Database, monkeypatch):
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "111")
    get_settings.cache_clear()
    try:
        assert _authorised(_update(111)) is True
        assert _authorised(_update(555)) is False
    finally:
        get_settings.cache_clear()


def test_an_update_without_a_chat_is_refused(db: Database):
    assert _authorised(SimpleNamespace(effective_chat=None)) is False


# ---------------------------------------------------------------- notifier
def test_the_notifier_does_nothing_before_a_chat_is_bound(db: Database):
    from amy.interfaces.telegram.notifier import Notifier

    _proposal(db)
    notifier = Notifier(bot=MagicMock(), loop=None)  # loop unused on this path
    assert notifier.push_pending() == 0
    # The proposal is left unnotified so it goes out once a chat exists.
    assert len(repo.unnotified_proposals(5)) == 1


def test_a_failed_send_leaves_the_proposal_unnotified(db: Database):
    """Otherwise a transient network error silently loses the proposal forever."""
    from amy.interfaces.telegram import notifier as notifier_module

    _proposal(db)
    state.set_value(state.TELEGRAM_CHAT_ID, "555")
    notifier = notifier_module.Notifier(bot=MagicMock(), loop=object())

    with patch.object(
        notifier_module.asyncio, "run_coroutine_threadsafe", side_effect=RuntimeError("no loop")
    ):
        assert notifier.push_pending() == 0

    assert len(repo.unnotified_proposals(5)) == 1


def test_a_successful_send_marks_the_proposal_notified(db: Database):
    from amy.interfaces.telegram import notifier as notifier_module

    _proposal(db)
    state.set_value(state.TELEGRAM_CHAT_ID, "555")
    notifier = notifier_module.Notifier(bot=MagicMock(), loop=object())

    class _Future:
        def result(self, timeout=None):
            return None

    with patch.object(notifier_module.asyncio, "run_coroutine_threadsafe", return_value=_Future()):
        assert notifier.push_pending() == 1

    assert repo.unnotified_proposals(5) == []


# ---------------------------------------------------------------- app state
def test_app_state_round_trip(db: Database):
    state.set_value("k", "v")
    assert state.get_value("k") == "v"
    state.set_value("k", "v2")
    assert state.get_value("k") == "v2"


def test_app_state_int_coercion_tolerates_junk(db: Database):
    state.set_value("n", "not a number")
    assert state.get_int("n") is None
    state.set_value("n", "42")
    assert state.get_int("n") == 42


def test_missing_app_state_keys_are_none(db: Database):
    assert state.get_value("absent") is None
    assert state.get_int("absent") is None
