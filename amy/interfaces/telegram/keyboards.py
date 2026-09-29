"""Inline keyboards and the callback payloads behind them.

Telegram allows 64 bytes of callback_data per button, which is the whole reason this is a
module rather than a few f-strings: the payload has to be a compact, parseable code, and both
ends of it need to agree. Encoding it in one place means a typo cannot ship a button that
does nothing when tapped.

Format: `<kind>:<action>:<id>` — e.g. "p:ok:17". Short by design.
"""

from __future__ import annotations

from dataclasses import dataclass

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

PROPOSAL = "p"

ACCEPT = "ok"
REJECT = "no"
WHY = "why"
LATER = "later"


@dataclass(slots=True)
class Callback:
    kind: str
    action: str
    target: int

    @property
    def data(self) -> str:
        return f"{self.kind}:{self.action}:{self.target}"


def parse(data: str) -> Callback | None:
    """Decode callback data. Returns None for anything unrecognised.

    Old buttons survive restarts and redeploys — a message from last week still has tappable
    buttons — so unparseable data is an expected input, not a bug to raise on.
    """
    parts = (data or "").split(":")
    if len(parts) != 3:
        return None
    kind, action, target = parts
    if not target.isdigit():
        return None
    return Callback(kind=kind, action=action, target=int(target))


def proposal_keyboard(proposal_id: int) -> InlineKeyboardMarkup:
    """The approval card's buttons.

    Two rows: the decision on top where the thumb lands, the non-committal options below, so a
    mis-tap is less likely to be the destructive one. "Perché" is there because a proposal the
    user does not understand is one they will dismiss — being able to see the sentence Amy
    based it on is what makes accepting it reasonable.
    """
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ Aggiungi", callback_data=Callback(PROPOSAL, ACCEPT, proposal_id).data
                ),
                InlineKeyboardButton(
                    "🗑 Ignora", callback_data=Callback(PROPOSAL, REJECT, proposal_id).data
                ),
            ],
            [
                InlineKeyboardButton(
                    "🔎 Perché", callback_data=Callback(PROPOSAL, WHY, proposal_id).data
                ),
                InlineKeyboardButton(
                    "⏳ Dopo", callback_data=Callback(PROPOSAL, LATER, proposal_id).data
                ),
            ],
        ]
    )


def why_keyboard(proposal_id: int) -> InlineKeyboardMarkup:
    """After showing the reasoning, the decision still has to be reachable."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ Aggiungi", callback_data=Callback(PROPOSAL, ACCEPT, proposal_id).data
                ),
                InlineKeyboardButton(
                    "🗑 Ignora", callback_data=Callback(PROPOSAL, REJECT, proposal_id).data
                ),
            ]
        ]
    )
