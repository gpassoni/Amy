"""Output schemas for the pipeline's model calls.

Every field here becomes part of a JSON schema that Ollama compiles into a sampling
grammar, so the shape of these classes is not documentation — it is enforcement. Literal
types in particular make an out-of-vocabulary answer unrepresentable rather than merely
discouraged: during Phase 0 a free-form `str` field with a helpful description still got
"questioning_about_future_activities" back from a small model.

Keep them small. Each additional field is tokens the model has to generate before it can
stop, and on a CPU-hosted 2B that is real latency.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field


def _as_unit_confidence(value: object) -> object:
    """Accept a percentage where a 0-1 fraction was asked for.

    Ollama compiles the JSON schema into a sampling grammar, and a grammar can express
    "a number" but not "a number at most 1.0". Models therefore do return 95 for a field
    documented as 0-1 — observed from the 2b, which answered 100, 95 and 90 in one batch
    and failed validation on all three. Rescaling is strictly better than discarding an
    otherwise correct classification over a units mistake.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 1:
        return min(float(value) / 100.0, 1.0)
    return value


Confidence = Annotated[float, BeforeValidator(_as_unit_confidence), Field(ge=0.0, le=1.0)]

Category = Literal["importante", "da_leggere", "inutile"]
CommitmentKind = Literal["appuntamento", "scadenza", "evento", "viaggio", "nessuno"]

# What kind of thing an email *is*. The model picks one of these; the category is then
# derived in code by SIGNAL_CATEGORY below.
#
# This split is deliberate and it is what made triage work. Asking the 2B directly for
# "importante | da_leggere | inutile" meant asking it to apply a three-way policy over a
# 450-word rulebook, and it did not: 11 of 12 portal notifications came back as
# "da_leggere" when the rules plainly said "inutile". It also had to write a justifying
# sentence, which it cheerfully invented ("comunicazione su abbonamento già attivo" for an
# email that mentioned no subscription) at a cost of ~55 output tokens and 8 seconds each.
#
# Recognising what an email is, is a perception task a small model is good at. Deciding
# what that means for the user is a policy, and policy belongs in code where it is
# readable, testable and adjustable without re-prompting.
Signal = Literal[
    "persona_reale",            # a human wrote to him
    "appuntamento_data",        # a real appointment, summons or booking with a date
    "scadenza_pagamento",       # bill, due date, reminder, renewal charge
    "sicurezza_account",        # genuine security alert
    "ricevuta_ordine",          # receipt, order confirmation
    "ente_istituzione",         # school, bank, doctor, public administration, employer
    "newsletter_scelta",        # a newsletter he opted into
    "aggiornamento_servizio",   # service notice about something already active
    "spedizione",               # shipping update
    "promozione",               # marketing, discount, offer
    "notifica_social",          # "X commented", "3 new views"
    "annuncio_portale",         # automated listing alerts (property, e-commerce, games)
    "registrazione_benvenuto",  # welcome / thanks-for-signing-up
    "spam_phishing",
]

# The policy. Editing this reclassifies the mailbox without touching a prompt or a model.
SIGNAL_CATEGORY: dict[str, Category] = {
    "persona_reale": "importante",
    "appuntamento_data": "importante",
    "scadenza_pagamento": "importante",
    "sicurezza_account": "importante",
    "ricevuta_ordine": "importante",
    "ente_istituzione": "importante",
    "newsletter_scelta": "da_leggere",
    "aggiornamento_servizio": "da_leggere",
    "spedizione": "da_leggere",
    "promozione": "inutile",
    "notifica_social": "inutile",
    "annuncio_portale": "inutile",
    "registrazione_benvenuto": "inutile",
    "spam_phishing": "inutile",
}

# Human-readable text per signal, so the "why" shown to the user is accurate by
# construction rather than generated prose that may not describe the email at all.
SIGNAL_REASON: dict[str, str] = {
    "persona_reale": "scritta da una persona, non da un sistema",
    "appuntamento_data": "contiene un appuntamento con data e ora",
    "scadenza_pagamento": "c'è una scadenza o un pagamento",
    "sicurezza_account": "avviso di sicurezza sull'account",
    "ricevuta_ordine": "ricevuta o conferma d'ordine",
    "ente_istituzione": "comunicazione da un ente o un'istituzione",
    "newsletter_scelta": "newsletter a cui sei iscritto",
    "aggiornamento_servizio": "aggiornamento su un servizio già attivo",
    "spedizione": "aggiornamento di spedizione",
    "promozione": "promozione commerciale",
    "notifica_social": "notifica da una piattaforma social",
    "annuncio_portale": "annuncio automatico di un portale",
    "registrazione_benvenuto": "email di benvenuto o registrazione",
    "spam_phishing": "sembra spam o phishing",
}


class Classification(BaseModel):
    """Triage perception for one email: what it is, and how sure the model is.

    No free-text field on purpose — see the note on `Signal`. The reason the user sees is
    looked up from the signal.
    """

    signal: Signal
    confidence: Confidence

    @property
    def category(self) -> Category:
        return SIGNAL_CATEGORY[self.signal]

    @property
    def reason(self) -> str:
        return SIGNAL_REASON[self.signal]


class Commitment(BaseModel):
    """A thing in an email that might belong on the calendar.

    **Field order here is load-bearing.** Ollama compiles the schema into a sampling grammar
    and generates the properties in declaration order, so the order is the model's order of
    reasoning. The first version of this class asked for `has_commitment` first, and the
    model answered `false` while going on to fill in
    `date_phrase="giovedì 24 settembre alle 15:00"` — it was made to commit to a verdict
    before it had written down any evidence. Seven of seven real appointments were missed
    that way.

    Evidence now comes first, so the verdict is conditioned on tokens the model has already
    produced. This is the structured-output equivalent of letting it show its work.

    The date is also requested twice on purpose: `date_phrase` is the literal text, which
    donna/pipeline/dates.py re-resolves in code, while `start_iso` is the model's own guess
    used only as a cross-check. Small models are unreliable at date arithmetic and reliable
    at quoting the phrase in front of them.
    """

    # --- evidence first: what the model actually found in the text
    evidence: str | None = Field(
        default=None,
        max_length=300,
        description="La frase dell'email che indica l'impegno, copiata alla lettera. Vuoto se non c'è.",
    )
    date_phrase: str | None = Field(
        default=None,
        max_length=80,
        description="Solo le parole che indicano quando, copiate alla lettera. Vuoto se non c'è.",
    )
    start_iso: str | None = Field(
        default=None, description="La tua stima della data, formato YYYY-MM-DDTHH:MM. Vuoto se incerto."
    )
    location: str | None = Field(default=None, max_length=120)

    # --- then the interpretation
    kind: CommitmentKind = "nessuno"
    title: str | None = Field(default=None, max_length=120)
    all_day: bool = False

    # --- and only then the verdict, conditioned on everything above
    has_commitment: bool = False
    confidence: Confidence = 0.0


class RouterDecision(BaseModel):
    """Which agent should handle a message. Used from Phase 3."""

    intent: Literal[
        "schedule_query",
        "schedule_mutate",
        "inbox_query",
        "task_query",
        "task_mutate",
        "proposal_action",
        "briefing",
        "smalltalk",
        "multi",
    ]
    confidence: Confidence


class ExtractedFacts(BaseModel):
    """Durable facts about the user, harvested from conversation."""

    facts: list[str] = Field(default_factory=list, max_length=5)


class Summary(BaseModel):
    summary: str = Field(max_length=400)
