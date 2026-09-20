"""Hand-written evaluation cases.

Deliberately not drawn from the mirrored mailbox. That account is a test account whose mail
is 78 property-portal alerts and 45 LinkedIn notifications, with not one real appointment
and an empty calendar — tuning extraction against it would bake in the wrong priors and the
work would be invalidated the moment Donna points at a real account.

These cases are written to cover what actually goes wrong: appointments phrased six
different ways, commercial deadlines that must NOT become calendar entries, dates in
signatures and footers, and relative phrasing that needs the email's own receipt date to
resolve.

`expect_start` is local time, resolved against `received` — which is Monday
21 September 2026, 09:00 Europe/Rome for every case, so the expectations are readable.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

# Monday 21 September 2026, 09:00 local (07:00 UTC).
REFERENCE = datetime(2026, 9, 21, 7, 0, tzinfo=timezone.utc)


@dataclass(slots=True)
class EmailCase:
    name: str
    sender_name: str
    sender_addr: str
    subject: str
    body: str
    # Triage expectation
    expect_signal: str
    # Extraction expectation
    expect_commitment: bool
    expect_start: str | None = None   # "YYYY-MM-DD HH:MM" local, or "YYYY-MM-DD" all-day
    expect_title_contains: str | None = None
    note: str = ""


CASES: list[EmailCase] = [
    # ---------------------------------------------------------------- real appointments
    EmailCase(
        name="dentista_conferma",
        sender_name="Studio Dentistico Bianchi",
        sender_addr="appuntamenti@studiobianchi.it",
        subject="Conferma appuntamento",
        body=(
            "Gentile Sig. Passoni,\n\n"
            "le confermiamo l'appuntamento per l'igiene dentale di giovedì 24 settembre "
            "alle 15:00 presso il nostro studio in via Verdi 12.\n\n"
            "La preghiamo di avvisarci con 24 ore di anticipo in caso di disdetta.\n\n"
            "Cordiali saluti,\nStudio Dentistico Bianchi"
        ),
        expect_signal="appuntamento_data",
        expect_commitment=True,
        expect_start="2026-09-24 15:00",
        expect_title_contains="dental",
    ),
    EmailCase(
        name="udienza_tribunale",
        sender_name="Studio Legale Rossi",
        sender_addr="segreteria@studiolegalerossi.it",
        subject="Convocazione udienza",
        body=(
            "Buongiorno,\n\n"
            "la informiamo che l'udienza è fissata per il 3 ottobre alle ore 9:30 "
            "presso il Tribunale di Milano, aula 4.\n\n"
            "È necessaria la sua presenza. La invitiamo ad arrivare 20 minuti prima.\n\n"
            "Avv. Rossi"
        ),
        expect_signal="appuntamento_data",
        expect_commitment=True,
        expect_start="2026-10-03 09:30",
        expect_title_contains="udienza",
    ),
    EmailCase(
        name="riunione_relativa",
        sender_name="Marco Bianchi",
        sender_addr="marco.bianchi@azienda.it",
        subject="Allineamento sul progetto",
        body=(
            "Ciao Gabriele,\n\n"
            "ci vediamo domani alle 14:30 in sala riunioni per fare il punto sul progetto? "
            "Dovremmo avere bisogno di circa due ore.\n\n"
            "Fammi sapere se ti va bene.\nMarco"
        ),
        # A person wrote it AND it contains an appointment. Both signals are true;
        # appuntamento_data is the more useful of the two and lands on the same category.
        expect_signal="appuntamento_data",
        expect_commitment=True,
        expect_start="2026-09-22 14:30",
        note="Relative date: needs the email's receipt date, plus an explicit 2h duration.",
    ),
    EmailCase(
        name="visita_medica_weekday",
        sender_name="Poliambulatorio San Marco",
        sender_addr="prenotazioni@sanmarco.it",
        subject="Promemoria visita specialistica",
        body=(
            "Promemoria: la sua visita cardiologica è prevista per martedì prossimo "
            "alle 11:00. Si presenti con l'impegnativa e la tessera sanitaria."
        ),
        expect_signal="appuntamento_data",
        expect_commitment=True,
        expect_start="2026-09-29 11:00",
        note="'martedì prossimo' said on a Monday means the following week, not tomorrow.",
    ),
    EmailCase(
        name="volo_prenotazione",
        sender_name="Prenotazioni Voli",
        sender_addr="noreply@compagniaaerea.com",
        subject="La tua prenotazione è confermata - MXP to LHR",
        body=(
            "Prenotazione confermata.\n\n"
            "Volo AZ 204\nMilano Malpensa (MXP) → Londra Heathrow (LHR)\n"
            "Partenza: 12 ottobre 2026, 07:25\nArrivo: 08:55\n"
            "Codice prenotazione: XK9P2M\n\n"
            "Il check-in online apre 48 ore prima della partenza."
        ),
        # Also a receipt, but a flight is first of all something you have to be at.
        expect_signal="appuntamento_data",
        expect_commitment=True,
        expect_start="2026-10-12 07:25",
        note="A receipt that is also a commitment. Both are true; the commitment matters.",
    ),
    EmailCase(
        name="scadenza_bollo",
        sender_name="Agenzia delle Entrate",
        sender_addr="noreply@agenziaentrate.gov.it",
        subject="Scadenza pagamento bollo auto",
        body=(
            "Le ricordiamo che il pagamento del bollo auto per il veicolo targato AB123CD "
            "deve essere effettuato entro il 30 settembre 2026.\n\n"
            "Importo dovuto: 212,00 euro."
        ),
        expect_signal="scadenza_pagamento",
        expect_commitment=True,
        expect_start="2026-09-30",
        note="A deadline with no time: an all-day entry, not an appointment at midnight.",
    ),
    EmailCase(
        name="colloquio_orario_esteso",
        sender_name="HR Tecnologie SRL",
        sender_addr="hr@tecnologie.it",
        subject="Convocazione colloquio",
        body=(
            "Gentile candidato,\n\n"
            "la convochiamo per un colloquio venerdì dalle 10 alle 12 presso la nostra "
            "sede di Torino, corso Francia 45.\n\nCordialmente,\nUfficio HR"
        ),
        expect_signal="appuntamento_data",
        expect_commitment=True,
        expect_start="2026-09-25 10:00",
        note="Time range: must start at 10:00 and last 120 minutes.",
    ),

    # ---------------------------------------------------------------- NOT commitments
    EmailCase(
        name="promo_scadenza_commerciale",
        sender_name="MegaStore",
        sender_addr="offerte@megastore.it",
        subject="Ultimi giorni! Sconti fino al 70%",
        body=(
            "Approfitta subito: la promozione scade il 30 settembre!\n\n"
            "Sconti fino al 70% su tutto il catalogo. Spedizione gratuita sopra i 50 euro. "
            "Non perdere questa occasione, acquista entro venerdì."
        ),
        expect_signal="promozione",
        expect_commitment=False,
        note="A commercial deadline is the classic false positive. It has a date and it is "
             "not his appointment.",
    ),
    EmailCase(
        name="newsletter_con_date",
        sender_name="Rassegna Tech",
        sender_addr="news@rassegnatech.it",
        subject="Le notizie della settimana",
        body=(
            "Questa settimana: la conferenza WWDC si terrà il 9 giugno 2027, mentre il "
            "rilascio della nuova versione è previsto per ottobre.\n\n"
            "Fondata nel 1998, la nostra rassegna arriva ogni lunedì."
        ),
        expect_signal="newsletter_scelta",
        expect_commitment=False,
        note="Dates about the world, not about him. Also a 1998 in the footer.",
    ),
    EmailCase(
        name="portale_annuncio",
        sender_name="Idealista",
        sender_addr="noresponder@idealista.com",
        subject="Cambio de precio en tus favoritos",
        body=(
            "El precio de una vivienda que tienes en favoritos ha bajado de 250.000 a "
            "240.000 euros. Entra ahora para verla antes que otros."
        ),
        expect_signal="annuncio_portale",
        expect_commitment=False,
    ),
    EmailCase(
        name="social_notifica",
        sender_name="LinkedIn",
        sender_addr="notifications-noreply@linkedin.com",
        subject="Hai 3 nuove visualizzazioni del profilo",
        body="Il tuo profilo è stato visualizzato 3 volte questa settimana. Scopri da chi.",
        expect_signal="notifica_social",
        expect_commitment=False,
    ),
    EmailCase(
        name="sicurezza_accesso",
        sender_name="Google",
        sender_addr="no-reply@accounts.google.com",
        subject="Nuovo accesso al tuo account",
        body=(
            "È stato effettuato un nuovo accesso al tuo account Google da un dispositivo "
            "Windows a Milano. Se sei stato tu, puoi ignorare questo messaggio."
        ),
        expect_signal="sicurezza_account",
        expect_commitment=False,
        note="Important, but there is nothing to put on a calendar.",
    ),
    EmailCase(
        name="vago_senza_data",
        sender_name="Luca Verdi",
        sender_addr="luca.verdi@example.it",
        subject="Ci sentiamo",
        body=(
            "Ciao Gabriele, dobbiamo assolutamente trovare il tempo per vederci. "
            "Ti faccio sapere appena possibile, magari un giorno di questi. Un abbraccio!"
        ),
        expect_signal="persona_reale",
        expect_commitment=False,
        note="A real person, a real intention, no date. Must not invent one.",
    ),
    EmailCase(
        name="spedizione",
        sender_name="Corriere Espresso",
        sender_addr="tracking@corriere.it",
        subject="Il tuo pacco è in consegna",
        body=(
            "Il tuo pacco 1Z999AA è in transito e sarà consegnato nei prossimi giorni. "
            "Segui la spedizione online."
        ),
        expect_signal="spedizione",
        expect_commitment=False,
        note="'nei prossimi giorni' is not a date.",
    ),
]


def triage_cases() -> list[EmailCase]:
    return CASES


def extraction_cases() -> list[EmailCase]:
    return CASES


def by_name(name: str) -> EmailCase:
    for case in CASES:
        if case.name == name:
            return case
    raise KeyError(name)
