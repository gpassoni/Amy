"""Prompts for the pipeline's model calls.

Kept in one module so they can be diffed, versioned and A/B'd against the eval harness
without touching pipeline logic.

Written in Italian because the mail being classified is Italian, English and Spanish, and
matching the assistant's own working language to the user's reduces the amount of
translation the model does implicitly. The classification rules are carried over from v1's
services/ai_classifier.py, which encoded real judgements worth keeping — sharpened with the
distinctions that a 2B model gets wrong without them.
"""
from __future__ import annotations

# ---------------------------------------------------------------- triage
# Asks "what is this email?", not "how important is it?". The importance policy lives in
# schemas.SIGNAL_CATEGORY. Kept short: the 2B followed a 450-word rulebook poorly, and
# every word of system prompt is prompt-eval time on each of 154 emails.
CLASSIFY_SYSTEM = """Riconosci CHE TIPO di email è. Scegli una sola etichetta.

persona_reale           una persona ha scritto personalmente
appuntamento_data       appuntamento, convocazione, prenotazione con data e ora
scadenza_pagamento      bolletta, scadenza, sollecito, rinnovo a pagamento, addebito
sicurezza_account       avviso di sicurezza autentico (accesso sospetto, password)
ricevuta_ordine         ricevuta, conferma d'ordine, fattura di un acquisto fatto
ente_istituzione        banca, medico, scuola, datore di lavoro, pubblica amministrazione
newsletter_scelta       newsletter o rivista a cui si è iscritti
aggiornamento_servizio  avviso su un servizio già attivo (modifiche, manutenzione)
spedizione              stato di una spedizione
promozione              sconti, offerte, marketing
notifica_social         "X ha commentato", "hai nuove visualizzazioni", inviti a collegarsi
annuncio_portale        avvisi automatici di annunci: immobili, e-commerce, giochi, aste
registrazione_benvenuto benvenuto, verifica indirizzo, grazie per l'iscrizione
spam_phishing           spam, o finge di essere qualcun altro per ottenere dati

ESEMPI DECISIVI:
- "Cambio de precio en tus favoritos" da un portale immobiliare -> annuncio_portale
- "Uno de tus favoritos ya no está publicado" -> annuncio_portale
- "X ti ha inviato una richiesta di collegamento" -> notifica_social
- "Il tuo abbonamento verrà rinnovato il 30/09 a 20€" -> scadenza_pagamento
- "Confermiamo l'appuntamento di giovedì alle 15" -> appuntamento_data
- "Nuovo accesso al tuo account da Chrome" -> sicurezza_account

Se un'email automatica contiene una data e un'ora che riguardano davvero il destinatario,
scegli appuntamento_data o scadenza_pagamento, non annuncio_portale.

Rispondi SOLO con il JSON: l'etichetta e quanto sei sicuro (0-1)."""


def classify_user(sender_name: str, sender_addr: str, subject: str, body: str) -> str:
    # 1200 characters is enough to classify: the decisive signal is almost always in the
    # sender, the subject and the opening lines. More text is slower and no more accurate.
    return (
        f"DA: {sender_name} <{sender_addr}>\n"
        f"OGGETTO: {subject}\n"
        f"CORPO:\n{(body or '').strip()[:1200]}"
    )


# ---------------------------------------------------------------- commitment extraction
EXTRACT_SYSTEM = """Estrai da un'email un eventuale impegno da mettere in calendario.

Un impegno esiste SOLO se l'email indica un momento preciso che riguarda Gabriele:
un appuntamento, una convocazione, una scadenza con data, un viaggio, un evento a cui
partecipa.

NON è un impegno:
- una promozione con scadenza commerciale ("offerta valida fino al 30", "acquista entro
  venerdì"): è una scadenza di chi vende, non un obbligo suo
- una data passata o una data citata come riferimento storico
- un orario di apertura, una data di fondazione, una data in una firma
- una data che riguarda il mondo e non lui ("la conferenza si terrà il 9 giugno")
- "ti faremo sapere", "a breve", "appena possibile", "nei prossimi giorni"
- la data di invio dell'email stessa

ATTENZIONE ALLA DIFFERENZA, è quella che si sbaglia più spesso:
- "il bollo auto deve essere pagato entro il 30 settembre"  -> È un impegno (kind=scadenza).
  Deve fare qualcosa lui, entro una data.
- "paga la fattura entro il 15", "presenta la domanda entro il 10", "rinnova entro fine
  mese" -> sono impegni.
- "offerta valida fino al 30 settembre" -> NON è un impegno. Nessuno gli chiede niente.

Se una scadenza obbliga LUI a fare qualcosa, è un impegno anche senza un orario preciso:
in quel caso all_day=true.

REGOLE SULLA DATA — leggile con attenzione:
- In "date_phrase" copia ESATTAMENTE le parole dell'email che indicano quando, senza
  riscriverle e senza interpretarle. Esempi: "martedì 24 settembre alle 15:00",
  "domani alle 9:30", "entro il 30 settembre", "dalle 14 alle 16 di venerdì".
- In "start_iso" metti la tua stima in formato YYYY-MM-DDTHH:MM. Se non sei sicuro,
  lascialo vuoto: la data viene ricalcolata da un programma a partire da "date_phrase",
  quindi copiare bene la frase è molto più importante che indovinare l'ISO.
- "all_day": true se non c'è un orario preciso.

In "title" scrivi un titolo breve da calendario, come lo scriverebbe una persona:
"Dentista", "Udienza tribunale", "Volo per Londra", "Scadenza IMU". Non copiare l'oggetto
dell'email.

In "evidence" copia la frase dell'email che dimostra l'impegno.

COMPILA I CAMPI IN QUESTO ORDINE, è importante:
1. "evidence": la frase dell'email che dimostra l'impegno, copiata alla lettera.
2. "date_phrase": solo le parole che dicono quando.
3. "start_iso": la tua stima, oppure vuoto.
4. "location", "kind", "title", "all_day".
5. "has_commitment": true solo se hai davvero trovato un impegno nei campi sopra.
6. "confidence": quanto sei sicuro, da 0 a 1.

Se non c'è nessun impegno: lascia evidence e date_phrase vuoti, has_commitment=false,
kind="nessuno". È il caso più comune, non inventare nulla.

Rispondi SOLO con il JSON richiesto."""


def extract_user(
    sender_name: str,
    sender_addr: str,
    subject: str,
    body: str,
    received_human: str,
) -> str:
    """Includes the email's own receipt date, so relative phrases have an anchor.

    Without it the model cannot tell what "domani" refers to and tends to invent an
    absolute date out of nothing.
    """
    return (
        f"Email ricevuta: {received_human}\n"
        f"DA: {sender_name} <{sender_addr}>\n"
        f"OGGETTO: {subject}\n"
        f"CORPO:\n{(body or '').strip()[:2500]}"
    )
