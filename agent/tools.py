import logging
from langchain.tools import tool
from datetime import datetime, timedelta
from services.gmail_api import leggi_email_non_lette, applica_label_a_messaggio, assicurati_labels_esistenti, LABEL_DONNA
from services.ai_classifier import determina_categoria
from services.calendar_api import elenco_prossimi_eventi, crea_evento_calendario, aggiorna_evento_calendario, elimina_evento_calendario
from services.tasks_api import lista_task_non_completate, aggiungi_task, completa_task

logger = logging.getLogger("DonnaTools")


@tool
def strumento_leggi_email(max_risultati: int = 5, giorni: int = 3) -> str:
    """
    Usa questo strumento quando l'utente ti chiede di leggere, controllare o
    riassumere le sue nuove email o le email non lette.
    Puoi specificare quanti giorni indietro guardare (default 3).
    """
    try:
        lista_email = leggi_email_non_lette(max_risultati, giorni)

        if not lista_email:
            return "Non sono state trovate nuove email non lette nelle ultime 24 ore."

        # Otteniamo gli ID delle label Donna una volta sola
        ids_donna = assicurati_labels_esistenti()
        ids_set = set(ids_donna.values())

        output_formattato = []
        for email in lista_email:
            # Controlliamo se l'email ha già una delle label di Donna
            label_gia_presente = any(lid in ids_set for lid in email.get('labelIds', []))

            prefisso_label = ""
            if not label_gia_presente:
                # Classificazione AI intelligente
                categoria = determina_categoria(email['mittente'], email['oggetto'], email['corpo'])
                # Applichiamo la label su Gmail
                applica_label_a_messaggio(email['id'], categoria)
                prefisso_label = f"[{LABEL_DONNA.get(categoria, categoria).upper()}] "
            else:
                # Identifichiamo quale label ha già
                for chiave, lid in ids_donna.items():
                    if lid in email.get('labelIds', []):
                        prefisso_label = f"[{LABEL_DONNA[chiave].upper()}] "
                        break

            testo_email = (
                f"📧 {prefisso_label} Da: {email['mittente']}\n"
                f"📌 Oggetto: {email['oggetto']}\n"
                f"📅 Data: {email['data']}\n"
                f"📄 Testo: {email['corpo']}\n"
            )
            output_formattato.append(testo_email)

        return "\n---\n".join(output_formattato)

    except Exception as e:
        logger.error(f"Errore in strumento_leggi_email: {e}", exc_info=True)
        return "Non sono riuscita a leggere le email. Riprova tra un attimo."


@tool
def strumento_leggi_calendario(giorni: int = 7) -> str:
    """
    Usa questo strumento quando l'utente ti chiede quali sono i suoi prossimi impegni.
    Puoi specificare il numero di giorni da controllare (default 7).
    """
    try:
        eventi = elenco_prossimi_eventi(giorni=giorni)
        if not eventi:
            return f"Non hai eventi in programma per i prossimi {giorni} giorni."

        output = []
        for e in eventi:
            id_evento = e.get('id')
            inizio = e['start'].get('dateTime', e['start'].get('date'))
            titolo = e.get('summary', 'Senza Titolo')
            output.append(f"📅 [{id_evento}] {inizio} - {titolo}")

        return "\n".join(output)
    except Exception as e:
        logger.error(f"Errore in strumento_leggi_calendario: {e}", exc_info=True)
        return "Non sono riuscita a leggere il calendario. Riprova tra un attimo."


@tool
def strumento_crea_evento(titolo: str, inizio_iso: str, fine_iso: str, descrizione: str = None) -> str:
    """Usa questo strumento per creare un nuovo evento sul calendario."""
    try:
        link = crea_evento_calendario(titolo, inizio_iso, fine_iso, descrizione)
        return f"Evento '{titolo}' creato con successo! Puoi vederlo qui: {link}"
    except Exception as e:
        logger.error(f"Errore in strumento_crea_evento: {e}", exc_info=True)
        return "Non sono riuscita a creare l'evento. Riprova tra un attimo."


@tool
def strumento_aggiorna_evento(id_evento: str, titolo: str = None, inizio_iso: str = None, fine_iso: str = None, descrizione: str = None) -> str:
    """Usa questo strumento per aggiornare un evento esistente."""
    try:
        link = aggiorna_evento_calendario(id_evento, titolo, inizio_iso, fine_iso, descrizione)
        return f"Evento aggiornato con successo! Puoi vederlo qui: {link}"
    except Exception as e:
        logger.error(f"Errore in strumento_aggiorna_evento: {e}", exc_info=True)
        return "Non sono riuscita ad aggiornare l'evento. Riprova tra un attimo."


@tool
def strumento_elimina_evento(id_evento: str) -> str:
    """Usa questo strumento per eliminare un evento esistente."""
    try:
        elimina_evento_calendario(id_evento)
        return "Evento eliminato con successo!"
    except Exception as e:
        logger.error(f"Errore in strumento_elimina_evento: {e}", exc_info=True)
        return "Non sono riuscita a eliminare l'evento. Riprova tra un attimo."


@tool
def strumento_leggi_tasks(max_risultati: int = 10) -> str:
    """Usa questo strumento per leggere la lista delle cose da fare (tasks)."""
    try:
        tasks = lista_task_non_completate(max_risultati)
        if not tasks:
            return "Non hai impegni o task in sospeso al momento."

        output = []
        for t in tasks:
            output.append(f"📝 [{t['id']}] {t['title']}")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Errore in strumento_leggi_tasks: {e}", exc_info=True)
        return "Non sono riuscita a leggere le task. Riprova tra un attimo."


@tool
def strumento_aggiungi_task(titolo: str, note: str = None, scadenza_iso: str = None) -> str:
    """Usa questo strumento per aggiungere una nuova task."""
    try:
        task = aggiungi_task(titolo, note, scadenza_iso)
        if task:
            return f"Task '{titolo}' aggiunta con successo!"
        return "Non sono riuscita ad aggiungere la task."
    except Exception as e:
        logger.error(f"Errore in strumento_aggiungi_task: {e}", exc_info=True)
        return "Non sono riuscita ad aggiungere la task. Riprova tra un attimo."


@tool
def strumento_completa_task(identificatore: str) -> str:
    """
    Usa questo strumento per segnare una task come completata.
    Passa l'ID della task se lo conosci, oppure il TITOLO esatto o una PAROLA CHIAVE (es. 'lampadine').
    Il sistema cercherà tra le task non completate quella che contiene il testo fornito.
    """
    try:
        tasks = lista_task_non_completate(max_risultati=50)

        # Cerchiamo per ID esatto
        task_trovata = next((t for t in tasks if t['id'] == identificatore), None)

        # Se non trovata per ID, cerchiamo per titolo (case insensitive e partial match)
        if not task_trovata:
            per_titolo = [t for t in tasks if identificatore.lower() in t['title'].lower()]
            if per_titolo:
                task_trovata = per_titolo[0]

        if task_trovata:
            completa_task(task_trovata['id'])
            return f"Task '{task_trovata['title']}' segnata come completata con successo!"
        else:
            return f"Non ho trovato nessuna task aperta che corrisponde a '{identificatore}'. Se pensi sia un errore, prova a elencare le task con 'strumento_leggi_tasks'."

    except Exception as e:
        logger.error(f"Errore in strumento_completa_task: {e}", exc_info=True)
        return "Non sono riuscita a completare la task. Riprova tra un attimo."


@tool
def strumento_briefing_completo() -> str:
    """
    Usa questo strumento quando l'utente chiede un 'briefing', un riassunto della giornata
    o cosa deve fare oggi/domani.
    Raccoglie automaticamente Email, Calendario e Tasks in un unico blocco di dati.
    Capisce da solo se l'utente si riferisce a oggi o a domani in base all'ora attuale.
    """
    ora_attuale = datetime.now().hour
    # Se è dopo le 18:00, prepariamo il briefing per domani
    periodo = "oggi" if ora_attuale < 18 else "domani"
    giorni_calendar = 1 if periodo == "oggi" else 2

    try:
        # Chiamiamo le funzioni api direttamente per evitare ricorsione tra tool
        email = leggi_email_non_lette(max_risultati=5)
        eventi = elenco_prossimi_eventi(giorni=giorni_calendar)
        tasks = lista_task_non_completate(max_risultati=10)

        # Gestione label per briefing
        ids_donna = assicurati_labels_esistenti()
        ids_set = set(ids_donna.values())

        # Formattazione sintetica per l'agente
        res = f"--- DATI BRIEFING PER {periodo.upper()} ---\n"
        res += f"EMAIL NON LETTE (Ultime 24h): {len(email)}\n"

        for e in email[:3]:
            # Classificazione automatica se manca
            if not any(lid in ids_set for lid in e.get('labelIds', [])):
                cat = determina_categoria(e['mittente'], e['oggetto'], e['corpo'])
                applica_label_a_messaggio(e['id'], cat)
                label_text = LABEL_DONNA[cat]
            else:
                label_text = next((LABEL_DONNA[k] for k, v in ids_donna.items() if v in e.get('labelIds', [])), "Letta")

            res += f"- {label_text} | {e['mittente']}: {e['oggetto']}\n"

        res += f"\nEVENTI CALENDARIO ({periodo}): {len(eventi)}\n"
        for ev in eventi:
            inizio = ev['start'].get('dateTime', ev['start'].get('date'))
            res += f"- {inizio}: {ev.get('summary')}\n"

        res += f"\nCOSE DA FARE (TASKS): {len(tasks)}\n"
        for t in tasks[:5]:
            res += f"- {t.get('title')}\n"

        return res
    except Exception as e:
        logger.error(f"Errore in strumento_briefing_completo: {e}", exc_info=True)
        return "Non sono riuscita a raccogliere i dati per il briefing. Riprova tra un attimo."


# Lista degli strumenti aggiornata
strumenti_disponibili = [
    strumento_leggi_email,
    strumento_leggi_calendario,
    strumento_crea_evento,
    strumento_aggiorna_evento,
    strumento_elimina_evento,
    strumento_leggi_tasks,
    strumento_aggiungi_task,
    strumento_completa_task,
    strumento_briefing_completo
]
