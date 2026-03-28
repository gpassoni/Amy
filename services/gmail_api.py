import base64
import logging
from bs4 import BeautifulSoup
from services.google_auth import get_google_service

# Configurazione Logger locale
logger = logging.getLogger("DonnaGmail")

# Costanti per le Label di Donna
LABEL_DONNA = {
    "importante": "🔴 Importanti",
    "inutile": "⚪ Inutili",
    "da_leggere": "🟡 Da Leggere"
}

def assicurati_labels_esistenti(servizio=None):
    """
    Verifica l'esistenza delle label di Donna e le crea se mancano.
    Ritorna un dizionario {nome_label: id_label}.
    Se servizio non è fornito, lo ottiene.
    """
    try:
        if servizio is None:
            servizio = get_google_service('gmail', 'v1')
            
        risultati = servizio.users().labels().list(userId='me').execute()
        labels_esistenti = risultati.get('labels', [])
        
        mappa_labels = {l['name']: l['id'] for l in labels_esistenti}
        ids_donna = {}

        for chiave, nome in LABEL_DONNA.items():
            if nome in mappa_labels:
                ids_donna[chiave] = mappa_labels[nome]
            else:
                logger.info(f"Creazione label mancante: {nome}")
                label_object = {
                    "name": nome,
                    "labelListVisibility": "labelShow",
                    "messageListVisibility": "show"
                }
                nuova_label = servizio.users().labels().create(userId='me', body=label_object).execute()
                ids_donna[chiave] = nuova_label['id']
        
        return ids_donna
    except Exception as e:
        logger.error(f"Errore nella gestione delle label: {e}")
        return {}

def applica_label_a_messaggio(msg_id, label_chiave):
    """
    Applica una label di Donna a un messaggio specifico.
    """
    try:
        servizio = get_google_service('gmail', 'v1')
        mappa_ids = assicurati_labels_esistenti(servizio)
        
        label_id = mappa_ids.get(label_chiave)
        if not label_id:
            logger.error(f"Label ID non trovato per chiave: {label_chiave}")
            return False

        servizio.users().messages().batchModify(
            userId='me',
            body={
                'ids': [msg_id],
                'addLabelIds': [label_id]
            }
        ).execute()
        
        logger.info(f"Label {label_chiave} applicata al messaggio {msg_id}")
        return True
    except Exception as e:
        logger.error(f"Errore nell'applicazione della label: {e}")
        return False

def estrai_corpo_ricorsivo(payload):
    """
    Estrae ricorsivamente il corpo del messaggio Gmail.
    Cerca prioritariamente il testo semplice (text/plain), altrimenti pulisce l'HTML.
    """
    parts = payload.get('parts', [])
    
    # Se non ci sono parti, il contenuto è nel body principale (email semplici)
    if not parts:
        data = payload.get('body', {}).get('data', '')
        if not data:
            return ""
        
        testo_decodificato = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        
        if payload.get('mimeType') == 'text/html':
            return BeautifulSoup(testo_decodificato, "html.parser").get_text(separator="\n", strip=True)
        return testo_decodificato

    # Strategia: 1. Cerca text/plain in tutte le parti attuali
    for part in parts:
        if part.get('mimeType') == 'text/plain' and 'data' in part.get('body', {}):
            return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')

    # 2. Se non trovato, cerca text/html nelle parti attuali
    for part in parts:
        if part.get('mimeType') == 'text/html' and 'data' in part.get('body', {}):
            html = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
            return BeautifulSoup(html, "html.parser").get_text(separator="\n", strip=True)

    # 3. Se ancora non trovato, scendi ricorsivamente (per multipart/mixed o multipart/alternative nidificati)
    for part in parts:
        if 'parts' in part:
            corpo = estrai_corpo_ricorsivo(part)
            if corpo:
                return corpo

    return ""


def leggi_email_non_lette(max_risultati=10, giorni=3):
    """
    Cerca le email non lette degli ultimi X giorni.
    Restituisce una lista di dizionari con i dettagli di ogni email.
    """
    try:
        # Otteniamo il servizio tramite il manager Singleton
        servizio = get_google_service('gmail', 'v1')

        logger.info(f"Ricerca nuove email non lette degli ultimi {giorni} giorni...")
        
        # Query: is:unread (non lette) e newer_than:Xd
        query = f"is:unread newer_than:{giorni}d"
        risultati = servizio.users().messages().list(
            userId='me', q=query, maxResults=max_risultati
        ).execute()
        
        messaggi = risultati.get('messages', [])

        if not messaggi:
            logger.info("Nessuna email non letta trovata.")
            return []

        email_elaborate = []
        
        for msg in messaggi:
            try:
                msg_dettaglio = servizio.users().messages().get(userId='me', id=msg['id']).execute()
                payload = msg_dettaglio.get('payload', {})
                headers = payload.get('headers', [])

                # Estrazione metadati dagli headers
                oggetto = next((h['value'] for h in headers if h['name'].lower() == 'subject'), "Nessun Oggetto")
                mittente = next((h['value'] for h in headers if h['name'].lower() == 'from'), "Sconosciuto")
                data_invio = next((h['value'] for h in headers if h['name'].lower() == 'date'), "Data non disponibile")
                label_ids = msg_dettaglio.get('labelIds', [])

                # Estrazione corpo del testo
                corpo = estrai_corpo_ricorsivo(payload)
                
                # Taglio a 1000 caratteri per risparmiare token e restare nei limiti di Telegram
                corpo_troncato = corpo[:1000] + "..." if len(corpo) > 1000 else corpo
                
                email_elaborate.append({
                    "id": msg['id'],
                    "mittente": mittente,
                    "oggetto": oggetto,
                    "data": data_invio,
                    "corpo": corpo_troncato,
                    "labelIds": label_ids
                })
            except Exception as e:
                logger.error(f"Errore nel processare l'email {msg['id']}: {e}", exc_info=True)
                continue

        logger.info(f"Processate {len(email_elaborate)} email con successo.")
        return email_elaborate

    except Exception as e:
        logger.error(f"Errore generale nel servizio Gmail: {e}", exc_info=True)
        return []


if __name__ == "__main__":
    # Test locale
    email = leggi_email_non_lette()
    if not email:
        print("Nessuna nuova email trovata.")
    for e in email:
        print(f"--- EMAIL ---\nDa: {e['mittente']}\nOggetto: {e['oggetto']}\nData: {e['data']}\nTesto: {e['corpo']}\n")
