import os
import logging
from datetime import datetime, timedelta, timezone
from services.google_auth import get_google_service

# Configurazione Logger locale
logger = logging.getLogger("DonnaCalendar")

CALENDAR_TIMEZONE = os.getenv("CALENDAR_TIMEZONE", "Europe/Rome")

def elenco_prossimi_eventi(giorni=7):
    """Recupera gli eventi del calendario per il periodo indicato."""
    try:
        service = get_google_service('calendar', 'v3')
        
        # Calcoliamo l'intervallo di tempo (da ora a X giorni)
        now = datetime.now(timezone.utc)
        ora_inizio = now.isoformat()
        ora_fine = (now + timedelta(days=giorni)).isoformat()

        logger.info(f"Recupero eventi del calendario da {ora_inizio} a {ora_fine}...")

        eventi_result = service.events().list(
            calendarId='primary', 
            timeMin=ora_inizio,
            timeMax=ora_fine,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        return eventi_result.get('items', [])
    except Exception as e:
        logger.error(f"Errore nel recupero degli eventi del calendario: {e}", exc_info=True)
        return []

def crea_evento_calendario(sommario, inizio_iso, fine_iso, descrizione=None):
    """Crea un nuovo evento nel calendario."""
    try:
        service = get_google_service('calendar', 'v3')
        
        evento = {
            'summary': sommario,
            'description': descrizione,
            'start': {
                'dateTime': inizio_iso, # Formato: '2023-05-28T09:00:00Z'
                'timeZone': CALENDAR_TIMEZONE,
            },
            'end': {
                'dateTime': fine_iso,
                'timeZone': CALENDAR_TIMEZONE,
            },
        }

        evento_creato = service.events().insert(calendarId='primary', body=evento).execute()
        logger.info(f"Evento creato correttamente: {evento_creato.get('htmlLink')}")
        return evento_creato.get('htmlLink')
    except Exception as e:
        logger.error(f"Errore nella creazione dell'evento: {e}")
        return None

def aggiorna_evento_calendario(event_id, sommario=None, inizio_iso=None, fine_iso=None, descrizione=None):
    """Aggiorna i dettagli di un evento esistente."""
    try:
        service = get_google_service('calendar', 'v3')
        
        # Recuperiamo l'evento esistente
        evento = service.events().get(calendarId='primary', eventId=event_id).execute()

        if sommario:
            evento['summary'] = sommario
        if descrizione:
            evento['description'] = descrizione
        if inizio_iso:
            evento['start'] = {
                'dateTime': inizio_iso,
                'timeZone': CALENDAR_TIMEZONE,
            }
        if fine_iso:
            evento['end'] = {
                'dateTime': fine_iso,
                'timeZone': CALENDAR_TIMEZONE,
            }

        evento_aggiornato = service.events().update(calendarId='primary', eventId=event_id, body=evento).execute()
        logger.info(f"Evento aggiornato: {evento_aggiornato.get('htmlLink')}")
        return evento_aggiornato.get('htmlLink')
    except Exception as e:
        logger.error(f"Errore nell'aggiornamento dell'evento {event_id}: {e}")
        return None

def elimina_evento_calendario(event_id):
    """Rimuove un evento dal calendario."""
    try:
        service = get_google_service('calendar', 'v3')
        service.events().delete(calendarId='primary', eventId=event_id).execute()
        logger.info(f"Evento {event_id} eliminato dal calendario.")
        return True
    except Exception as e:
        logger.error(f"Errore nell'eliminazione dell'evento {event_id}: {e}")
        return False

if __name__ == "__main__":
    # Test veloce
    print("--- TEST CALENDARIO ---")
    eventi = elenco_prossimi_eventi(giorni=7)
    print(f"Numero di eventi trovati: {len(eventi)}")
    for e in eventi:
        print(f"- {e.get('summary')}")
