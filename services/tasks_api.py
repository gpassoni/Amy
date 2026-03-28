import logging
from services.google_auth import get_google_service

logger = logging.getLogger("DonnaTasks")

def lista_task_non_completate(max_risultati=10):
    """
    Recupera le task non completate dalla lista predefinita.
    Restituisce una lista di dizionari con i dettagli di ogni task.
    """
    try:
        service = get_google_service('tasks', 'v1')
        # Recuperiamo la lista di task principale (@default)
        logger.info("Recupero della lista di impegni (Google Tasks)...")
        risultati = service.tasks().list(
            tasklist='@default', 
            showCompleted=False, 
            maxResults=max_risultati
        ).execute()
        
        return risultati.get('items', [])
    except Exception as e:
        logger.error(f"Errore nel recupero delle task: {e}")
        return []

def aggiungi_task(titolo, note=None, data_scadenza=None):
    """
    Aggiunge una nuova task alla lista predefinita.
    :param titolo: Il titolo della task.
    :param note: Eventuali dettagli aggiuntivi.
    :param data_scadenza: Formato ISO 8601 (es. '2023-12-31T23:59:59Z').
    """
    try:
        service = get_google_service('tasks', 'v1')
        body = {'title': titolo}
        if note:
            body['notes'] = note
        if data_scadenza:
            # Google Tasks vuole la data con offset o Z
            body['due'] = data_scadenza
            
        task = service.tasks().insert(tasklist='@default', body=body).execute()
        logger.info(f"Task creata con successo: {task.get('title')} (ID: {task.get('id')})")
        return task
    except Exception as e:
        logger.error(f"Errore nell'aggiunta della task '{titolo}': {e}")
        return None

def completa_task(task_id):
    """
    Segna una task come completata.
    """
    try:
        service = get_google_service('tasks', 'v1')
        # Recuperiamo la task per assicurarci che esista
        task = service.tasks().get(tasklist='@default', task=task_id).execute()
        
        # Aggiorniamo lo stato
        task['status'] = 'completed'
        
        updated_task = service.tasks().update(
            tasklist='@default', 
            task=task_id, 
            body=task
        ).execute()
        
        logger.info(f"Task '{task.get('title')}' segnata come completata.")
        return updated_task
    except Exception as e:
        logger.error(f"Errore nel completamento della task {task_id}: {e}")
        return None

if __name__ == "__main__":
    # Test veloce
    print("--- TEST GOOGLE TASKS ---")
    tasks = lista_task_non_completate()
    print(f"Trovate {len(tasks)} task.")
    for t in tasks:
        print(f"- {t.get('title')}")
