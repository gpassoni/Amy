import os
import logging
import threading
from logging.handlers import RotatingFileHandler
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Configurazione Logging Professionale
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("donna.log", maxBytes=5 * 1024 * 1024, backupCount=3),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DonnaService")

# Questi sono i permessi che chiediamo a Google.
# Aggiunto 'tasks' per la gestione delle to-do list.
SCOPES = [
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/tasks'
]

# Percorsi dei file OAuth — configurabili via variabili d'ambiente
GOOGLE_CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")
GOOGLE_TOKEN_FILE = os.getenv("GOOGLE_TOKEN_FILE", "token.json")

class GoogleServiceManager:
    """
    Gestore Singleton per i servizi Google. 
    Usa Thread-local storage per evitare conflitti tra thread.
    """
    _instance = None
    _lock = threading.Lock()
    _creds = None
    
    # Storage locale al thread per i servizi
    _thread_local = threading.local()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GoogleServiceManager, cls).__new__(cls)
        return cls._instance

    def _get_creds(self):
        """Gestisce l'autenticazione e il refresh dei token in modo thread-safe."""
        with self._lock:
            if self._creds and self._creds.valid:
                return self._creds

            creds = None
            if os.path.exists(GOOGLE_TOKEN_FILE):
                creds = Credentials.from_authorized_user_file(GOOGLE_TOKEN_FILE, SCOPES)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        logger.info("Refresh del token Google in corso...")
                        creds.refresh(Request())
                    except Exception as e:
                        logger.error(f"Errore durante il refresh del token: {e}")
                        creds = None

                if not creds:
                    logger.info("Inizializzazione nuovo flusso di autenticazione...")
                    flow = InstalledAppFlow.from_client_secrets_file(GOOGLE_CREDENTIALS_FILE, SCOPES)
                    creds = flow.run_local_server(port=0)

                with open(GOOGLE_TOKEN_FILE, 'w') as token:
                    token.write(creds.to_json())
            
            self._creds = creds
            return creds

    def get_service(self, service_name, version):
        """Restituisce un'istanza del servizio richiesta, specifica per il thread corrente."""
        if not hasattr(self._thread_local, 'services'):
            self._thread_local.services = {}
            
        service_key = f"{service_name}_{version}"
        
        if service_key not in self._thread_local.services:
            try:
                creds = self._get_creds()
                # Disabilitiamo la discovery cache per evitare problemi di lock su Windows
                # e usiamo un timeout per evitare che rimanga appeso.
                self._thread_local.services[service_key] = build(
                    service_name, 
                    version, 
                    credentials=creds,
                    static_discovery=True
                )
                logger.info(f"Servizio Google '{service_name}' ({version}) inizializzato per il thread {threading.get_ident()}.")
            except Exception as e:
                logger.error(f"Impossibile inizializzare il servizio '{service_name}': {e}")
                raise e
        return self._thread_local.services[service_key]

def get_google_service(name, version):
    """Metodo consigliato per ottenere un servizio Google ottimizzato e sicuro."""
    return GoogleServiceManager().get_service(name, version)
