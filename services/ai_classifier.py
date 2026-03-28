import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger("DonnaAIClassifier")

# Inizializziamo un modello leggero per la classificazione
llm_mini = ChatOpenAI(model="gpt-4o-mini", temperature=0)

CLASSIFICAZIONE_PROMPT = """Classifica l'email fornita in una delle seguenti categorie basandoti RIGOROSAMENTE su queste regole:

1. importante: Comunicazioni dirette da persone reali, scadenze imminenti, ricevute d'acquisto, avvisi di sicurezza, inviti a riunioni. Tutto ciò che richiede attenzione o azione.
2. inutile: Newsletter promozionali, notifiche social ("X ha postato..."), messaggi automatici di "grazie per esserti registrato", spam.
3. da_leggere: Newsletter informative di valore, aggiornamenti spedizione non urgenti, report settimanali.

Rispondi SOLO con una delle tre parole chiave: importante, inutile, da_leggere."""

def determina_categoria(mittente, oggetto, corpo):
    """
    Usa l'IA per classificare l'email.
    """
    try:
        testo_email = f"""DA: {mittente}
OGGETTO: {oggetto}
CORPO: {corpo[:500]}"""
        
        messaggi = [
            SystemMessage(content=CLASSIFICAZIONE_PROMPT),
            HumanMessage(content=testo_email)
        ]
        
        risposta = llm_mini.invoke(messaggi).content.strip().lower()
        
        if risposta in ["importante", "inutile", "da_leggere"]:
            return risposta
        
        # Fallback se l'IA risponde in modo imprevisto
        logger.warning(f"IA ha restituito una categoria non valida: {risposta}. Fallback su 'da_leggere'.")
        return "da_leggere"
    except Exception as e:
        logger.error(f"Errore nella classificazione IA: {e}")
        return "da_leggere"
