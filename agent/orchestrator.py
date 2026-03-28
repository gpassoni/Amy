import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from agent.tools import strumenti_disponibili

# Carichiamo le variabili d'ambiente
load_dotenv()

logger = logging.getLogger("DonnaOrchestrator")

# Inizializziamo il modello
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# La memoria per l'agente
memoria = MemorySaver()

# Gestione delle sessioni per il comando /reset
session_counters = {}

def reset_memoria(chat_id: int):
    """Incrementa il contatore di sessione per il chat_id."""
    session_counters[chat_id] = session_counters.get(chat_id, 0) + 1
    logger.info(f"Memoria resettata per chat {chat_id}. Nuova sessione: {session_counters[chat_id]}")

def ottieni_prompt_sistema():
    """Genera il prompt di sistema con data e ora aggiornate."""
    ora_attuale = datetime.now().strftime("%H:%M")
    data_oggi = datetime.now().strftime("%d/%m/%Y, %A")
    
    return f"""Sei Donna Paulsen. Non una semplice segretaria, ma *Donna*. 
Sei l'anima dello studio, brillante, sicura di te e sai sempre cosa serve prima ancora che venga chiesto. 
Oggi è {data_oggi} e sono le ore {ora_attuale}.

IL TUO PERSONAGGIO:
1. IDENTITÀ: Se qualcuno ti chiede chi sei, la risposta è semplice: "Sono Donna". 
2. TONO: Sicuro, spiritoso, leggermente sarcastico ma estremamente leale e professionale. 
3. ONNISCIENZA: Ti comporti come se avessi già previsto tutto. Non sei servile, sei una partner indispensabile.
4. STILE: Usa metafore legate al mondo di Suits (Harvey, chiudere accordi, "il fascicolo rosso"). Sii sintetica ma d'impatto.

REGOLE OPERATIVE:
1. BRIEFING PROATTIVO: Se ti viene chiesto un 'briefing', usa 'strumento_briefing_completo'. Presentalo come se stessi preparando il tuo capo per una giornata da senior partner. 
2. GESTIONE EMAIL e CALENDARIO: Usa gli strumenti con precisione chirurgica.
3. PROATTIVITÀ: Esegui le richieste in silenzio e riporta il risultato finale con classe.

Se gli strumenti non rispondono, gestisci la situazione con eleganza."""

# Creiamo l'Agente senza state_modifier per compatibilità
# Passiamo il prompt iniziale (verrà comunque rinforzato ad ogni chiamata)
agente = create_react_agent(
    llm, 
    tools=strumenti_disponibili, 
    checkpointer=memoria,
    prompt=ottieni_prompt_sistema()
)

async def elabora_richiesta(testo_utente: str, chat_id: int) -> str:
    """Passa il messaggio all'agente mantenendo solo gli ultimi 10 messaggi di contesto."""
    try:
        session_id = session_counters.get(chat_id, 0)
        config = {"configurable": {"thread_id": f"{chat_id}_{session_id}"}}
        
        # Recuperiamo lo stato attuale per applicare manualmente la finestra scorrevole
        stato_attuale = await agente.aget_state(config)
        messaggi_precedenti = stato_attuale.values.get("messages", []) if stato_attuale.values else []
        
        # Finestra scorrevole: teniamo solo gli ultimi 10 messaggi
        # (Escludiamo il system prompt iniziale se presente per reinserirlo aggiornato)
        cronologia_filtrata = [m for m in messaggi_precedenti if not isinstance(m, SystemMessage)]
        if len(cronologia_filtrata) > 10:
            cronologia_filtrata = cronologia_filtrata[-10:]

        # Prepariamo l'input: System Prompt aggiornato + cronologia filtrata + nuovo messaggio
        input_agente = {
            "messages": [SystemMessage(content=ottieni_prompt_sistema())] + cronologia_filtrata + [HumanMessage(content=testo_utente)]
        }
        
        logger.info(f"Richiesta chat {chat_id} (Sessione {session_id}): '{testo_utente}'")
        
        # Eseguiamo l'agente
        risultato = await agente.ainvoke(input_agente, config=config)
        
        return risultato["messages"][-1].content
    except Exception as e:
        logger.error(f"Errore nell'orchestrator per chat {chat_id}: {e}", exc_info=True)
        return "Scusa, ho avuto un problema tecnico. Riprova tra un istante."
