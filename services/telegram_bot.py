import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from agent.orchestrator import elabora_richiesta, reset_memoria

# Usiamo il logger configurato in google_auth
logger = logging.getLogger("DonnaTelegram")

def split_message(text, limit=4090):
    """
    Semplice helper per dividere messaggi lunghi senza tagliare parole a metà.
    Telegram ha un limite di 4096 caratteri.
    """
    if not text:
        return ["(Nessuna risposta generata)"]
        
    if len(text) <= limit:
        return [text]
    
    parts = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        
        # Cerca l'ultimo spazio prima del limite (preferendo l'invio a capo)
        split_pos = text.rfind('\n', 0, limit)
        if split_pos == -1:
            split_pos = text.rfind(' ', 0, limit)
            if split_pos == -1:
                split_pos = limit
        
        parts.append(text[:split_pos].strip())
        text = text[split_pos:].strip()
    return parts

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gestisce il comando iniziale /start con il tocco di Donna"""
    logger.info(f"Comando /start ricevuto da {update.effective_user.first_name}")
    await update.message.reply_text(
        "Sono Donna. E so già perché sei qui. 😉\n\n"
        "Metterò in riga il tuo Calendario, le tue Email e i tuoi Task. "
        "Dopotutto, qualcuno deve pur gestire questo posto. Dimmi pure cosa ti serve.\n\n"
        "Se vuoi che dimentichi tutto (un nuovo inizio), usa /reset."
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gestisce il comando /reset per pulire la memoria della sessione"""
    chat_id = update.effective_chat.id
    logger.info(f"Comando /reset ricevuto da chat {chat_id}")
    reset_memoria(chat_id)
    await update.message.reply_text(
        "Fascicolo archiviato. 📂\n"
        "Ho dimenticato quello che ci siamo detti, ma non dimenticherò mai quanto sei fortunato ad avermi. "
        "Ricominciamo da capo, ma cerca di non annoiarmi. 😉"
    )

async def gestisci_messaggio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Questa funzione intercetta i messaggi e li passa a Donna con memoria del chat_id"""
    testo_utente = update.message.text
    chat_id = update.effective_chat.id
    logger.info(f"Messaggio ricevuto da {update.effective_user.username or update.effective_user.first_name} (ID: {chat_id}): {testo_utente}")
    
    # Mandiamo un messaggio di attesa coerente
    messaggio_attesa = await update.message.reply_text("⏳ Un attimo, ci penso io...")
    
    try:
        # Passiamo il testo e il chat_id per la memoria
        risposta_ia = await elabora_richiesta(testo_utente, chat_id)
        
        # Gestione messaggi lunghi per evitare errori di Telegram (4096 char)
        parti = split_message(risposta_ia)
        
        # Modifichiamo il messaggio di attesa con la prima parte della risposta
        await messaggio_attesa.edit_text(parti[0])
        
        # Se ci sono altre parti, le inviamo come messaggi aggiuntivi
        for parte in parti[1:]:
            if parte: # Evita l'invio di stringhe vuote
                await update.message.reply_text(parte)
            
    except Exception as e:
        logger.error(f"Errore critico durante l'elaborazione della richiesta: {e}", exc_info=True)
        await messaggio_attesa.edit_text("❌ Scusami, ho riscontrato un errore tecnico. Riprova tra un istante.")

def avvia_bot(token: str):
    """Inizializza e avvia il bot Telegram"""
    logger.info("Inizializzazione bot Telegram...")
    app = ApplicationBuilder().token(token).build()

    # Colleghiamo le funzioni create sopra agli eventi di Telegram
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, gestisci_messaggio))

    logger.info("Bot Telegram avviato e in ascolto.")
    app.run_polling()
