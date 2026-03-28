import os
from dotenv import load_dotenv

# Load environment variables before importing any service
load_dotenv()

from services.telegram_bot import avvia_bot  # noqa: E402 — triggers logging setup in google_auth

import logging
logger = logging.getLogger("DonnaMain")


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("ERRORE: TELEGRAM_BOT_TOKEN non trovato nel file .env!")
        return

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.error("ERRORE: OPENAI_API_KEY non trovato nel file .env!")
        return

    avvia_bot(token)


if __name__ == "__main__":
    main()
