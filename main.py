"""Donna — entry point.

Everything runs locally: inference goes to Ollama on this machine, and the only network calls
are to Google for your own mail, calendar and tasks.

    python main.py          start the bot and the background pipeline
    python -m donna doctor  check the environment first
    python -m donna chat    talk to her from the terminal, no Telegram needed
"""
from __future__ import annotations

import sys

from dotenv import load_dotenv

load_dotenv()

from donna.app import run  # noqa: E402 — must follow load_dotenv


if __name__ == "__main__":
    sys.exit(run())
