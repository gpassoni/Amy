"""Central configuration. Everything tunable lives here or in .env — nothing is hardcoded
deeper in the stack.

Testo rivolto all'utente resta in italiano; gli identificatori sono in inglese.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Who she works for ---
    # Needed because Italian agrees adjectives with gender, and Donna is female while the user
    # may not be. Told only in the persona, the 9b kept answering "sei libera" to a man; it is
    # also stated in the state block, which sits closest to the question.
    user_name: str = "Gabriele"
    user_grammatical_gender: str = "m"   # "m" or "f"

    # --- Interfaces ---
    telegram_bot_token: str = ""
    # Where proactive notifications go. Captured automatically on /start when unset.
    telegram_chat_id: int | None = None
    web_host: str = "127.0.0.1"
    web_port: int = 8765

    # --- Google ---
    google_credentials_file: Path = PROJECT_ROOT / "credentials.json"
    google_token_file: Path = PROJECT_ROOT / "token.json"
    calendar_timezone: str = "Europe/Rome"

    # --- Storage ---
    db_path: Path = PROJECT_ROOT / "donna.db"

    # --- Ollama ---
    # No cloud providers anywhere by design: everything runs on this machine.
    ollama_host: str = "http://localhost:11434"
    ollama_timeout: float = 180.0

    # The 9b holds ~6.4 GB of VRAM. Measured on an RTX 3060 Ti (8 GB): KV cache is
    # nearly free (4k -> 8k costs ~84 MiB), so context size is not the lever to pull
    # if VRAM gets tight — model choice is. Keep-alive is deliberately finite so the
    # GPU is handed back when Donna is idle; "-1" pins it forever.
    chat_keep_alive: str = "30m"
    worker_keep_alive: str = "10m"

    # --- Sync ---
    sync_interval_minutes: int = 5
    gmail_sync_window_days: int = 7
    gmail_max_per_sync: int = 100

    # --- Pipeline thresholds ---
    triage_batch_size: int = 20
    # Below this, extraction is retried once on the big model.
    extract_escalate_below: float = 0.6
    # Below this, no proposal is created at all.
    proposal_confidence_floor: float = 0.4
    proposal_expiry_days: int = 7

    # --- Logging ---
    log_level: str = "INFO"
    log_file: Path = PROJECT_ROOT / "donna.log"
    log_max_bytes: int = 5 * 1024 * 1024
    log_backup_count: int = 3


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor. Import this, not a module-level instance, so tests can clear it."""
    return Settings()
