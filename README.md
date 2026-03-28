# Donna

A personal AI assistant delivered via Telegram. Donna manages your inbox, calendar, and task list through natural conversation — built with a LangGraph ReAct agent and OpenAI GPT-4o-mini.

## Features

- **Email** — reads unread Gmail, auto-classifies each message (Important / To Read / Junk) using AI, and applies the corresponding label directly in Gmail
- **Calendar** — lists upcoming events, creates, updates, and deletes Google Calendar entries
- **Tasks** — lists, adds, and completes Google Tasks items
- **Daily briefing** — aggregates emails, events, and tasks into a single morning (or evening) summary
- **Conversation memory** — maintains a sliding window of the last 10 messages per chat, reset on demand with `/reset`

## Tech Stack

| Layer | Technology |
|---|---|
| Agent | LangGraph ReAct + `MemorySaver` checkpointer |
| LLM | OpenAI GPT-4o-mini (agent + email classifier) |
| Telegram | python-telegram-bot v22 |
| Google APIs | Gmail v1, Calendar v3, Tasks v1 |
| Auth | OAuth 2.0 via google-auth-oauthlib |

## Architecture

```
main.py                      # Entry point — validates env, starts the bot
├── services/telegram_bot.py # Telegram handlers (/start, /reset, messages)
├── agent/
│   ├── orchestrator.py      # LangGraph ReAct agent with sliding memory window
│   └── tools.py             # LangChain tools wrapping the service layer
└── services/
    ├── google_auth.py       # Thread-safe Singleton for Google API clients
    ├── gmail_api.py         # Gmail read + label operations
    ├── calendar_api.py      # Calendar CRUD
    ├── tasks_api.py         # Tasks CRUD
    └── ai_classifier.py     # LLM-based email classification
```

The Google service layer uses a **Singleton + thread-local storage** pattern so a single OAuth session is shared safely across all async handlers without re-authenticating on every request.

## Prerequisites

- Python 3.12+
- A [Google Cloud project](https://console.cloud.google.com/) with the Gmail, Calendar, and Tasks APIs enabled
- An OAuth 2.0 **Desktop** client — download the credentials file as `credentials.json`
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Setup

```bash
# 1. Clone
git clone https://github.com/your-username/donna.git
cd donna

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and fill in TELEGRAM_BOT_TOKEN and OPENAI_API_KEY

# 5. Place credentials.json in the project root
#    (downloaded from Google Cloud Console)

# 6. Run
python main.py
```

On the first run a browser window opens for Google OAuth consent. After authorization, `token.json` is created automatically and reused on subsequent runs.

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Yes | — | Telegram bot token from BotFather |
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `CALENDAR_TIMEZONE` | No | `Europe/Rome` | IANA timezone for calendar events |
| `GOOGLE_CREDENTIALS_FILE` | No | `credentials.json` | Path to the Google OAuth credentials file |
| `GOOGLE_TOKEN_FILE` | No | `token.json` | Path where the OAuth token is cached |

## Notes

Function and variable names are in Italian — this project was built for personal use and the naming is internally consistent throughout. The bot's persona is inspired by Donna Paulsen from *Suits*.
