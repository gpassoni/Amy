-- Small key/value store for things Amy learns about her own installation, as opposed to
-- things she learns about the user.
--
-- The motivating case is the Telegram chat id. It cannot come from .env on first run (you do
-- not know it until you message the bot), and it must survive a restart, so it is neither
-- configuration nor conversation. One table beats a config file Amy has to rewrite.
CREATE TABLE app_state (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT NOT NULL
);
