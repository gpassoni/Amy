-- Activity: what Donna is doing, and what she just did.
--
-- `traces` already records every LLM call, but a trace is one model request — it cannot answer
-- "which agent is working right now" or "what happened during that turn". Activity is the
-- unit of work above it: one conversational turn, or one pipeline stage. Traces belong to an
-- activity via their parent_id, so the dashboard can drill from "the inbox agent answered
-- this" down to the exact prompt it sent.
--
-- Rows are written when work STARTS, with status 'running', and updated when it ends. That is
-- what makes a live view possible: anything still 'running' is happening now. It also means a
-- crash leaves evidence — a stuck 'running' row is a bug you can see.
CREATE TABLE activity (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,              -- turn | cycle | sync | triage | extract | notify
    actor       TEXT NOT NULL,              -- agent name, or job name
    status      TEXT NOT NULL DEFAULT 'running',  -- running | done | failed
    summary     TEXT,                       -- one line, human readable
    detail      TEXT,                       -- JSON: intent, tools, counts, whatever fits
    channel     TEXT,
    chat_id     TEXT,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    duration_ms INTEGER,
    error       TEXT
);

CREATE INDEX idx_activity_recent  ON activity(started_at DESC);
CREATE INDEX idx_activity_running ON activity(started_at) WHERE status = 'running';
CREATE INDEX idx_activity_actor   ON activity(actor, started_at DESC);
