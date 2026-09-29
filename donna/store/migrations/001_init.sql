-- Donna v2 initial schema.
-- All *_ts / *_at columns hold UTC ISO-8601 strings ("2026-09-20T14:30:00+00:00").
-- The format is uniform, so string comparison is chronological comparison.

-- ---------------------------------------------------------------- emails
CREATE TABLE emails (
    id                  TEXT PRIMARY KEY,          -- Gmail message id
    thread_id           TEXT,
    from_addr           TEXT,
    from_name           TEXT,
    to_addrs            TEXT,
    subject             TEXT,
    snippet             TEXT,
    body                TEXT,
    received_at         TEXT NOT NULL,
    gmail_labels        TEXT,                      -- JSON array of label ids
    is_unread           INTEGER NOT NULL DEFAULT 1,

    -- Triage output
    category            TEXT,                      -- importante | da_leggere | inutile
    category_confidence REAL,
    category_reason     TEXT,
    classified_at       TEXT,
    classifier_model    TEXT,
    classify_trace_id   TEXT,
    label_applied       INTEGER NOT NULL DEFAULT 0,

    -- Commitment extraction
    extracted_at        TEXT,
    extract_trace_id    TEXT,

    synced_at           TEXT NOT NULL
);

CREATE INDEX idx_emails_received      ON emails(received_at DESC);
CREATE INDEX idx_emails_category      ON emails(category, received_at DESC);
CREATE INDEX idx_emails_thread        ON emails(thread_id);
-- Partial indexes for the two pipeline queues: both stay tiny.
CREATE INDEX idx_emails_untriaged     ON emails(received_at) WHERE category IS NULL;
CREATE INDEX idx_emails_unextracted   ON emails(received_at)
    WHERE category = 'importante' AND extracted_at IS NULL;

CREATE VIRTUAL TABLE emails_fts USING fts5(
    subject, body, content='emails', content_rowid='rowid', tokenize='unicode61'
);
CREATE TRIGGER emails_fts_ins AFTER INSERT ON emails BEGIN
    INSERT INTO emails_fts(rowid, subject, body) VALUES (new.rowid, new.subject, new.body);
END;
CREATE TRIGGER emails_fts_del AFTER DELETE ON emails BEGIN
    INSERT INTO emails_fts(emails_fts, rowid, subject, body)
        VALUES ('delete', old.rowid, old.subject, old.body);
END;
-- Scoped to the text columns so triage updates do not churn the index.
CREATE TRIGGER emails_fts_upd AFTER UPDATE OF subject, body ON emails BEGIN
    INSERT INTO emails_fts(emails_fts, rowid, subject, body)
        VALUES ('delete', old.rowid, old.subject, old.body);
    INSERT INTO emails_fts(rowid, subject, body) VALUES (new.rowid, new.subject, new.body);
END;

-- ---------------------------------------------------------------- events
CREATE TABLE events (
    id                  TEXT PRIMARY KEY,
    calendar_id         TEXT NOT NULL DEFAULT 'primary',
    summary             TEXT,
    description         TEXT,
    location            TEXT,
    start_ts            TEXT,                      -- normalised UTC
    end_ts              TEXT,
    start_raw           TEXT,                      -- verbatim from Google, fidelity kept
    end_raw             TEXT,
    all_day             INTEGER NOT NULL DEFAULT 0,
    status              TEXT,                      -- confirmed | tentative | cancelled
    organizer           TEXT,
    attendees           TEXT,                      -- JSON array
    html_link           TEXT,
    recurring_event_id  TEXT,
    updated_at          TEXT,
    source              TEXT NOT NULL DEFAULT 'google',   -- google | donna
    origin_proposal_id  INTEGER,
    synced_at           TEXT NOT NULL
);

CREATE INDEX idx_events_start  ON events(start_ts);
CREATE INDEX idx_events_active ON events(start_ts) WHERE status != 'cancelled';

CREATE VIRTUAL TABLE events_fts USING fts5(
    summary, description, content='events', content_rowid='rowid', tokenize='unicode61'
);
CREATE TRIGGER events_fts_ins AFTER INSERT ON events BEGIN
    INSERT INTO events_fts(rowid, summary, description) VALUES (new.rowid, new.summary, new.description);
END;
CREATE TRIGGER events_fts_del AFTER DELETE ON events BEGIN
    INSERT INTO events_fts(events_fts, rowid, summary, description)
        VALUES ('delete', old.rowid, old.summary, old.description);
END;
CREATE TRIGGER events_fts_upd AFTER UPDATE OF summary, description ON events BEGIN
    INSERT INTO events_fts(events_fts, rowid, summary, description)
        VALUES ('delete', old.rowid, old.summary, old.description);
    INSERT INTO events_fts(rowid, summary, description) VALUES (new.rowid, new.summary, new.description);
END;

-- ---------------------------------------------------------------- tasks
CREATE TABLE tasks (
    id                  TEXT PRIMARY KEY,
    tasklist_id         TEXT NOT NULL DEFAULT '@default',
    title               TEXT,
    notes               TEXT,
    due_ts              TEXT,
    status              TEXT,                      -- needsAction | completed
    completed_at        TEXT,
    updated_at          TEXT,
    position            TEXT,
    origin_proposal_id  INTEGER,
    synced_at           TEXT NOT NULL
);

CREATE INDEX idx_tasks_open ON tasks(due_ts) WHERE status = 'needsAction';

-- ---------------------------------------------------------------- proposals
-- Donna never writes an inferred action to Google. She writes a row here and asks.
-- State machine: pending -> accepted | rejected | edited | expired
CREATE TABLE proposals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kind            TEXT NOT NULL,                 -- calendar_event | task
    source_type     TEXT NOT NULL,                 -- email | conversation | manual
    source_id       TEXT,
    payload_json    TEXT NOT NULL,                 -- the proposed object
    reasoning       TEXT,                          -- why, in her words
    evidence_quote  TEXT,                          -- span of source text that justifies it
    confidence      REAL,
    state           TEXT NOT NULL DEFAULT 'pending',
    created_at      TEXT NOT NULL,
    notified_at     TEXT,
    resolved_at     TEXT,
    resolved_via    TEXT,                          -- telegram | web | cli | expiry
    result_ref      TEXT,                          -- id of the thing actually created
    trace_id        TEXT,
    dedupe_key      TEXT
);

CREATE INDEX idx_proposals_pending ON proposals(created_at) WHERE state = 'pending';
-- Prevents re-proposing the same thing, including after a rejection.
CREATE UNIQUE INDEX idx_proposals_dedupe ON proposals(dedupe_key) WHERE dedupe_key IS NOT NULL;

-- ---------------------------------------------------------------- facts (semantic memory)
CREATE TABLE facts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    text            TEXT NOT NULL UNIQUE,
    kind            TEXT,                          -- preference | routine | relationship | constraint
    source          TEXT,
    confidence      REAL,
    always_on       INTEGER NOT NULL DEFAULT 0,    -- injected into every context, not recalled
    embedding       BLOB,                          -- float32 array
    embedding_model TEXT,
    created_at      TEXT NOT NULL,
    last_seen_at    TEXT,
    superseded_by   INTEGER REFERENCES facts(id)
);

CREATE INDEX idx_facts_live ON facts(id) WHERE superseded_by IS NULL;

-- ---------------------------------------------------------------- conversation
CREATE TABLE conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    channel     TEXT NOT NULL,                     -- telegram | web | cli
    chat_id     TEXT NOT NULL,
    role        TEXT NOT NULL,                     -- user | assistant
    content     TEXT,
    agent       TEXT,
    intent      TEXT,
    trace_id    TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX idx_conversations_chat ON conversations(channel, chat_id, id DESC);

CREATE TABLE conversation_summaries (
    channel           TEXT NOT NULL,
    chat_id           TEXT NOT NULL,
    summary           TEXT,
    upto_message_id   INTEGER,
    updated_at        TEXT,
    PRIMARY KEY (channel, chat_id)
);

-- ---------------------------------------------------------------- traces
-- Every LLM call, so the web UI can answer "why did she suggest that?".
CREATE TABLE traces (
    id            TEXT PRIMARY KEY,
    parent_id     TEXT,
    task          TEXT NOT NULL,
    model         TEXT NOT NULL,
    device        TEXT NOT NULL,                   -- gpu | cpu
    system_prompt TEXT,
    prompt        TEXT,                            -- JSON messages array
    output        TEXT,
    schema_name   TEXT,
    tokens_in     INTEGER,
    tokens_out    INTEGER,
    latency_ms    INTEGER,
    load_ms       INTEGER,
    ok            INTEGER NOT NULL DEFAULT 1,
    error         TEXT,
    created_at    TEXT NOT NULL
);

CREATE INDEX idx_traces_recent ON traces(created_at DESC);
CREATE INDEX idx_traces_task   ON traces(task, created_at DESC);

-- ---------------------------------------------------------------- feedback
-- Every correction. This is the fine-tuning dataset, accumulated from day one.
CREATE TABLE feedback (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    kind             TEXT NOT NULL,                -- reclassify | proposal_accept | proposal_reject | proposal_edit
    trace_id         TEXT,
    proposal_id      INTEGER,
    email_id         TEXT,
    original_output  TEXT,
    corrected_output TEXT,
    note             TEXT,
    created_at       TEXT NOT NULL
);

CREATE INDEX idx_feedback_kind ON feedback(kind, created_at DESC);

-- ---------------------------------------------------------------- sync bookkeeping
CREATE TABLE sync_state (
    resource         TEXT PRIMARY KEY,             -- gmail | calendar | tasks
    cursor           TEXT,                         -- historyId / syncToken
    last_run_at      TEXT,
    last_success_at  TEXT,
    last_error       TEXT,
    items_seen       INTEGER NOT NULL DEFAULT 0
);
