-- backend: sqlite
-- PR-NA-030b: durable model usage metering (hermes session_model_usage,
-- simplified). One row per model turn; per-mission aggregates are views
-- over this table, never separate mutable counters.
CREATE TABLE IF NOT EXISTS model_usage (
    usage_id TEXT PRIMARY KEY,
    mission_id TEXT NOT NULL,
    turn_id TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    profile TEXT,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cost_microunits INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    provider_request_id TEXT,
    context_id TEXT,
    context_revision INTEGER,
    finish_reason TEXT,
    recorded_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_model_usage_mission ON model_usage (mission_id);
