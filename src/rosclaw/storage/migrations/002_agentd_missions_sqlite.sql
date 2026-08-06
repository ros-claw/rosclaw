-- backend: sqlite
-- PR-NA-011: MissionStore / TaskGraph persistence (ADR-0002).
-- Journal-first: mission_events is the source of truth for every state
-- transition; the missions/task_nodes rows are the current projection.

CREATE TABLE IF NOT EXISTS missions (
    mission_id TEXT PRIMARY KEY,
    owner_principal TEXT NOT NULL,
    goal_json TEXT NOT NULL,
    body_id TEXT NOT NULL,
    effective_body_hash TEXT NOT NULL,
    mode TEXT NOT NULL,
    state TEXT NOT NULL,
    budgets_json TEXT NOT NULL,
    authorization_json TEXT NOT NULL,
    context_revision INTEGER NOT NULL DEFAULT 0,
    task_graph_revision INTEGER NOT NULL DEFAULT 0,
    budget_usage_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mission_events (
    event_id TEXT PRIMARY KEY,
    mission_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT,
    reason_code TEXT,
    actor_id TEXT,
    trace_id TEXT,
    payload_json TEXT NOT NULL,
    idempotency_key TEXT,
    occurred_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    UNIQUE (mission_id, seq),
    UNIQUE (mission_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_mission_events_mission ON mission_events (mission_id, seq);

CREATE TABLE IF NOT EXISTS task_nodes (
    mission_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    node_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (mission_id, task_id)
);

CREATE TABLE IF NOT EXISTS task_edges (
    mission_id TEXT NOT NULL,
    from_task TEXT NOT NULL,
    to_task TEXT NOT NULL,
    revision INTEGER NOT NULL,
    PRIMARY KEY (mission_id, from_task, to_task)
);

CREATE TABLE IF NOT EXISTS idempotency_records (
    scope TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    status TEXT NOT NULL,
    result_digest TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (scope, idempotency_key)
);
