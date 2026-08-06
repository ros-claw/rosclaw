-- backend: sqlite
-- PR-WF-050/051: Worker Fabric persistence (ADR-0003).
-- worker_cards: registered declarations + operator lifecycle state.
-- work_orders: dual-track state machine rows; the lease is embedded in
-- order_json but indexed here for the sweeper.
CREATE TABLE IF NOT EXISTS worker_cards (
    worker_id TEXT PRIMARY KEY,
    card_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ENABLED',   -- ENABLED | DISABLED | QUARANTINED
    trust_level TEXT NOT NULL DEFAULT 'UNVERIFIED',
    registered_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS worker_events (
    event_id TEXT PRIMARY KEY,
    worker_id TEXT NOT NULL,
    event_type TEXT NOT NULL,      -- rosclaw.worker.<entity>.<verb>.v1
    actor_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    idempotency_key TEXT,
    occurred_at TEXT NOT NULL,
    UNIQUE (worker_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS work_orders (
    work_order_id TEXT PRIMARY KEY,
    mission_id TEXT NOT NULL,
    task_id TEXT,
    worker_id TEXT,
    capability TEXT NOT NULL,
    status TEXT NOT NULL,          -- DRAFT..ACCEPTED | BLOCKED/FAILED/EXPIRED/CANCELLED
    order_json TEXT NOT NULL,
    lease_id TEXT,
    lease_expires_at TEXT,
    heartbeat_seq INTEGER NOT NULL DEFAULT 0,
    last_heartbeat_at TEXT,
    idempotency_key TEXT,
    verify_report_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_work_orders_mission ON work_orders (mission_id);
CREATE INDEX IF NOT EXISTS idx_work_orders_lease ON work_orders (lease_expires_at);
