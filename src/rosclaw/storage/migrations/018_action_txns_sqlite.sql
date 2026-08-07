-- 018：ActionTxn（四审 HOTFIX-2，P0-4C）——动作事务状态机。
-- request→session→mission→context→approval→grant→action→receipt
-- 全 ID 链的单一持久化承载；idempotency_key UNIQUE 防重复建卡。

CREATE TABLE IF NOT EXISTS action_txns (
    txn_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    request_hash TEXT NOT NULL,
    pi_session_id TEXT NOT NULL,
    mission_id TEXT NOT NULL,
    context_lease_id TEXT NOT NULL,
    context_revision INTEGER NOT NULL,
    body_hash TEXT NOT NULL,
    mode TEXT NOT NULL,
    capability_id TEXT NOT NULL,
    arguments_hash TEXT NOT NULL,
    risk_tier TEXT NOT NULL,
    approval_id TEXT NOT NULL DEFAULT '',
    display_hash TEXT NOT NULL DEFAULT '',
    grant_id TEXT NOT NULL DEFAULT '',
    action_id TEXT NOT NULL DEFAULT '',
    receipt_id TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    completed_at TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_action_txns_mission
    ON action_txns (mission_id, state);
CREATE INDEX IF NOT EXISTS idx_action_txns_approval
    ON action_txns (approval_id);
