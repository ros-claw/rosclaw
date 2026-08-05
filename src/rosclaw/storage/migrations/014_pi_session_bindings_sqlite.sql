-- 014：Pi Session ↔ Mission 绑定与 writer lease（重构规格 §12，PR-PNA-1）
--
-- pi_session_bindings：一个 Pi Session 只绑定一个 active Mission；
--   一个 Mission 可有多个历史认知 Session。
-- pi_session_leases：同一 Mission 同时只有一个主认知 writer；
--   lease 过期可回收（崩溃进程不永久占锁）。

CREATE TABLE IF NOT EXISTS pi_session_bindings (
    binding_id TEXT PRIMARY KEY,
    pi_session_id TEXT NOT NULL,
    pi_session_path TEXT NOT NULL DEFAULT '',
    mission_id TEXT NOT NULL,
    body_id TEXT NOT NULL DEFAULT '',
    execution_mode TEXT NOT NULL DEFAULT 'SIMULATION',
    created_at TEXT NOT NULL,
    created_by TEXT NOT NULL,
    parent_binding_id TEXT,
    source_mission_id TEXT,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    binding_revision INTEGER NOT NULL DEFAULT 1
);

-- 一个 active binding 只属一个 (pi_session_id)（status=ACTIVE 唯一）。
CREATE UNIQUE INDEX IF NOT EXISTS idx_pi_binding_active_session
    ON pi_session_bindings (pi_session_id) WHERE status = 'ACTIVE';

CREATE TABLE IF NOT EXISTS pi_session_leases (
    lease_id TEXT PRIMARY KEY,
    mission_id TEXT NOT NULL,
    pi_session_id TEXT NOT NULL,
    owner_pid INTEGER NOT NULL,
    owner_uid INTEGER NOT NULL,
    host_id TEXT NOT NULL DEFAULT '',
    lease_token_hash TEXT NOT NULL,
    issued_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    heartbeat_at TEXT NOT NULL
);

-- 未过期 lease 对 mission 唯一（一个 writer）。
CREATE UNIQUE INDEX IF NOT EXISTS idx_pi_lease_active_mission
    ON pi_session_leases (mission_id);
