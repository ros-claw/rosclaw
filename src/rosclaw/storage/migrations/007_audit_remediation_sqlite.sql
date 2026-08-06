-- backend: sqlite
-- 审计修复（对照总纲 §10.8/§12.2/§12.3）：
-- - team_tasks 增加副作用类别：成员失联时只有无副作用任务才可重新公告；
-- - decisions / context_manifests / work_results：打通 G4 归因链；
-- - operator_events：Operator 域事件日志；
-- - broker_state：broker 签名密钥持久化（随机生成，不可由公开策略推导）。
ALTER TABLE team_tasks ADD COLUMN side_effect_class TEXT NOT NULL DEFAULT 'none';

CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    mission_id TEXT NOT NULL,
    context_id TEXT NOT NULL,
    context_revision INTEGER NOT NULL,
    decision_json TEXT NOT NULL,
    validated INTEGER NOT NULL,
    reason_code TEXT,
    actor_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decisions_mission ON decisions (mission_id);

CREATE TABLE IF NOT EXISTS context_manifests (
    context_id TEXT NOT NULL,
    context_revision INTEGER NOT NULL,
    mission_id TEXT NOT NULL,
    bundle_hash TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (context_id, context_revision)
);

CREATE TABLE IF NOT EXISTS work_results (
    work_order_id TEXT NOT NULL,
    lease_id TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (work_order_id, lease_id)
);

CREATE TABLE IF NOT EXISTS operator_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,   -- rosclaw.operator.<entity>.<verb>.v1
    actor_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    occurred_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS broker_state (
    key TEXT PRIMARY KEY,
    value BLOB NOT NULL
);
