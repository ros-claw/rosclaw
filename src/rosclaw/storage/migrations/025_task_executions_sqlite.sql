-- backend: sqlite
-- 十五审 PR-RF-2: Task Control Plane——一个用户任务一个 owning
-- execution（Gate 3 不裂变）；WorkOrder 退居执行细节。
CREATE TABLE IF NOT EXISTS task_executions (
    execution_id TEXT PRIMARY KEY,
    mission_id TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    domain TEXT NOT NULL,          -- executor | agent_harness | physical
    runtime TEXT NOT NULL,         -- executor:simulation | harness:pi-builtin | ...
    state TEXT NOT NULL,           -- PREFLIGHT..SUCCEEDED/FAILED/BLOCKED/CANCELLED
    work_order_id TEXT,            -- harness 域的执行细节（诊断字段）
    summary TEXT,
    artifacts_json TEXT,
    verifier_feedback TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_task_executions_mission ON task_executions (mission_id);
CREATE INDEX IF NOT EXISTS idx_task_executions_fingerprint ON task_executions (fingerprint);
