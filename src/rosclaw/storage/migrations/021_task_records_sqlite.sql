-- 021：TaskRecord（八审 P0-5）——任务级状态机持久化。
-- 一个任务 = 一个确定性编译器入口 + 一个 action goal + 自动 verifier；
-- 崩溃/compact 后可从本表恢复权威任务状态。

CREATE TABLE IF NOT EXISTS task_records (
    task_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    mission_id TEXT NOT NULL,
    goal TEXT NOT NULL,
    params_json TEXT NOT NULL,
    state TEXT NOT NULL,
    plan_id TEXT NOT NULL DEFAULT '',
    plan_digest TEXT NOT NULL DEFAULT '',
    approval_id TEXT NOT NULL DEFAULT '',
    txn_id TEXT NOT NULL DEFAULT '',
    verification_json TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_task_records_mission
    ON task_records (mission_id, state);
