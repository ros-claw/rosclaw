-- P0-D（0824 总纲 §7.4/§19.P0-D）：TaskOutcomeV2 与修复指令。
-- BLOCKED 不再是万能终态——结果拆六维；delivery 失败不关 lifecycle。
CREATE TABLE IF NOT EXISTS task_outcomes (
    outcome_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    outcome_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(task_id, revision)
);
-- 同一错误指纹的修复尝试记录（同指纹再现 → WAITING_INPUT，
-- 不继续烧 token）。
CREATE TABLE IF NOT EXISTS task_repairs (
    repair_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    directive_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_task_repairs_task
    ON task_repairs(task_id, fingerprint);
