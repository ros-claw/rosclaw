-- backend: sqlite
-- 十四审 PR-14.2: 稳定 Job + Attempt 账本（RetryCoordinator 是唯一重试
-- 决策者——总纲 §3.5）。一个用户任务 = 一个 root Job（root_work_order_id）；
-- retry/resume 只是新 attempt，UI 永远只显示一张任务卡。
CREATE TABLE IF NOT EXISTS worker_jobs (
    root_job_id TEXT PRIMARY KEY,      -- = 首个 attempt 的 root_work_order_id
    mission_id TEXT NOT NULL,
    user_goal TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS worker_attempts (
    attempt_id TEXT PRIMARY KEY,       -- = work_order_id（内部诊断字段）
    root_job_id TEXT NOT NULL,
    attempt_seq INTEGER NOT NULL,
    actor TEXT NOT NULL,               -- user | auto | native_agent
    failure_fingerprint TEXT NOT NULL DEFAULT '',
    termination_cause TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT 'ACTIVE',  -- ACTIVE | SETTLED
    created_at TEXT NOT NULL,
    settled_at TEXT,
    UNIQUE (root_job_id, attempt_seq)
);

-- auto retry 幂等：同一 root 同一失败指纹最多自动重试一次（手动/用户
-- 权威的 retry 不受此限——但同一时刻只能有一个 ACTIVE attempt）。
CREATE UNIQUE INDEX IF NOT EXISTS idx_worker_attempts_fingerprint
    ON worker_attempts (root_job_id, failure_fingerprint)
    WHERE failure_fingerprint <> '';

-- 活跃 attempt 唯一：一个 root job 同时最多一个 ACTIVE attempt
-- （coordinator 应用层 CAS + 本约束兜底）。
CREATE UNIQUE INDEX IF NOT EXISTS idx_worker_attempts_active
    ON worker_attempts (root_job_id) WHERE state = 'ACTIVE';

CREATE INDEX IF NOT EXISTS idx_worker_attempts_root ON worker_attempts (root_job_id);
