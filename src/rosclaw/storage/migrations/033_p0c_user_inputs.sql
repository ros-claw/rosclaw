-- P0-C（0824 总纲 §6.1/§19.P0-C）：Conversation/Task 分离。
-- 每条输入先落 user_inputs（不立即创建 Task）；首个 effectful
-- call 经 ensure_task_for_effect 原子建 task 并回写 task_id。
CREATE TABLE IF NOT EXISTS user_inputs (
    input_id TEXT PRIMARY KEY,
    mission_id TEXT NOT NULL,
    session_ref TEXT NOT NULL,
    message_id TEXT NOT NULL UNIQUE,
    text TEXT NOT NULL,
    text_digest TEXT NOT NULL,
    task_id TEXT,
    delivery_state TEXT NOT NULL DEFAULT 'PERSISTED',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_user_inputs_session
    ON user_inputs(mission_id, session_ref, created_at);
