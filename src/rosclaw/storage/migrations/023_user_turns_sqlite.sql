-- 023：UserTurnV2 + task 因果链（九审 §6.1，NINE-2）。
-- 用户输入先落账（turn_id/delivery_seq/source/text_hash）；
-- task_records 增加 caused_by_turn_id——每个副作用可追溯到 turn。

CREATE TABLE IF NOT EXISTS user_turns (
    turn_id TEXT PRIMARY KEY,
    pi_session_id TEXT NOT NULL,
    mission_id TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL,
    delivery_seq INTEGER NOT NULL,
    text_hash TEXT NOT NULL,
    received_at TEXT NOT NULL,
    persisted_at TEXT NOT NULL,
    UNIQUE (pi_session_id, delivery_seq)
);

CREATE INDEX IF NOT EXISTS idx_user_turns_session
    ON user_turns (pi_session_id, delivery_seq);

ALTER TABLE task_records ADD COLUMN caused_by_turn_id TEXT NOT NULL DEFAULT '';
