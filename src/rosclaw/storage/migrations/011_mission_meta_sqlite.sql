-- backend: sqlite
-- 批次 B: Mission 展示元数据（不改 MissionSessionV1 契约/状态机）。
CREATE TABLE IF NOT EXISTS mission_meta (
    mission_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL DEFAULT '',
    archived INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);
