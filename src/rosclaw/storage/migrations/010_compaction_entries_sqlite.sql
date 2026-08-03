-- backend: sqlite
-- PR-07: 持久化 CompactionEntry（canonical journal 永不删除）。
CREATE TABLE IF NOT EXISTS compaction_entries (
    compaction_id TEXT PRIMARY KEY,
    mission_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    reason TEXT NOT NULL,
    entry_json TEXT NOT NULL,
    tokens_before INTEGER NOT NULL DEFAULT 0,
    tokens_after INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_compaction_mission
    ON compaction_entries (mission_id, created_at);
