-- 016：Pi 认知事件镜像（重构规格 §24，PR-PNA-8）
-- 只镜像哈希/元数据，绝不镜像 assistant 全文（防双写不一致）。

CREATE TABLE IF NOT EXISTS pi_event_mirrors (
    mirror_id TEXT PRIMARY KEY,
    pi_session_id TEXT NOT NULL,
    mission_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    pi_entry_id TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    usage_json TEXT NOT NULL DEFAULT '{}',
    occurred_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pi_mirrors_mission ON pi_event_mirrors (mission_id);
CREATE INDEX IF NOT EXISTS idx_pi_mirrors_session ON pi_event_mirrors (pi_session_id);
