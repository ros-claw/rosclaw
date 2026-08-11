-- 022：SessionCatalogV1（总纲 WP-P0-2）——产品级会话索引。
-- 标题/摘要/Robot/Mode/Task 状态/最近活动/成本/归档/lineage。
-- 红线：这些是产品检索投影，不得塞进安全用的 pi_session_bindings。

CREATE TABLE IF NOT EXISTS agent_sessions (
    session_id TEXT PRIMARY KEY,
    pi_session_path TEXT NOT NULL,
    display_name TEXT NOT NULL,
    title_source TEXT NOT NULL DEFAULT 'auto',
    mission_id TEXT,
    body_id TEXT,
    execution_mode TEXT NOT NULL DEFAULT '',
    lifecycle_state TEXT NOT NULL DEFAULT 'NEW',
    task_state TEXT,
    last_task_id TEXT,
    last_receipt_id TEXT,
    provider_id TEXT,
    model_id TEXT,
    locale TEXT,
    summary TEXT NOT NULL DEFAULT '',
    search_text TEXT NOT NULL DEFAULT '',
    message_count INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_microunits INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    last_active_at TEXT NOT NULL,
    completed_at TEXT,
    archived_at TEXT,
    revision INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_agent_sessions_active
    ON agent_sessions (archived_at, last_active_at);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_search
    ON agent_sessions (search_text);

CREATE TABLE IF NOT EXISTS agent_session_lineage (
    child_session_id TEXT PRIMARY KEY,
    parent_session_id TEXT NOT NULL,
    parent_entry_id TEXT,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);
