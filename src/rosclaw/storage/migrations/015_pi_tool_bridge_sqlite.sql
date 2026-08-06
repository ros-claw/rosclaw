-- 015：Pi Tool Bridge 幂等与审计（重构规格 §17，PR-PNA-3）

CREATE TABLE IF NOT EXISTS pi_tool_idempotency (
    idempotency_key TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
