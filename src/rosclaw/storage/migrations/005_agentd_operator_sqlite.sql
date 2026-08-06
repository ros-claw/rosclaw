-- backend: sqlite
-- PR-OP-060/061: Operator Broker persistence.
-- operator_requests: approval cards shown to a human, with resolution.
-- mission_grants: public scope only. The private signature/permit lives
-- in grant_private — never serialized into agent-facing context.
CREATE TABLE IF NOT EXISTS operator_requests (
    request_id TEXT PRIMARY KEY,
    mission_id TEXT NOT NULL,
    task_id TEXT,
    request_json TEXT NOT NULL,         -- ApprovalRequestV2 (public)
    status TEXT NOT NULL DEFAULT 'PENDING',  -- PENDING | APPROVED | DENIED | EXPIRED
    decided_by TEXT,
    decided_at TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_operator_requests_mission
    ON operator_requests (mission_id);

CREATE TABLE IF NOT EXISTS mission_grants (
    grant_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    public_json TEXT NOT NULL,          -- MissionGrantV1 public scope
    private_signature TEXT NOT NULL,    -- broker-side only, never leaves broker
    consumed INTEGER NOT NULL DEFAULT 0,
    revoked INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
