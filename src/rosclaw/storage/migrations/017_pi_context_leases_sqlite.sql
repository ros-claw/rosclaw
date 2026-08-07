-- 017：ValidatedContextLeaseV1（四审 HOTFIX-1，P0-4A）
-- agentd 签发的短期具身上下文准入证——action propose/execute 必须
-- 出示有效 lease；context fetch 失败/TTL 到期/session 切换立即失效。
-- 模型永远看不到它（只对 admission 有效，不是执行权）。

CREATE TABLE IF NOT EXISTS pi_context_leases (
    context_lease_id TEXT PRIMARY KEY,
    pi_session_id TEXT NOT NULL,
    mission_id TEXT NOT NULL,
    context_revision INTEGER NOT NULL,
    context_hash TEXT NOT NULL,
    body_hash TEXT NOT NULL,
    mode TEXT NOT NULL,
    issued_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    revoked INTEGER NOT NULL DEFAULT 0
);

-- 一个 (session, mission) 同时只有一个有效 lease（新签发即撤旧）。
CREATE INDEX IF NOT EXISTS idx_pi_context_leases_session
    ON pi_context_leases (pi_session_id, mission_id, revoked);
