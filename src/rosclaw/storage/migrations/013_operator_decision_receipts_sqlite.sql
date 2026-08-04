-- 013：Operator Decision Protocol v1（二次复核 R1/R3）
--
-- decision_receipts：agentd 已应用的 daemon 签名 DecisionReceiptV1——
--   challenge_nonce UNIQUE 是重放防线（同一 receipt 第二次 apply 必败）。
-- operatord_keys：SIM（DEV_SIM_ONLY）剖面下 operatord Ed25519 公钥的
--   TOFU 钉住——同 enrollment_id 公钥变化即拒绝并告警。

CREATE TABLE IF NOT EXISTS decision_receipts (
    receipt_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    challenge_nonce TEXT NOT NULL UNIQUE,
    decision TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS operatord_keys (
    enrollment_id TEXT PRIMARY KEY,
    public_key_pem TEXT NOT NULL,
    first_seen_at TEXT NOT NULL
);
