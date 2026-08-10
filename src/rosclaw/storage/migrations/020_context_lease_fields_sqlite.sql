-- 020：ValidatedContextLeaseV2 字段语义修正（六审 §5.3/§5.5）——
-- binding_id 必须是 session binding ID（019 实际写入 writer lease ID），
-- writer_lease_id 独立成字段；caller_pid 落库（签发时 SO_PEERCRED）。

ALTER TABLE pi_context_leases ADD COLUMN writer_lease_id TEXT NOT NULL DEFAULT '';
ALTER TABLE pi_context_leases ADD COLUMN caller_pid INTEGER NOT NULL DEFAULT -1;
