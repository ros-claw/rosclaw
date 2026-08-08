-- 019：ValidatedContextLeaseV2（五审 P0-5A）——caller 身份绑定。
-- lease 记录 binding_id 与 caller_uid；签发时必须验证 writer owner，
-- admission 校验 lease 的 caller 与当前调用进程一致。

ALTER TABLE pi_context_leases ADD COLUMN binding_id TEXT NOT NULL DEFAULT '';
ALTER TABLE pi_context_leases ADD COLUMN caller_uid INTEGER NOT NULL DEFAULT -1;
