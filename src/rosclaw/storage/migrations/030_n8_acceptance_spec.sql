-- 030: PR-N8 验收规格冻结——AcceptanceSpecV2 per revision。
--
-- task_revisions.acceptance_spec_json：编译后的冻结验收规格（N0 的
-- acceptance_json 是原始输入；spec 是编译+来源归因后的权威冻结）。

ALTER TABLE task_revisions ADD COLUMN acceptance_spec_json TEXT;
