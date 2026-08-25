-- P1-C1（0824 总纲 §7.1）：task_revisions.task_spec_json——每个
-- revision 冻结的 TaskSpecV2（intent/subjects/constraints 工单）。
ALTER TABLE task_revisions ADD COLUMN task_spec_json TEXT NOT NULL DEFAULT '';
