-- 029: PR-HP1 NativeEventV2——一等链路段落库（调整方案 §四）。
--
-- session_id/revision/item_id/call_id/operation_id/model_visible 成为
-- agent_events 的一等列：resume/trace/TUI 投影直接读列，不挖
-- payload JSON。全部可空（旧行兼容）。

ALTER TABLE agent_events ADD COLUMN session_id TEXT;
ALTER TABLE agent_events ADD COLUMN revision INTEGER;
ALTER TABLE agent_events ADD COLUMN item_id TEXT;
ALTER TABLE agent_events ADD COLUMN call_id TEXT;
ALTER TABLE agent_events ADD COLUMN operation_id TEXT;
ALTER TABLE agent_events ADD COLUMN model_visible INTEGER;
