-- P0-A（0824 总纲 §19.P0-A）：(session_id, event_id) 幂等去重。
-- 调用方提供的稳定 event_id（provider retry/断流重连/重放注入的
-- 同一逻辑事件）只落账一次；mission 级事件（session_id 为 NULL）
-- 不受约束（它们由 mission 内 sequence 排序）。
CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_events_session_event
    ON agent_events(session_id, event_id)
    WHERE session_id IS NOT NULL;
