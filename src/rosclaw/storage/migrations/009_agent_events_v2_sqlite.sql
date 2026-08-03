-- backend: sqlite
-- PR-02: AgentEventV2 持久事件流（大纲 §9）。
-- 事件必须先落 journal（本表，事务内 sequence 单调递增），再推给 UI。
CREATE TABLE IF NOT EXISTS agent_events (
    event_id TEXT PRIMARY KEY,
    mission_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    turn_id TEXT,
    task_id TEXT,
    trace_id TEXT,
    type TEXT NOT NULL,              -- e.g. mission.state.changed
    visibility TEXT NOT NULL,        -- USER | DEBUG | AUDIT
    payload_json TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    UNIQUE (mission_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_agent_events_mission
    ON agent_events (mission_id, sequence);
