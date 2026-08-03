-- backend: sqlite
-- 批次 F: ReasoningBranch（推理分支树；物理事实线不在此表，永远只追加）。
CREATE TABLE IF NOT EXISTS reasoning_branches (
    branch_id TEXT PRIMARY KEY,
    mission_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    branch_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_branches_mission
    ON reasoning_branches (mission_id, created_at);
