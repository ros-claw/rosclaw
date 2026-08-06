-- backend: sqlite
-- PR-EV-080/081: benchmark runs + learning candidates.
CREATE TABLE IF NOT EXISTS bench_runs (
    run_id TEXT PRIMARY KEY,
    scenario_id TEXT NOT NULL,
    seed INTEGER NOT NULL,
    group_id TEXT NOT NULL,          -- A | B | C | ... （基线组）
    config_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    report_ref TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bench_runs_scenario ON bench_runs (scenario_id, group_id);

CREATE TABLE IF NOT EXISTS learning_candidates (
    candidate_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,              -- MEMORY | KNOW | HOW | AUTO
    title TEXT NOT NULL,
    content_json TEXT NOT NULL,
    evidence_class TEXT NOT NULL,    -- measured|verified_receipt|curated|inferred|unverified
    evidence_refs_json TEXT NOT NULL,
    body_scope TEXT,
    source_mission_id TEXT,
    prompt_hash TEXT,
    status TEXT NOT NULL DEFAULT 'CANDIDATE',  -- CANDIDATE|EVALUATING|PROMOTED|REJECTED
    evaluation_ref TEXT,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    promoted_by TEXT,
    promoted_at TEXT
);
