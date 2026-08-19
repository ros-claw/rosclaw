-- PR-H2（ADR-0012，总纲 v2 §9.2）：Task Kernel schema——
-- 一个用户目标 = 一个 root task = 一个 workspace = 一个 active
-- primary Harness Session。task_execution/work_order/job 语义不再
-- 混用（旧表由迁移器只读处理，正常运行不写）。

CREATE TABLE IF NOT EXISTS tasks (
  task_id TEXT PRIMARY KEY,
  mission_id TEXT,
  root_goal TEXT NOT NULL,
  mode TEXT NOT NULL,
  body_id TEXT,
  workspace_path TEXT NOT NULL,
  state TEXT NOT NULL,
  active_revision INTEGER NOT NULL DEFAULT 1,
  locale TEXT NOT NULL DEFAULT 'auto',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  accepted_at TEXT,
  terminal_reason TEXT
);

CREATE TABLE IF NOT EXISTS task_revisions (
  task_id TEXT NOT NULL,
  revision INTEGER NOT NULL,
  user_message_id TEXT NOT NULL UNIQUE,
  goal_delta TEXT NOT NULL,
  acceptance_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  PRIMARY KEY(task_id, revision)
);

CREATE TABLE IF NOT EXISTS harness_sessions (
  session_ref TEXT PRIMARY KEY,
  backend_id TEXT NOT NULL,
  backend_native_id TEXT NOT NULL,
  cwd TEXT NOT NULL,
  model_profile_revision TEXT,
  state TEXT NOT NULL,
  created_at TEXT NOT NULL,
  closed_at TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  UNIQUE(backend_id, backend_native_id)
);

CREATE TABLE IF NOT EXISTS task_session_bindings (
  task_id TEXT NOT NULL,
  session_ref TEXT NOT NULL,
  role TEXT NOT NULL,
  active INTEGER NOT NULL,
  supersedes_session_ref TEXT,
  created_at TEXT NOT NULL,
  PRIMARY KEY(task_id, session_ref)
);

-- 一个 task 同时只有一个 active primary session（DB 级不变量）。
CREATE UNIQUE INDEX IF NOT EXISTS one_active_primary_session
ON task_session_bindings(task_id)
WHERE role = 'primary' AND active = 1;

CREATE TABLE IF NOT EXISTS task_attempts (
  attempt_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  revision INTEGER NOT NULL,
  actor_type TEXT NOT NULL,
  actor_ref TEXT NOT NULL,
  state TEXT NOT NULL,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  failure_code TEXT
);

CREATE TABLE IF NOT EXISTS operations (
  operation_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  attempt_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  state TEXT NOT NULL,
  resumable INTEGER NOT NULL DEFAULT 0,
  checkpoint_json TEXT,
  started_at TEXT NOT NULL,
  heartbeat_at TEXT,
  ended_at TEXT,
  failure_code TEXT
);

CREATE TABLE IF NOT EXISTS task_events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL,
  session_ref TEXT,
  attempt_id TEXT,
  operation_id TEXT,
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  path TEXT NOT NULL,
  media_type TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  producer_operation_id TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verifications (
  verification_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  revision INTEGER NOT NULL,
  status TEXT NOT NULL,
  checks_json TEXT NOT NULL,
  evidence_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
