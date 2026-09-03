-- 0902 审计 R1-a：Approval Broker（shell 降级授权）——删除全局环境
-- 变量授权方案（ROSCLAW_ALLOW_UNSANDBOXED_SHELL）的正式路径。
--
-- 0902 实证：用户已在会话里明确回答"允许！"，系统仍要求其退出、
-- export 全局环境变量、重启——全局、粗粒度、难撤销、难审计。
--
-- grant 绑定：task_id + revision + scope（shell.unsandboxed）+
-- mission + session + 决定时间——本任务允许只对当前 revision 有效
--（revision 变化 = 语义变化 = 重新询问）。一次允许 = 该 request
-- 行本身（consume-once，不落 standing grant）。
CREATE TABLE IF NOT EXISTS shell_gate_requests (
    request_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    mission_id TEXT NOT NULL,
    session_ref TEXT NOT NULL,
    scope TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    decided_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_shell_gate_task
    ON shell_gate_requests(task_id, revision, scope);

-- 本任务级 standing grant（本任务允许同类操作——任务终态或 revision
-- 变化后失效由 check 侧判定：grant 行的 revision 与当前
-- active_revision 不符即不再命中）。
CREATE TABLE IF NOT EXISTS shell_grants (
    grant_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    mission_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(task_id, revision, scope)
);
