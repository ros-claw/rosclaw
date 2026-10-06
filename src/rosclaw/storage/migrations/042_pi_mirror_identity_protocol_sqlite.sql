-- backend: sqlite
-- 042：pi_event_mirrors durable opt-in 身份协议标记（BOUNDED-IDENTITY，
-- NATIVE_MIRROR_IDENTITY_COMPAT）。
-- 新列 identity_protocol 持久区分 prospective stable-entry/v1 身份与
-- 历史 legacy 行：既有行取默认值 ''（legacy mirror_id 准入语义不变——
-- 不回填、不合并、不重分类任何历史用量）；只有显式声明协议的新行写入
-- 'stable-entry/v1'。稳定身份 (pi_session_id, mission_id, event_type,
-- pi_entry_id) 只在同协议行内做幂等/冲突匹配——legacy 同 entry 多载荷
-- 行保持原字节，绝不坍缩，也绝不污染新的 opt-in 稳定身份。

ALTER TABLE pi_event_mirrors
    ADD COLUMN identity_protocol TEXT NOT NULL DEFAULT '';

-- 稳定身份匹配限定在同协议行——部分索引与该谓词精确对齐；
-- legacy 行不进入此索引（历史行零额外索引开销）。
CREATE INDEX IF NOT EXISTS idx_pi_mirrors_stable_identity
    ON pi_event_mirrors (pi_session_id, mission_id, event_type, pi_entry_id)
    WHERE identity_protocol = 'stable-entry/v1';
