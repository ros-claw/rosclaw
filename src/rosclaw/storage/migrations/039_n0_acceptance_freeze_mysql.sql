-- backend: mysql
-- 039: 028 的 MySQL 双生（PR-SDB-140-2)。028 的 `TEXT NOT NULL DEFAULT` 在
-- MySQL 方言非法（错误 1067)，拆出方言变体；SQLite 侧 028 已应用且受
-- checksum 保护，不得修改。
ALTER TABLE artifacts ADD COLUMN producer VARCHAR(255) NOT NULL DEFAULT 'model:tool';
ALTER TABLE tasks ADD COLUMN user_accepted_at TEXT;
