-- 031: WP-1 终态一致性——operations.revision（启动时 task 活跃
-- revision）。旧 revision 的迟到终态事件只存档，不消费。
ALTER TABLE operations ADD COLUMN revision INTEGER;
