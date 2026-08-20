-- 027: PR-H9 旧链数据迁移——Worker/WorkOrder/TaskRunner/ControlPlane
-- 默认链已删除（总纲 v2 §18）。存量非终态行不得永远假装 RUNNING：
-- 诚实标记 CANCELLED（reason=h9_legacy_chain_removed），历史终态行
-- 原样保留（审计不可改写）。表结构保留（历史数据可读），但不再有
-- 任何生产代码写入。
UPDATE work_orders
SET status = 'CANCELLED'
WHERE status NOT IN (
    'ACCEPTED', 'FAILED', 'EXPIRED', 'CANCELLED', 'INTERRUPTED_RESUMABLE'
);

UPDATE task_records
SET state = 'CANCELLED', error = 'h9_legacy_chain_removed'
WHERE state NOT IN (
    'VERIFIED', 'FAILED', 'DENIED', 'CANCELLED', 'INCONCLUSIVE'
);

UPDATE task_executions
SET state = 'CANCELLED',
    summary = 'h9_legacy_chain_removed: control plane 已删除，任务归 TaskKernel'
WHERE state NOT IN ('SUCCEEDED', 'FAILED', 'BLOCKED', 'CANCELLED');
