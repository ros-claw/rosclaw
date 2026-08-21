-- 028: PR-N0 假成功熔断——证据 provenance + 用户接受。
--
-- artifacts.producer：登记来源（'kernel:<pipeline>' = 受信管道内
-- 部登记；'model:<tool>' = 模型工具调用）。机器人行为任务的
-- SUCCEEDED 必须含至少一个受信管道产物（模型自产证据不算数）。
--
-- tasks.user_accepted_at：用户 /done 接受时间（与 accepted_at 区分
-- ——accepted_at 是系统验收通过时间）。SUCCEEDED 但未经用户接受
-- 的任务被用户修正消息重开（revision+1，旧 verification 作废）；
-- 接受后任务永久关闭。
ALTER TABLE artifacts ADD COLUMN producer TEXT NOT NULL DEFAULT 'model:tool';
ALTER TABLE tasks ADD COLUMN user_accepted_at TEXT;
