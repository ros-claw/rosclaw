---
name: rosclaw-embodied
description: ROSClaw 具身任务纪律——权威资产、证据验收、安全分层的精简操作手册（何时用：任何涉及机器人/仿真/动作的任务）
---

# ROSClaw 具身任务纪律

## 权威资产

- 机器人模型只认 e-URDF-Zoo 权威资产（`e-urdf-zoo/<robot>/robot.mjcf.xml`）；
  测试 fixture（tests/fixtures/）与简化模型绝不出现在交付证据里。
- 不确定资产来源时先调查：`rosclaw robot list` / 读取 sandbox/loader
  源码确认解析顺序，不从 / 全盘搜索同名文件。

## 证据与验收

- 交付物必须登记（rosclaw_artifact_register）——口头提到不算。
- 验收条件在任务创建时冻结；task_finish 不接受新规则。
- 机器人行为任务的完成必须有受信管道的独立验证证据；
  手写脚本的输出不算行为证据。
- 用户否定结果后任务重开 revision——不得沿用旧 receipt 宣称成功。

## 安全分层

- 纯计算/封闭本地仿真：自动执行。
- 真机动作：必须走 rosclaw_execute/request_action 的 admission 链
  （rosclawd + permit + operator），任何其他路径都不是执行权威。
- 仿真失败先读结构化诊断，按真实 Schema 修正参数；同一调用同一
  参数不机械重试。
