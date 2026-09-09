# W05 交付生命周期：追加交付（规格 §9.2）

**分支**：a-w05-delivery-lifecycle（基于 main=ab0ae5a，§17 依赖：W05←W00）
**日期**：2026-09-09

## 问题（W00 R1 / 0907 实证）

`_artifact_register` 只认「最近 active task」：任务 SUCCEEDED
后模型补交付物（补图/补分析/改视角产物）被
`TASK_ALREADY_COMPLETED` 反复拒绝；而 dispatcher 前置的
`ensure_task_for_effect` 在有未附着新输入时会把迟到请求绑定成
新 revision（RUNNING 复活反模式）。

## 改动（`src/rosclaw/agentd/pi_bridge/tool_dispatch.py`）

1. **追加交付**：无 active task 时查最近任务——终态
   （SUCCEEDED/FAILED/BLOCKED/CANCELLED）即追加：登记在既有
   任务的当前 revision，**不改回 RUNNING、不 bump revision、
   审计历史保持终态**。追加产物 metadata 标
   `appended_post_terminal` + `task_state_at_registration`
   （不冒充验收期交付）；结果 summary 带可行动引导
   （「任务保持 SUCCEEDED……用户有新目标时会开始新任务」）。
2. **迟到请求不自动激活新 revision**：`rosclaw_deliver` /
   `rosclaw_artifact_register` 移除 dispatcher 前置
   `ensure_task_for_effect`；admission 收缩进
   `_artifact_register` 且**仅在无任务史时**触发（P0-C 交付
   优先金丝雀保留）。有任务史时未附着的新输入不被迟到工具
   请求绑定——新 revision 由该输入自己的回合创建。
3. **模型不能挂任意任务**：请求合约无 `task_id` 字段（既有
   保证，回归测试锚定）——任务解析只在服务端按
   mission+session。

## 验证（红→绿）

| 测试 | 结果 |
|---|---|
| test_w05_delivery_lifecycle.py（5：wire 路径追加/追加标记可审计/迟到请求不激活新 revision/交付优先 admission 保留/合约无 task_id） | 红（4 failed：拒绝+复活确认）→ 绿 5/5 |
| W00 R1 解标 | 绿（剩 R2/R3 xfail = W03/W04 目标——W03 已在其分支解 R2） |
| test_canary_residuals 语义更新（引导进追加 summary，不再拒绝） | 绿 |
| test_pi_tool_bridge 验证链 | 绿 |
| tests/agentd + tests/sandbox 全量（除 PTY journey） | 见 PR（后台跑） |
| ruff check src tests | 全过 |

## §9 其余条款现状（本轮未改，记录备查）

- §9.1 客观事实进结果对象：大道至简 R0-2 已落（kernel 只报客观
  执行事实；UI 不按 exit 0 显示成功）——既有。
- §9.3 幂等/取消/重启：execute() idempotency 重放 + Operation
  状态机（前期审计）——既有；REAL UNKNOWN 不重试——REAL 门禁
  仍关闭。
- §9.4 Artifact 来源独立：内容寻址幂等 upsert + producer 区分
  （kernel:/model:）+ evidence 区 kernel-only——既有（P0-E/N4.1）。
- 真实模型追加交付验收：**NOT_RUN**（无 key，不合成冒充）。
