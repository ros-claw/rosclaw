# ADR-0002：MissionSession 与物理 AgentSession 分离

- 状态：Accepted
- 日期：2026-08-01
- 依据：实施总纲 §4.2

## 背景

`rosclaw.daemon.session_manager.AgentSession` 是物理执行会话：UID 绑定、TTL/heartbeat、body/capability scope，服务于“谁被允许在这个时间窗内对这台身体发动作”。Native Agent 需要的是对话/任务级会话：目标、成功标准、预算、授权引用、上下文 revision、任务图 revision——生命周期以小时/天计、可跨进程重启与模型切换恢复。两者复用会造成语义混淆（聊天 session 被当物理授权，或物理 TTL 误杀长期任务）。

## 决策

1. 新增 `MissionSessionV1`（`rosclaw.mission_session.v1`，见 `contracts/agent`），由 `rosclaw-agentd` 持久化（SQLite + event journal + revision CAS）。
2. MissionSession 字段最小集：`mission_id`、`owner_principal`、`goal{text,language,success_criteria[]}`、`body_binding{body_id,effective_body_hash}`、`mode(SIMULATION|SHADOW|REAL)`、`state`、`budgets{wall_time_sec,model_tokens,monetary_microunits,worker_concurrency,physical_action_count}`、`authorization{mission_grant_id,allowed_risk_tiers}`、`context_revision`、`task_graph_revision`、时间戳。
3. 规则：
   - `mode` 只能由受信 UI/CLI 变更；模型输出不得改变 mode；
   - body hash 变化后，所有待执行物理任务节点自动置 `NEEDS_REBINDING`；
   - MissionGrant 在上下文中只出现 public scope 与 ID，私密 permit 不进入模型上下文；
   - 任一预算超限 → `WAIT_INPUT` 或 `SUSPENDED`；
   - 恢复来源是持久化状态与事件 journal，消息历史只是证据之一。
4. 物理执行时，AgentLoop 仍以现有 daemon AgentSession/permit 机制向 `rosclawd` 发起；MissionSession 持有的是 grant/approval 引用，不是物理 lease 本身。

## 后果

- 对话/任务状态可跨模型、跨重启连续；物理安全语义不被会话层稀释。
- `daemon/session_manager.py` 不做功能扩展来承载聊天语义。
