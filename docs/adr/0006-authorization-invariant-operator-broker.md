# ADR-0006：授权不变量与 Operator Broker / MissionGrant

- 状态：Accepted
- 日期：2026-08-01
- 依据：实施总纲 §1.3、§11

## 背景

“用户不需要手动 arm daemon”不等于取消授权。必须明确：消除糟糕 UX 的同时，授权在架构上仍然独立于模型与 Agent。

## 决策

授权不变量（任何 PR 不得违反）：

1. `rosclawd` 是唯一物理执行与授权裁决边界。每个真实动作由 daemon 独立核验：调用者身份、body hash、SelfSnapshot freshness、capability 兼容、参数/轨迹验证、授权公开范围、lease、时效与策略。
2. Agent/Worker/模型永远不持有私密 permit；上下文中只出现 MissionGrant 的 public scope 与 hash 引用。
3. Agent 不能授权自己；mode（SIMULATION/SHADOW/REAL）只能由受信 UI/CLI 变更。
4. 授权分级：`EXACT_ACTION`（默认）→ `PLAN`（显式开启）→ `MISSION`（后期）→ `SITE_POLICY`（首版不开放）。MissionGrant 必含 principal、body、mode、scope、risk ceiling、区域、时间、动作次数/速度/能量预算、可撤销 ID、policy hash。
5. Grant 撤销即时生效并通知 daemon；过期/身体 hash 变化/参数偏离/重放/错误 principal 一律由 daemon 拒绝（fail closed）。
6. E-Stop 通路不依赖 LLM、云 API、A2A 或 Console 长连接；急停请求永远被允许，不为完成推理而延迟。
7. REAL 模式下 fixture/fake/sim 证据显式不可用；任何真实动作有最终 receipt，receipt coverage 目标 100%。

流程（替代手动 arm）：

```text
Agent 提出动作 → Context/Decision/Trajectory Validator 通过
→ Operator Broker 显示 ActionDisplay 卡片（对象/位置/路径/风险/失败处理）
→ 用户在 Console/受支持 elicitation 确认 → Broker 生成受约束授权引用
→ Agent 向 rosclawd 请求动作 → daemon 独立核验 → request status + 最终 receipt
```

`OperatorBroker` 统一 Console、CLI、MCP elicitation、人类 Worker 渠道；扩展现有 `rosclaw.interaction`（ActionDisplay v2、AuthorizationTier、InteractionCoordinator），不撤销 daemon 端强制校验。

## 后果

- 攻击回归测试（旧许可重放、body hash 变化、参数偏离、错误 principal、Console 断线）成为发布门禁 G4 的一部分。
- `rosclaw.operator` 模块只产生授权引用；签名/私材留在 Broker/daemon 侧。
