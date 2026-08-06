# ADR-0007：双层 Operator Consent 集成路径（agentd grants ↔ daemon proposals）

- 状态：Accepted
- 日期：2026-08-02
- 依据：上游 #185（feat/operator-broker-core，daemon 侧 consent plane）与
  ADR-0006（agentd 侧 MissionGrant）的合并整合

## 背景

两个互补的 consent 层同时存在：

- **agentd 侧 OperatorBroker**（本仓库 `rosclaw/operator/broker.py`）：
  面向认知层的 ApprovalRequestV2 卡片与 MissionGrant public scope，
  解决"Agent 如何向用户呈现动作意图并引用授权"。
- **daemon 侧 operator-broker consent plane**（上游 #185，
  `rosclaw/operator/{protocol,store}.py` + daemon service RPC）：
  proposal 绑定 Unix peer UID、body snapshot、action-intent hash、daemon
  generation、nonce、TTL；只有 daemon 服务 UID 能列出/裁决；接受即生成
  daemon Permit 并以原 Agent UID 提交，监督到终态 Receipt。

## 决策

1. agentd 的 `REQUEST_APPROVAL` 面向用户（Console/CLI）走 agentd broker；
   涉及 REAL 动作时，agentd 必须额外经 `operator.proposal.create` 向
   rosclawd 提交 proposal，由 daemon consent plane 独立裁决——agentd 的
   MissionGrant 永不替代 daemon proposal/permit。
2. 映射关系：ApprovalRequestV2 ≈ OperatorProposal（认知层呈现 vs 物理层
   提案）；MissionGrant public_hash 可作为 proposal 的关联引用，但不携带
   任何许可效力。
3. `REQUEST_ACTION` 的真实执行路径（REAL）：
   agentd broker verify（public scope）→ daemon proposal decide（服务 UID）
   → daemon Permit → ActionEnvelope 提交 → Receipt。SIMULATION 维持当前
   `agentd/action_channel.py` 的 SIM executor 路径。
4. 两侧都遵守：pending 在 daemon 重启后失效；参数变化、challenge 不符、
   intent hash 不符、过期、Agent Session 丢失、无 REAL executor 全部
   fail closed。

## 后果

- `rosclaw/operator/__init__.py` 同时导出两层（`broker` 与
  `protocol`/`store`），命名不冲突。
- agentd→daemon proposal 的具体接线是后续 PR（依赖本合并先落地）。
