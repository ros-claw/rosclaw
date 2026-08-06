# ADR-0001：Native Agent 独立进程边界（rosclaw-agentd）

- 状态：Accepted
- 日期：2026-08-01
- 依据：实施总纲 §1、§3.1、§4

## 背景

当前 ROSClaw 只作为 Codex/Claude Code/OpenClaw 的 MCP 工具面存在，身份、会话、任务图、记忆、Worker 管理分散在外部 Harness 中，价值归因不清、换 Agent 即“失忆”。同时不能把 Agent 长循环塞回 `rosclawd`（其职责是最小权限的物理执行边界，LLM 调用不属于该信任域）。

## 决策

1. 新增 ROSClaw 自有的 Native Agent 进程 `rosclaw-agentd`（`src/rosclaw/agentd/`），以**普通用户权限**运行，不属于任何硬件控制组。
2. `rosclaw-agentd` 拥有：MissionSession、TaskGraph、ContextCompiler、ModelPolicy、DecisionValidator、认知 WorkerManager、TeamClient、ConsentClient。
3. `rosclaw-agentd` 不得：实例化物理 `Runtime` 控制面副本、注册 driver/executor、打开 `/dev`/串口/CAN/GPIO/厂商 SDK、发布危险 ROS 控制 topic、持有 daemon permit 私材。
4. 与 `rosclawd` 的通信走明确的本地 IPC（Unix socket / local HTTP），复用现有 daemon protocol 与 permit 校验；物理动作一律以 `request_action` 语义提交，由 daemon 独立裁决。
5. `rosclawd` 不加载 LLM provider，不运行 Agent loop。
6. Console（`src/rosclaw/console/`）只是 Agent/Operator 版本化 API 的视图客户端，不拥有领域状态。

## 论证

- 认知闭环（0.5–60 s+）与物理/安全闭环（0.5–500 ms）属于不同时间尺度与不同信任域（总纲 §3.2）；合并进程会把 LLM 延迟、崩溃与依赖直接引入安全边界。
- 独立进程使模型升级、崩溃恢复、预算熔断不影响 daemon 的 watchdog 与 E-Stop 通路。

## 后果

- `rosclaw-agentd` 崩溃时物理安全不受影响；daemon 端 lease 超时自动终止挂起动作（既有 orphan policy）。
- 需要静态不变量测试保证 `agentd` 模块不 import 硬件路径（`tests/architecture/`）。
- console script：`rosclaw-agentd = rosclaw.agentd.cli:main`。
