# ADR-0000：ADR 流程、命名规范与功能成熟度等级

- 状态：Accepted
- 日期：2026-08-01
- 依据：《ROSClaw Native Agent、Worker Fabric 与多机器人 Team Fabric 实施总纲 v1.0》§16 PR-NA-000

## 背景

ROSClaw 从“可信物理执行 Runtime”扩展为“具身 Agent OS”（新增 Native Agent、Worker Fabric、Team Fabric）之前，需要先冻结架构决策的记录方式、跨进程契约的命名规范、以及功能成熟度标签，避免新模块边界漂移。

## 决策

### 1. ADR 流程

- 所有架构级决策（新进程、新信任边界、新契约族、新持久化域、改变既有边界）必须先有 `docs/adr/NNNN-<kebab-title>.md`。
- ADR 使用状态机：`Proposed → Accepted → Superseded by ADR-XXXX / Deprecated`。
- 修改稳定规则（物理边界、授权不变量、schema 主版本）必须附回归评测或静态不变量测试。

### 2. Schema 命名规范

- 每个跨进程/跨版本契约对象带 `schema_version` 字段，格式：`rosclaw.<domain>.<entity>.v<N>`，例如 `rosclaw.mission_session.v1`、`rosclaw.work_order.v1`。
- 契约代码集中在 `src/rosclaw/contracts/{agent,worker,team}`；领域私有 DTO 仍在各模块内，但不得跨进程使用。
- 契约 reader 必须 forward-compatible：未知字段保留或忽略并记录，未知主版本拒绝（fail closed）。
- 所有契约对象有 canonical JSON（键排序、无空白差异）与内容 hash（`sha256`，带类型前缀如 `ctxb_`、`body_`）。
- 公开契约（会进入模型上下文或跨信任边界）禁止出现 secret-like 字段（`api_key`、`secret`、`token`、`password`、`permit` 私材等）；凭据只传引用（`api_key_ref`、`credential_profile_ref`）。

### 3. 事件命名规范

- 统一 `rosclaw.<domain>.<entity>.<verb>.v1`，如 `rosclaw.agent.mission.created.v1`、`rosclaw.worker.work_result.submitted.v1`。
- 事件 envelope 必备：`event_id`、`schema_version`、`occurred_at`、`recorded_at`、`trace_id`/`span_id`、`mission_id`（如适用）、`actor_id`、`producer`、`body/team binding`、`idempotency_key`、`payload_digest`。
- 进程内 EventBus 不承担跨进程投递；跨进程事件必须先写本地状态 + outbox，再投递，消费端以 idempotency key 去重。

### 4. 功能成熟度等级

| 等级 | 含义 | 要求 |
|---|---|---|
| `experimental_legacy` | 历史原型，无生产语义保证 | 文档/CLI 标注；冻结新增功能；不得被新代码依赖 |
| `experimental` | 新功能探索，接口可破 | 有测试，不得默认开启 |
| `preview` | 接口趋稳，限场景可用 | 契约冻结、迁移说明、故障语义文档化 |
| `stable` | 生产承诺 | 完整测试金字塔、版本矩阵、升级/回滚说明 |

- 等级标注在模块 `__init__` docstring、CLI `--help` 与 docs 中一致出现。
- 首批标注：`rosclaw.swarm` = `experimental_legacy`；`rosclaw.agent_runtime.ai_collaboration` / `llm_provider` = `experimental_legacy`（由 ContextCompiler/ModelGateway/AgentLoop 替换）。

### 5. 目标形态与非目标（写入 ARCHITECTURE.md）

目标形态：Native Agent（`rosclaw-agentd`）拥有身份/会话/任务图/上下文/Worker/Team 生命周期；`rosclawd` 保持唯一物理执行与授权边界；外部 Agent（Codex、Claude Code、PicoClaw、ZeroClaw 等）是受管 Worker。

非目标（首版）：不做通用消息渠道/插件市场/浏览器自动化；不自研基础模型、SLAM、实时控制器；不让 LLM 进入毫秒级闭环；不支持 Agent 自我授权。

## 后果

- 新增模块（`agentd`、`contracts`、`team`、`operator`）的默认成熟度从 `experimental` 起步，达到对应门禁后晋升。
- 静态架构不变量测试（`tests/architecture/`）成为 CI 一部分。
