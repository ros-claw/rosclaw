# 十五审 PR-RF-0：参考源码模块映射与删改清单

日期：2026-08-15（/tmp/rosclaw-agent-references/ 只读克隆，不写进生产依赖）

## 参考源码模块映射

| 来源 | 模块 | 借鉴点 | ROSClaw 落点 |
|---|---|---|---|
| openclaw/acpx `src/acp/client.ts` | ACP client（session/new、prompt、cancel、terminal-manager、error-normalization） | 完整 ACP 会话生命周期管理 | PR-RF-3 `agent_runtime/acp_client.ts` |
| openclaw/acpx `src/acp/agent-command.ts` | 各 Harness 启动命令/超时差异（Claude session/new 超时特例） | Harness 差异适配层 | PR-RF-3 runtime descriptors |
| openclaw/acpx `src/acp/terminal-manager.ts` | ACP terminal 权限路由 | PR-RF-4 sandbox 权限请求路由 |
| agentclientprotocol/agent-client-protocol | 协议 schema（initialize/session/prompt/update/cancel/permission） | 协议事实源 | PR-RF-3 合约测试对齐 |
| openclaw/openclaw codex 插件 | Codex app-server 原生集成（thread resume/compaction/tool events） | runtime=codex_app_server 优化路径 | PR-RF-5 |
| victor-software-house/pi-acp `src/acp/agent.ts`/`session.ts` | Pi ↔ ACP 映射（pi 侧实现 ACP server） | Pi 走 ACP 的现成入口 | PR-RF-3 pi harness 首选路径 |
| pi-acp docs | 明确的协议覆盖缺口 | 锁定版本 + 缺口清单进合约 | PR-RF-3 验收 |
| openai/codex app-server/multi_agents.rs | Codex 内部机制确认（spawn_agent 非异构协议） | 不做自研 CodexAdapter | ADR-0011 术语冻结 |
| earendil-works/pi | AgentSession/SessionManager/TUI（已在用） | 内置 Pi Worker 保留为默认 Harness | RF-9 前 |
| a2aproject/A2A | AgentCard/Task/SSE/push | 远程 Agent | PR-RF-7（后续） |

## 删改清单（PR-RF 序列执行）

### 删除/降级（RF-1/RF-9）
- `rosclaw_delegate(worker_id=...)` 模型自由挑 Worker → RF-1 移除参数，
  delegate  deprecated 由 `task_submit` 取代；
- Native Basic 兜底 / 跨 Worker fallback → RF-1 删除路径；
- 自研 Pi worker 控制协议（pi-worker-main.ts 自研循环）→ RF-3 接入
  pi-acp 后 RF-9 删除；
- PTY/stdout 文本状态推断 → 已有事件面替代，RF-9 删除残留；
- 把 Worker attempt 当用户顶层任务卡 → RF-2/RF-8（十四审已做一半：
  job 聚合；RF-2 升级为 task 单一所有权）。

### 新增
- `agentd/control_plane/`：TaskSpec（effects/deliverables/acceptance/
  recovery）+ ExecutionRouter + readiness preflight + owning-session
  单一约束（PR-RF-2）；
- `rosclaw-agent/src/runtime/acp-client.ts`：ACP Client Runtime
  （PR-RF-3）；
- Runtime Descriptor registry（readiness/supports/isolation 投影）
  （PR-RF-2/RF-3）；
- WorkerRuntimeSandbox（PR-RF-4）。

### 保留
- Task/Mission 账本、EventBus、artifact/evidence store、verifier、
  rosclawd/permit、TUI 主会话、Worker scope/effect/lease 安全语义、
  十四审控制协议 ACK 与 termination cause（RF-3 接入 ACP 后映射保留）。

## 红线复核

- 不复制整套 OpenClaw；acpx 只作参考/可选 vendored adapter（MIT，
  进 THIRD_PARTY_NOTICES）；
- pi-subagents 不作为 Worker Fabric 总控（ADR-0011 §0 已冻结）；
- Harness 永远接触不到 rosclawd socket/permit/operator 凭据；
- REAL 门禁保持关闭。
