# ADR-0011: Agent Runtime Fabric——术语、状态与边界冻结

- 状态：Superseded（部分）by ADR-0012（2026-08-19）——"Native Agent 只做治理、
  工作经 Worker 链执行"的执行模型被废除（Native Agent 自己干活，Worker 退出
  默认链）；术语冻结表仍然有效。
- 原状态：Accepted（十五审 PR-RF-0）
- 日期：2026-08-15
- 依据：《ROSClaw_NativeAgent无为而治与AgentRuntimeFabric重建方案_2026-08-15.md》

## 背景

十四审前的真实失败：一个简单任务被裂变成多个 Worker 且全部失败；Native
Agent 读日志猜原因、自建重试、跨 Worker 横跳；PTY/stdout 文本被当作状态。
根因不是某个超时参数，而是**把完整 Agent Harness 压扁成自研 WorkOrder
Adapter**：Harness 原生最有价值的能力（session/resume/compaction/权限/
内部协作）被丢弃，只剩文本猜测。

## 决定

### 术语（唯一权威定义）

| 术语 | 定义 | 反例 |
|---|---|---|
| **Provider** | 仅模型推理 API（Kimi/Anthropic/OpenAI/…） | 不是 Agent |
| **Agent Harness Runtime** | 完整 Agent 产品运行时（Codex/Claude Code/Pi/OpenCode/Gemini/Kimi CLI），自带工具循环、会话、压缩、权限、内部 subagents | 不是 ROSClaw 子进程脚本 |
| **Worker** | `Runtime Session + Task Lease + Scope + Acceptance Contract`——ROSClaw 赋予某执行会话的任务角色 | 不是一种 Agent 实现 |
| **Subagent** | Harness 内部子实例（Codex spawn_agent 等）——ROSClaw 不逐个调度，只折叠展示 | 不生成根级任务卡 |
| **Executor** | 确定性能力执行域（仿真/MCP/ROS 观测/Workflow/数据）——无 LLM | 不叫 Worker |
| **Native Agent** | 用户唯一具身入口与治理者：目标、对话、解释、验收 | 不是执行者 |

### 协议边界（固定）

- **ACP**（Agent Client Protocol）：本机完整 Harness 会话级管理；
- **Codex app-server**：Codex 的优化原生路径（runtime=codex_app_server）；
- **A2A**：远程/跨机/跨组织 Agent 服务；
- **MCP**：工具与确定性能力，不承载 Agent 会话生命周期；
- **rosclawd**：REAL/SHADOW 物理动作唯一执行与准入平面；
- **Subagent**：Harness 内部事务，ROSClaw 不调度。

### Native Agent 禁止事项（无为而治）

- 不亲自执行仿真/写代码/装依赖/猜 capability 参数；
- 不直接 `rosclaw_delegate(worker_id=...)` 挑 Worker（由 ExecutionRouter
  按注册表+策略确定性选择）；
- 不读取整本 Worker transcript 重新推理每一步（只收事件摘要+verifier 结果）；
- 不自行循环创建多个 Worker；
- 不从 stdout 文本/exit code 猜执行状态。

### 任务单一所有权

一个用户任务 = 一个 owning execution session。运行中禁止自动横跳其它
Runtime；verifier 失败把证据反馈给同一 session 修复；进程崩溃恢复同一
session；只有结构化 BLOCKED 才返回 Native Agent。

### 物理边界（不变红线）

Worker/Harness 永不持有 daemon permit/operator 私钥；物理动作只能产出
ActionProposal，经 rosclawd admission/permit/operator；E-Stop 不依赖
LLM/云；REAL 门禁保持关闭。

## 弃用（deprecated，RF-1/RF-9 移除）

- `rosclaw_delegate` 的 `worker_id` 自由选择参数；
- 每个外部 CLI 一套自研 Adapter（走 ACP/app-server 替代）；
- PTY stdout 文本状态推断；
- 自研 Pi Worker session loop（pi-worker-main 控制协议保留至 RF-9，
  其后由 ACP/pi-acp 路径替代）；
- 自动跨 Worker fallback（含 Native Basic 兜底）；
- 把 Worker attempt 当用户顶层任务展示。

## 后果

- 新代码必须使用本 ADR 术语；review 拒绝"Worker=子进程"的表述。
- 旧 API 在 RF-9 前保持只读兼容（历史任务可查）。
- Gate 3（不裂变）/Gate 4（无为）成为合并门禁。
