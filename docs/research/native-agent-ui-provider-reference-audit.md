# Native Agent UI/Provider 参考审计报告

> 依据：《ROSClaw Native Agent + Pi TUI/Provider 完整实施大纲》§2.7
> 日期：2026-08-03 · 审计人：Claude Code

## 1. 参考仓库与锁定

| 仓库 | 用途 | Commit SHA | 许可证 | clone 位置 |
|---|---|---|---|---|
| `earendil-works/pi` | pi-tui / pi-ai 直接 import；compaction/slash 借鉴 | `b9d360a2c753e058a90dc0a252950631501c10d0` | MIT | `references/pi` |
| `openai/codex` | slash command、bottom pane、App Server 协议借鉴（sparse: `codex-rs/tui`、`app-server-protocol`、`protocol`） | `79479cdf09` | Apache-2.0 | `references/codex` |
| `anomalyco/opencode` | 客户端/服务端分层、prompt_async + 全局 SSE 借鉴 | `89130db` | MIT（以仓库 LICENSE 为准） | `references/opencode` |
| `NousResearch/hermes-agent` | TUI Gateway（NDJSON/JSON-RPC 解耦）、approval modal、steering 借鉴 | `87bc710609f8b89b6e6b4aa418dde8ee30ec6873` | MIT | `references/harnesses/hermes-agent`（沿用） |
| Claude Code（黑盒） | 交互验收基线（无 vendoring） | 2.1.220（本机） | proprietary | 无 |

生产 npm 依赖精确锁定（不使用 `^`）：

```json
{
  "@earendil-works/pi-tui": "0.83.0",
  "@earendil-works/pi-ai": "0.83.0"
}
```

## 2. 具体阅读文件

### pi（MIT）

- `packages/tui/src/{index.ts, autocomplete.ts, editor-component.ts, keybindings.ts, fuzzy.ts, components/}` — IME/编辑器/overlay/滚动/自动补全 → **直接 import**。
- `packages/ai/src/{index.ts, auth/, api/anthropic-messages.ts, api/*openai*.ts, env-api-keys.ts}` — Provider 抽象、流式 tool call、OAuth/API key、usage → **直接 import**。
- `packages/coding-agent/docs/compaction.md` — compaction 算法：**只借鉴**，核心要点：倒序累计 `keepRecentTokens`（默认 20k）找切点、`CompactionEntry{summary, firstKeptEntryId}`、重复压缩从上一次 kept boundary 继续、`tokensBefore` 重算、journal 只 append 不覆写。
- `packages/coding-agent/docs/{usage.md, tui.md}`、`src/core/slash-commands.ts`、`src/modes/interactive/` — 命令注册/UI 行为：**只借鉴**。

### codex（Apache-2.0）

- `codex-rs/tui/src/slash_command.rs` — 命令枚举/排序/说明/dispatch：**借鉴命令体系设计**。
- `codex-rs/tui/src/chatwidget/slash_dispatch.rs`、`bottom_pane/command_popup.rs` — popup 交互：**借鉴**。
- `codex-rs/app-server-protocol/schema/` — 客户端/服务端协议解耦：**借鉴**。
- `codex-rs/tui/src/chatwidget/tests/` — 快照测试方法：**借鉴**。

### opencode（MIT）

- `packages/web/src/content/docs/server.mdx` — 服务端事实源、`prompt_async` 与全局 SSE 分离、Session/Provider/Command/Permission 独立 API：**借鉴**（ROSClaw 用 Mission 替换其 Coding Session）。
- `packages/opencode/` — TUI 退出不销毁任务：**借鉴**。

### hermes-agent（MIT）

- `gateway/`、`ui-tui/`、`tools/delegate_tool.py`、`agent/delegation_context.py` — NDJSON/JSON-RPC 解耦、`prompt.submit → message.delta → tool.start/progress → approval.request → complete` 事件序列、masked secret prompt、steering/follow-up：**借鉴**（事件设计映射到 AgentEventV2）。

## 3. 直接 import vs 借鉴 vs 禁止

| 对象 | 处置 | 映射到 ROSClaw |
|---|---|---|
| `@earendil-works/pi-tui` | **直接 import（0.83.0 锁版）** | `node/apps/rosclaw-tui` |
| `@earendil-works/pi-ai` | **直接 import（0.83.0 锁版）** | `node/services/rosclaw-modeld` |
| Pi compaction 算法 | 借鉴（数据结构+切点+持久化） | `agentd/context/compaction.py` + CompactionEntryV1 |
| Pi slash-command registry | 借鉴 | `node/apps/rosclaw-tui/src/commands/` |
| Codex slash/popup | 借鉴产品行为 | 同上 |
| OpenCode server 分层 | 借鉴（prompt_async+SSE） | `agentd` `/v2` API |
| Hermes TUI Gateway 事件序列 | 借鉴（事件类型与 masked prompt） | `contracts/agent/agent_event.py`（AgentEventV2） |
| `@earendil-works/pi-agent-core` | **禁止**进入运行时 | 架构测试强制 |
| `@earendil-works/pi-coding-agent` | **禁止**作为运行时 Agent | 架构测试强制 |
| Hermes `AIAgent`、Codex Agent、OpenCode Agent | **禁止**进入 Native Agent 执行路径 | 架构测试强制 |
| Claude Code TUI 实现 | 不复制；仅作黑盒验收基线 | `tests/tui/` 验收清单 |

## 4. 运行时禁令（进入架构测试）

1. TUI 直接请求模型（TUI 只走 agentd `/v2`）。
2. modeld 执行工具 / 访问 ROS、`/dev`、串口、CAN、GPIO、Robot SDK。
3. modeld 依赖 pi-agent-core / pi-coding-agent。
4. Worker 访问 rosclawd 私有 socket、Permit、Operator 私钥。
5. 模型直接调用"批准授权"接口 / 把 MCP action tool 当普通工具执行。
6. Provider 故障时自动让 Codex/Claude Worker 接管 Native Agent。
7. npm 依赖出现 `^0.83.0` 或上述禁止包名。

## 5. 结论

- **谁是 Agent**：`rosclaw-agentd`（唯一 Native Agent，认知监督者）。
- **谁是 Provider**：`rosclaw-modeld`（pi-ai，单次推理）。
- **谁是执行权威**：`rosclawd`（唯一 Permit/Receipt）。
- **TUI**：`rosclaw-tui`（pi-tui，/v2 事件客户端，不拥有 Agent）。
- **Worker**：Codex/Claude/Hermes/OpenClaw（受限承包人）。
