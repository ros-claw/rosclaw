# Third-Party Notices

## @earendil-works/pi-coding-agent (MIT)

ROSClaw Native Agent 与内置 Worker 构建在 Pi agent harness（MIT）之上。

以下文件的实现思想移植自上游 `examples/extensions/subagent/index.ts`
（JSONL 行缓冲与事件解析、usage 聚合、AbortSignal + SIGTERM/SIGKILL、
partial output 处理）：

- `packages/rosclaw-agent/src/workers/pi-worker-main.ts`
- `src/rosclaw/agentd/workers/pi_managed.py`（supervisor 语义）
- `src/rosclaw/agentd/workers/process.py`（进程组清理）

按十审 §8.2 约束**有意未移植**：`~/.pi/agent/agents` 自动发现、PATH 上
的 `pi` 命令执行、项目 `.pi/agents` 注入、Worker 无限再委派。

Upstream license: MIT — see
`packages/rosclaw-agent/node_modules/@earendil-works/pi-coding-agent/package.json`.

## pi-subagents（十四审 PR-14.4 参考移植）

`pi-packages/tintinweb-pi-subagents`（上游 commit `0e39260`，v0.10.3，MIT）。
`packages/rosclaw-agent/src/workers/tasks-center.ts` 的信息架构（后台任务
卡片、attempt 聚合、完整会话查看、mid-run steering、恢复语义）对照其
`agent-manager.ts`/`ui/agent-widget.ts`/`ui/conversation-viewer.ts` 移植。

按十四审 §2.2 约束**有意未安装**该扩展：Pi 扩展默认拥有宿主进程权限，
ROSClaw 保留自己的 WorkOrder 工具白名单、隔离 workspace 与 rosclawd
安全边界；其宽权限 Agent 工具注册未移植。
