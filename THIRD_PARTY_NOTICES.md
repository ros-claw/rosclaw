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
