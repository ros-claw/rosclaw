# ADR-0008：Pi 依赖边界（pi-tui/pi-ai 可 import，Pi Agent 生态禁止）

- 状态：Accepted
- 日期：2026-08-03
- 依据：《ROSClaw Native Agent + Pi TUI/Provider 完整实施大纲》§1.3、
  `docs/research/native-agent-ui-provider-reference-audit.md`

## 背景

TUI 与 Model Provider 是成熟生态的主战场。自研终端 UI 与 Provider SDK
重复造轮子且容易在安全边界上犯错；全量引入外部 Agent 生态则会稀释
"谁是 Agent"的根本定位（ADR-0001）。

## 决策

1. **生产 npm 依赖只允许并精确锁定**：
   `@earendil-works/pi-tui@0.83.0` 与 `@earendil-works/pi-ai@0.83.0`
   （不使用 `^`）。
2. **禁止进入运行时的包**：`@earendil-works/pi-agent-core`、
   `@earendil-works/pi-coding-agent`、Hermes `AIAgent`、Codex Agent、
   OpenCode Agent（由架构测试 `tests/architecture/test_pi_dependency_boundary.py`
   强制扫描 npm dependency tree 与 Python imports）。
3. 角色边界（与大纲 §1.1 一致，作为后续所有 PR 的冻结前提）：
   - `rosclaw-agentd` = 唯一 Native Agent（Mission/TaskGraph/授权申请/验证/学习）；
   - `rosclaw-modeld`（pi-ai）= 单次模型推理（文本、tool calls、usage），
     不执行工具、不接触 ROS/`/dev`/串口/CAN/GPIO/Robot SDK；
   - `rosclaw-tui`（pi-tui）= /v2 事件客户端，不直接请求模型，不直接向
     执行器发动作；
   - External Worker = 受限承包人，不掌控 Mission 和真机，不接触
     rosclawd 私有 socket/Permit/Operator 私钥；
   - `rosclawd` = 唯一物理权威。
4. 借鉴而不引入：Pi compaction 算法、slash-command registry、Codex
   popup/App-Server 协议、OpenCode server 分层、Hermes 事件序列——吸收
   设计，不 import 其实现。
5. 架构测试同时禁止：模型直接调用批准授权接口、MCP action tool 当普通
   工具执行、Provider 故障自动切换 External Worker 接管。

## 后果

- PR-08/09（rosclaw-tui、rosclaw-modeld）的 npm 工作区以本 ADR 为依赖
  边界基线；`package.json` 的锁版由架构测试持续校验。
- 任何新增外部依赖需要新 ADR。
