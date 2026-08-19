# ADR-0012：ROSClaw owns Native Runtime; Pi is default Harness Backend

- 状态：Accepted（2026-08-19，PR-H0）
- 基线：main=`4553ffa`
- 依据：`ROSClaw_Native_Harness_重构实施总纲_v2_2026-08-19.md`
- Supersedes：ADR-0011 中"Native Agent 只做治理、工作经 Worker 链执行"的部分

## 决定

1. **ROSClaw Native Runtime 持有任务/具身/安全/证据/验收语义**；Pi 是当前唯一默认的进程内 Harness Backend（模型访问、Agent Loop、会话、工具循环、TUI），通过很薄的 `NativeHarnessBackend` SPI 隔离——不为多引擎生态，只为依赖边界。
2. **Native Agent 自己干活**：主会话直接拥有策略包装的工作工具（read/grep/find/ls/edit/write/bash + ROSClaw 工具），普通任务不开第二个 Pi Session。
3. **Worker 退出默认链**：`TaskControlPlane → PiTaskRunner → WorkerManager → 第二个 Pi Session` 不再是默认执行路径；Worker 仅用于显式委派（H10 前 `worker.enabled=false`）。
4. **一个用户目标 = 一个 root task = 一个 workspace = 一个 primary Harness Session**；纠错/追问/恢复只加 revision/attempt。
5. **终态由 Verifier 决定**；模型自由文本不能宣布成功。
6. **模型/凭据单轨**：ROSClaw ModelService 是唯一对外契约（Pi-backed adapter）；Python model gateway/双读/环境变量互写在 H7 删除。
7. **品牌封装**：UI 不出现 Pi/engine/extension 名称。

## 为什么

历史链 `用户 → Pi 主会话（无工作工具）→ TaskControlPlane → PiTaskRunner → WorkOrder → 第二个 Pi Session → completion watcher` 制造了两个 Agent、两个 Session、多套状态、多个 workspace、模型双轨、幽灵执行与假成功。十六审修好了契约层，但"主 Agent 没有手"这个根因只能靠本 ADR 的结构变更解决。

## 后果

- 新增：`harness/port.ts`（SPI）、`harness/pi/`（唯一实现）、`native/`（Native Runtime 装配）、`task_kernel/`（无模型持久内核）。
- 删除（替代路径过 Gate 后）：`pi_task_runner.py`、`agentd/workers/` 默认链、`contracts/worker/` 默认依赖、completion watcher、模型面 `task_submit`。
- 不变：rosclawd 是 REAL 唯一权威；E-Stop 不依赖 LLM；REAL 门禁保持关闭。
