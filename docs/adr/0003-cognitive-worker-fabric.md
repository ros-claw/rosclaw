# ADR-0003：认知 Worker Fabric 与 daemon 硬件 adapter WorkerManager 分离

- 状态：Accepted
- 日期：2026-08-01
- 依据：实施总纲 §9；参考 hermes-agent（delegation 双轨状态机）、zeroclaw（budget 血统继承、estop 分级）、paperclip（多 scope 预算）

## 背景

`rosclaw.daemon.worker_manager` 监管的是硬件 adapter 子进程（ready/heartbeat/degraded，最小权限内操作指定设备）。Native Agent 还需要管理认知 Worker：Native 子任务执行器、确定性工具、Codex/Claude Code 等 Harness、人类、其他机器人。两者权限域、故障语义、schema 完全不同，混用一个 WorkerManager 会把 Agent 的心跳误当硬件 readiness。

## 决策

1. 新建 `rosclaw.agentd.workers.WorkerManager`，管理认知 Worker；与 `rosclaw.daemon.worker_manager` 使用不同 schema（`rosclaw.worker_card.v1` vs `rosclaw.adapter.worker.v1`）、不同日志类别、不同 CLI 命令面。
2. Worker 五类：native / tool / harness / human / robot。任何 Worker 不因能力获得物理授权；其输出默认是建议/工件/patch/验证结果，物理副作用必须回到 Native Agent → Consent → `rosclawd`。
3. 生命周期采用**双轨状态机**（借鉴 hermes-agent `async_delegations` 的运行态×投递态）：
   - WorkOrder：`DRAFT → OFFERED → CLAIMED → RUNNING → SUBMITTED → VERIFYING → ACCEPTED`，旁路 `BLOCKED/FAILED/EXPIRED/CANCELLED`；
   - Lease/heartbeat 独立记录，`SUSPECT → EXPIRED`；`COMPLETED` 仅表示 Worker 提交，验收需 verifier 通过。
4. 招聘分两阶段：确定性硬过滤（能力/schema/health/隔离/side-effect class/预算/许可证/熔断），然后可解释加权排序（记录 feature vector、分数、策略版本）；安全与权限永远硬过滤，不参与加权。
5. 副作用管控：所有 side-effect WorkOrder 必须有 idempotency key；heartbeat 丢失后禁止盲目重发，先 reconcile（查 adapter journal/idempotency record）；旧 lease 结果记 late/stale 不自动接纳；预算沿委派链血统继承（借鉴 zeroclaw ActionBudget），`max_children` 默认 0。
6. 隔离：Worker 默认独立进程（process-stdio JSONL envelope 起步），最小路径/网络/数据 scope；凭据按引用运行时注入，WorkOrder 不夹带 secret。
7. 信任多维建档（capability×task family×body/sim family×版本），安全违规是独立 veto，不可被普通成功抵消；样本不足显示 `UNVERIFIED`。
8. 第三方 Harness 以 Official WorkerPack 分级接入（T0 Discovered / T1 Compatible / T2 Verified / T3 Recommended），锁版本与 digest；不 vendoring 第三方源码进核心 wheel。

## 不采纳

- hermes 同进程线程隔离（认知 Worker 故障不得传染 Agent 主循环，进程级起步）；
- paperclip “agent 不常驻按需唤醒”用于物理相关 Worker；
- 分钟级 stall 阈值用于有副作用的任务。

## 后果

- Phase 5 按 PR-WF-050~055 实施；无 Worker 时 Native Agent 仍可完成核心任务（Worker 是增强，不是依赖）。
