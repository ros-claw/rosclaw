# ROSClaw Native Agent（rosclaw-agentd）

> Maturity: **experimental**（ADR-0000）。这是 ROSClaw 自有的原生 Agent 进程：
> 不依赖 Codex/Claude Code/OpenClaw 即可对话、规划、调用工具并完成 SIMULATION
> 任务。物理执行边界仍唯一属于 `rosclawd`（ADR-0001/0006）。

## 快速开始

```bash
# 1. 持久保存凭据（交互式隐藏输入；目录 0700、文件 0600）
rosclaw agentd credential set --provider kimi-code
# 或 rosclaw agentd credential set --provider kimi-api

# 也可以不持久保存，只给当前进程环境提供凭据
export ROSCLAW_KIMI_API_KEY=sk-kimi-...        # Kimi Coding Plan

# 2. 配置模型（config.yaml 只写 api_key_ref，不写真实密钥）
rosclaw agentd init --provider kimi-code       # 或 kimi-api / openai-compat / local

# 3. 就绪检查（自动加载已保存凭据；四项 probe）
rosclaw agentd doctor                          # READY 或诚实的 MODEL_NOT_READY

# 4. 对话（默认 SIMULATION）
rosclaw chat
rosclaw chat --goal "检查仿真身体状态"

# 5. 本地 HTTP 服务 + Console
rosclaw agentd start --port 8765
#   http://127.0.0.1:8765/console   最小聊天 Console
#   /health /status /probe /missions /missions/{id}/turns
```

`--mode REAL` 只是请求：创建 Mission 前必须存在 hash-valid 的真实身体、
已加载并验签的 Robot Pack、在线 rosclawd 与对应 REAL executor；任一缺失都
必须拒绝并列出缺口（fail closed）。MissionGrant 在具体动作提出后由用户批准，
不会被错误地要求在 Mission 创建前预先存在。

### 绑定真实身体（例如 LIMO）

模型初始化后，在同一个 `~/.rosclaw/config.yaml` 的 `agent` 段绑定 Body：

```yaml
agent:
  enabled: true
  body_id: limo
  default_mode: REAL
  budgets:
    physical_action_count: 3
```

`body_id` 由 Body Registry 解析；旧配置的 `sim_body_id` 仍用于 SIMULATION，
但不会被当作真实 Body。真实身体上下文读取 EffectiveBody 的重算哈希，并从
rosclawd 获取新鲜的控制面 Self 状态；daemon 断开、Body hash 漂移或 Robot
Pack/执行器缺失时均停止推进。REAL Mission 还要求显式的
`physical_action_count > 0`；默认值为 0，不会隐式获得真机动作预算。

## 架构落点

| 模块 | 职责 |
|---|---|
| `rosclaw.contracts.{agent,worker,team}` | 版本化跨进程契约（schema v1 冻结） |
| `rosclaw.agentd.mission` | MissionStore：SQLite WAL + event journal + revision CAS + 预算 + 会话 journal |
| `rosclaw.agentd.context` | ContextCompiler：L0–L8 分层、确定性 hash、fail-closed freshness |
| `rosclaw.agentd.models` | ModelPolicy/Gateway：Kimi K3 等 OpenAI 兼容端点，严格工具 schema，流式聚合 |
| `rosclaw.agentd.decisions` | DecisionValidator：DecisionV1 绑定当前 context revision |
| `rosclaw.agentd.loop` | AgentLoop：显式状态机、工具循环、预算、崩溃恢复、决策块流式过滤 |
| `rosclaw.agentd.tooling` | Tool/Capability Catalog（PR-05）：ToolDescriptorV2、ToolResolver 硬过滤+排序、MCP adapter、证据封装 |
| `rosclaw.agentd.service` | AgentService + FastAPI 本地 API（含 SSE 流式 turn）+ 最小 Console |
| `rosclaw.agentd.usage` | model_usage 持久化计量（每轮一行，聚合由查询计算） |

## 模型层能力（PR-NA-030b，借鉴 picoclaw/zeroclaw/hermes/openharness）

- **SSE 流式**：`OpenAICompatRuntime.invoke_stream` — `stream_options.include_usage`、
  心跳注释、10MB 上限、60s 空闲 watchdog；tool_calls 按 index 增量聚合。
- **错误三级分类**：HTTP status → kind（429→rate_limited、408→timeout、
  401/403→auth_error、5xx→http_error、其他 4xx→invalid_response 不重试）；
  指数退避 + jitter + 尊重 Retry-After（≤30s）。
- **持久化计量**：`model_usage` 表（migration 003）记录每轮
  prompt/completion/reasoning tokens、成本（profile 单价，微单位）、延迟、
  request_id、context 绑定；成本计入 mission `monetary_microunits` 预算。
- **流式 UX**：`rosclaw chat` 边到边输出 + 每轮/累计用量行；Console 经
  `POST /missions/{id}/turns/stream`（SSE）实时渲染；DecisionV1 协议块
  在流中被 `DecisionBlockFilter` 过滤，不打扰用户。
- **会话连续性**：对话追加进 mission journal，重启后 `--mission` 恢复完整历史。

已知未做（诚实清单）：多 key 轮换。

## 持久化 Compaction（PR-07 + 批次 A）

- **append-only journal**：canonical 对话事件永不删除；每条消息在
  `append_conversation` 时获得稳定 `entry_id` + 单调 `seq`；
  内部 journal 键（entry_id/seq/atomic_group）经 gateway sanitizer 净化，
  永不越过 provider wire。
- **持久化压缩**：Pi 算法切点（keepRecentTokens 倒序 + 边界对齐，不拆
  tool_call/tool_result 对、不拆 `atomic_group` 物理原子组）；
  `CompactionEntryV1` 记录 covered_entry_ids、covered_span_hash、
  supersedes、prompt_version、provider/model、protected_groups；
  手动 `/compact`、阈值自动压缩、overflow reactive 压缩全部走同一
  持久化引擎（不原地改写内存列表，`_persisted_count` 始终对齐）。
- **视图投影**：模型看到的是 journal 投影（summary marker + kept），
  重启后从 journal + 最新 compaction 恢复相同 view；物理事实不依赖
  摘要——每轮从权威存储重新编译。

## Worker Fabric（PR-WF-050/051/053，experimental）

认知 Worker 是受管理的承包人，不是第二套主人（ADR-0003）。与
`rosclaw.daemon.worker_manager`（硬件 adapter 子进程）完全分离。

- **WorkerCardV1**：声明而非事实；注册校验 adapter/许可证/能力 schema/
  数据范围/hard-forbidden scopes（`daemon_private_ledger`、
  `physical_permits`、`raw_secrets`、`direct_hardware`）。
- **WorkOrder 双轨生命周期**：`DRAFT→OFFERED→CLAIMED→RUNNING→SUBMITTED
  →VERIFYING→ACCEPTED`；lease 超时 SUSPECT→EXPIRED；副作用任务先
  reconcile 再谈重派（禁止盲目双发）；旧 lease 结果记 late 不接纳。
- **调度两阶段**：硬过滤（能力/状态/隔离/副作用类/并发/许可证/熔断）
  不过即拒，安全永不进入加权；评分 0.30C+0.20R+0.15A+0.10L+0.10K+
  0.10P+0.05D，feature vector 与策略版本全部入 journal。
- **验证**：identity 绑定、期望工件、secret 扫描、claim-证据绑定、
  用量合理性、伪造成功（COMPLETED 无工件）——任一不过则 FAILED。
- **native-basic（T3）**：同模型隔离子任务（独立 conversation、预算
  envelope、P0 禁止再委派）；worker 输出永远是 proposal。
- **CLI**：`rosclaw worker list|catalog|inspect|enable|disable|probe`。
- **外部 WorkerPack（PR-WF-054）**：claude-code（T1，repo/test/docs 分析，
  `--disallowedTools *` 零工具）与 codex-cli（T0 缺二进制时给安装指导）；
  Official WorkerPack 模式——不 vendoring、版本锁、env 白名单透传
  （API key 只从宿主环境走，不进 WorkOrder）。

K4 验收（live）：委派闭环 ACCEPTED 且全归因（offered→claimed→started→
submitted→accepted）；密钥注入拒绝；work order 中 0 secret。

## Operator Broker 与授权（PR-OP-060/061/062，experimental）

取消"手动 arm daemon"，不等于取消授权（ADR-0006）。EXACT_ACTION 流程：

```text
Agent REQUEST_APPROVAL → Broker 生成 ActionDisplay 卡片 → mission 进入
WAIT_APPROVAL → 用户 /approve <id>（chat）或 Console /approvals 决定 →
Broker 签发 MissionGrant（public scope + public_hash；HMAC 私签只存
broker 侧，永不进入模型上下文）→ Agent REQUEST_ACTION 引用 grant_id →
Broker.verify 独立核验（principal/body hash/mode/risk/action_intent）→
EXACT_ACTION 单次消费，重放即拒
```

REAL 模式还有独立的物理边界确认：AgentD 校验 MissionGrant 后只调用
`operator.proposal.create`。rosclawd 返回 public proposal（无 challenge、无
permit、未派发），再由受信 Operator 进程审阅精确 ActionEnvelope 并决定。
AgentD 不调用 proposal decision RPC，也不会因为拿到 MissionGrant 而自批。
最终链路为：MissionGrant verify → daemon proposal → Operator decision → daemon
Permit → executor → terminal Receipt。SHADOW/SIMULATION 仍直接返回各自证据域的
回执，不能冒充 REAL 物理证据。

攻击回归（全部拒绝并给出 reason_code）：unknown/revoked/expired/
principal_mismatch/body_hash_changed/mode_mismatch/risk_above_ceiling/
forged_grant/grant_consumed。CLI：`/approvals`、`/approve`、`/deny`；
HTTP：`/approvals/pending`、`/approvals/{id}/decide`、`/grants`、
`/grants/{id}/revoke`。

K5 验收（live）：真实模型完成 请求授权→批准→授权验证→单次消费 闭环；
SIMULATION 下诚实声明"无物理派发，非执行回执"。

## Team Fabric（PR-TF-070/071/072/073 精简版，experimental）

多机器人不是开多个聊天窗口（ADR-0004）。每台机器人是自治安全单元，
团队分配是契约建议，本地 Native Agent + rosclawd 保留拒绝权。

- **Membership**：CANDIDATE→JOINING→READY→SUSPECT→LOST/LEFT；加入/离开
  提交 epoch；TTL sweep（超时直接 LOST）；epoch 使旧 award/lease 失效。
- **RoleLeaseV1**：DB 级 CAS——每个 (team, epoch, conflict_key) 至多一个
  ACTIVE lease；旧 epoch/非 READY holder 拒绝；过期/失联自动释放；
  冲突执行保守 contest 策略。
- **Contract Net allocator**（contract_net.v1）：announce → 本地可行性
  硬门槛（capability_fit=1、deadline、risk）→ 确定性特征评分
  （eta/energy/capability/risk/reliability/load/comms）→ award；
  特征向量全部入 journal，不用模型投票。
- **World model**：latest_valid merge、tombstone、时钟偏差超容忍拒绝融合、
  epoch 不匹配拒绝；freshness 查询是行动前提。
- **降级矩阵**（总纲 §10.8 已实现并测试）：成员失联→角色过期+任务
  重新公告（不原地重复）；Coordinator 失联→不产新任务；epoch 不一致→
  拒绝混合；时钟偏差→拒绝融合。
- **Transport**：`local_sim`（延迟/丢包/分区故障注入，seed 确定性）；
  ROS 2/DDS、Zenoh adapter 为后续 PR。
- agentd 集成：config `team.enabled` 开启本地协调器；TEAM_COORDINATE →
  team_task_claim 走真实分配；TEAM_COORDINATE 无 operation 被验证器
  强制修复（missing_operation）。

K6 验收：双机协作全生命周期（角色→共享世界→announce/bid/award/accept/
complete+证据）+ 故障矩阵（失联/分区/Coordinator down/epoch 混乱）+
live 模型协调。3v3 联赛基准（T-SIM-2/3）属 PR-TF-075 后续范围。

## 评测与学习（PR-EV-080/081，experimental）

- **Benchmark harness**（`rosclaw eval run`）：scenario × seed × 基线组
  （A=native-only / B=native+workers），指标含 success rate、
  unsupported-claim rate（目标 0）、tokens/cost、delegation accept rate；
  产物落盘（每 run 一个 JSON + aggregate.json），同 seed 确定性。
- **Learning pipeline**（`rosclaw learning`）：Practice 证据门——只有
  measured/verified_receipt/curated 事实形成 Memory/Know/How/Auto 候选，
  unverified/inferred 显式拒绝并记录；Darwin 晋升门 = 评测引用 + 人类
  principal，任何代码路径不能自动晋升。

## 命令面（批次 E）

- 全部命令经 CommandService 注册表（spec + availability + disabled_reason
  + 幂等）；未知命令不发给模型。
- 执行/安全可见性：`/workers /worker inspect|enable|disable|probe`
  （disable 有未终态订单时拒绝并提示 drain，写审计）、`/grants`（仅
  public scope）、`/revoke`、`/body`、`/doctor`、`/mode`（禁止原地升级）、
  `/context layers|compactions|refresh`、`/session`、`/new`、
  `/retry`（无副作用才可重放）、`/failover`、`/thinking`、
  `/scoped-models`。
- Mission 资产：`/export`（.rcmission zip：manifest/conversation/events/
  compactions/public-receipts/checksums，构造性脱敏）、`/import`
  （magic/checksum/路径穿越/zip bomb/secret scan 全校验；只读归档导入，
  不恢复任何授权效力）、`/share`（诚实指向 /export，不默认上传）。
- 配置：`/settings`（白名单非安全键、tmp+fsync+rename 原子写、文件锁、
  审计事件；安全域键永远拒绝）、`/reload`（prompts/workers/models 可
  reload；policy/robot_pack/body/permits 等安全域永远拒绝）。
- `/tree /fork` 已可用（批次 F 第一阶段）；`/clone`（第二阶段）与
  `/copy`（TUI 本地 clipboard）deferred——不伪造存在。

## ACP adapter（批次 G）

- `rosclaw acp serve`：官方 `agent-client-protocol` SDK（>=0.12.0）的
  stdio JSON-RPC server——Zed 等 ACP 客户端接入同一个 Native Agent。
- 映射：session → Mission；prompt → turn；AgentEventV2 → session
  update（text delta / tool / worker / approval 卡片 / receipt / usage）。
- 边界：ACP client 是 UI client——AgentLoop 仍在 agentd 运行；ACP
  generic permission 不等于 rosclawd Permit；approval 只呈现卡片与 id，
  客户端决定不构成物理授权（不自动批准、不退化为聊天口令）。

## ReasoningBranch（批次 F 第一阶段）

- 双时间线：推理分支树可 fork；物理事实线只追加、永不回滚、不被旧
  分支遮蔽（/tree 同时显示两条线）。
- /fork 永远创建新 SIMULATION mission：不复制 grant/approval/Permit/
  worker lease；旧 DecisionV1 对新 mission 无效（新 context，revision
  从 0 开始）；首轮编译从权威存储注入最新 Body/Self；fork 点之前的
  推理历史以 untrusted（source=fork:…）形态带入。
- 物理动作进行中（未终态 WorkOrder）fork 被拒（fail closed）。
- 同 Mission 分支切换与 /clone 属第二阶段，待不变量测试扩展后开放。

## LIMO 完整闭环与发布打包（PR-12）

- `rosclaw.limo.sim_mcp`：LIMO 仿真 MCP（观测 get_pose/health 为
  readOnlyHint → OBSERVE；play_tone/set_initial_pose 为 PHYSICAL_ACTION）。
- `SimActionChannel`（SIM 物理权威）：只在 EXACT_ACTION grant 验证消费
  后执行；产出 SIMULATED receipt（evidence_domain=simulation、
  usable_for_real_execution=false；无声学观测时诚实声明只证明驱动
  执行）。观测与执行共享 `PersistentMcpClient` 持久会话（有状态身体
  必须同进程）。
- 验收（§18 C1–C10）：`bench/limo_acceptance.run_acceptance` ——
  观测 → 授权卡 → operator.sock 批准（peer identity + display hash）→
  SIM 执行 → receipt → 位姿验证 → practice candidate；SHADOW/REAL 无
  daemon/硬件时诚实拒绝，绝不用 SIM 证据冒充。
- K9 live：真实 Kimi K3 自行决策完成同一闭环（grant.consumed、
  receipt.received、单次消费全部验证）。
- 打包：`scripts/build_release.sh` → `dist/rosclaw-<ver>-linux-arm64.
  tar.gz`（src + 已构建 packages + lockfiles + third_party 声明 +
  manifest hash）；`scripts/release/install_release.sh`（venv + npm ci
  重建 + 原子切换 current/previous，Node 缺失诚实降级）；
  `rollback.sh`（manifest 校验后回切，失败版本保留排查）。
- wire 约束：OpenAI 函数名不允许点号——工具 id 在 wire 上映射为
  `__`（catalog 注册拒绝含 `__` 的原生 id 保证单射）；内部协议工具的
  成功提交也必须回执 tool result（Kimi 拒绝悬空 tool_call）。

## 授权剖面（Operator Decision Protocol v1，二次复核 R1/R2/R3，2026-08-04）

- **协议对象**（contracts/operator/decision.py）：`DecisionChallengeV1`
  （daemon 签发的一次性挑战，challenge_nonce 与 proposal 同源）→
  `OperatorDecisionProofV1`（operatord Ed25519 签名，覆盖 challenge
  全字段 + decision + decided_at + human_confirmation_method）→
  `DecisionReceiptV1`（daemon 自己的 Ed25519 身份签名，绑定
  proposal/agent_request/mission/mode/capability/args/display_hash/
  decision/expires/daemon_key_id）。
- **rosclaw-operatord**：唯一持有 operator Ed25519 私钥的进程
  （0600、O_NOFOLLOW/O_EXCL、双 fsync、损坏按时间戳 quarantine）。
  REAL/SHADOW 决定：取 daemon challenge（nonce 同源）→ 校验请求方
  前台进程组（/proc tpgid）→ /dev/tty 显示不可变卡片读取显式 Y/N
  （默认/超时/EOF 一律 deny）→ 签 proof。
  `rosclaw operatord enroll|register-daemon|list-daemon|revoke-daemon|start|status`。
- **rosclawd**：持久化 enrollment registry（0600、原子写、fsync；
  空表全拒、无首调抢注窗口；register/revoke/list 仅 daemon 管理员；
  已焚毁 nonce 持久化防跨重启重放）。`proposal.decide` **没有
  daemon-UID 直通**——唯一凭证是有效 proof 且调用方 UID ==
  enrollment 登记的 operator UID；验证成功后经不暴露 socket 的
  内部 `_arm_after_operator_decision`/`_issue_permit_after_operator_decision`
  完成 arm/permit（P0-4）。`daemon.identity` 公开签名公钥。
- **agentd**（R3/P0-6）：daemon 卡只接受 daemon 签名、`decision=ACCEPT`
  且所有字段与本地卡片精确相等、未过期、未重放（sqlite UNIQUE）的
  DecisionReceiptV1；DECLINE 只关闭请求绝不铸 grant。SIM 卡为
  operatord Ed25519 签名 + TOFU 钉住公钥（明确 DEV_SIM_ONLY）。
  HTTP decide/revoke 默认 403；同 UID 一体运行标记 `DEV_SIM_ONLY`。
- **SHADOW 全链路**（tests/shadow/test_limo_shadow_chain.py）：
  enroll → register → proposal → challenge.get → sign → decide →
  daemon receipt（公钥可验）→ Permit/Lease → LIMO SHADOW executor →
  SHADOW receipt（actuated=false、拟执行 ROS 命令可审计）。
- **human presence**（tests/operatord/test_human_presence.py，T2）：
  真实 PTY 驱动——显式 Y 才批准，N/乱输入/EOF/超时/无 tty 全 deny；
  非前台进程组请求直接拒绝。

## 证据等级与证据包（审计 §1.2/§8）

- 等级命名（`bench/evidence_levels.py`）：E0 SPECIFIED …
  E7 OPERATIONALLY_QUALIFIED；任何"全链路通过"必须标注最高等级。
  当前：组件 E1/E2，SIM 主路径 E3，SHADOW 许可链 E4（部分）。
- 证据包（`bench/evidence_pack.py`）：`acceptance/<run_id>/` 含
  run_manifest（commit/dirty/level/test_ids/operator/secret_scan_clean）、
  environment、commands、events、mission_snapshot、approvals/permits/
  receipts(public)、metrics、secret_scan、artifact_hashes、
  operator_observer——**没有证据包就没有通过**。
  `run_acceptance(evidence_root=...)` 直接产出。

## Operator 安全通道（PR-11）

- `operator.sock`（0600，JSONL）：与模型可见的 Agent API 物理分面——
  `approvals.list / approvals.decide / grants.revoke / estop`。
- **peer identity**：principal 只从 SO_PEERCRED UID 派生
  （`user:local:<uid>`）；请求体里的 principal 字段永远被忽略（伪造
  root 的攻击回归测试锁定）。
- **display hash**：approve/deny 必须携带卡片 display_hash（title/
  summary/risk/parameters/body_hash/request_id 的 sha256 指纹），
  不匹配即拒（TOCTOU 换卡防护）；EXACT_ACTION 单次性由 Broker 保证。
- **CSRF/Origin**：HTTP 变更请求携带外部 Origin → 403；
  `ROSCLAW_CONSOLE_TOKEN` 设置时强制 X-Rosclaw-Token pairing。
- **/estop**：TUI 本地命令经 operator.sock 直达 rosclawd（不经过模型）；
  无 daemon 时诚实报不可用，绝不假装已停。
- operator 方法永不出现在模型工具目录（架构级断言）。

## rosclaw-modeld（批次 D）

- `packages/rosclaw-modeld`：精确锁定 `@earendil-works/pi-ai@0.83.0`，
  Unix socket（目录 0700、socket 0600）+ 启动时随机 bearer token（只经
  子进程环境传递）。modeld 不是 Agent：不碰 MissionStore/DecisionV1/
  工具/rosclawd（架构测试锁定，providers/all 禁止）。
- API：`GET /v1/providers /v1/models /v1/auth`，
  `POST /v1/auth/{provider}/login|logout`（OAuth 诚实 501，本批未做），
  `POST /v1/probe /v1/stream`（SSE: text.delta/tool_call/usage/done/error）。
- `ModeldGateway`（Python ModelGateway 协议）：AgentLoop 不再直接接触
  OpenAI-compatible 协议细节；崩溃/不可达诚实报错（modeld_crashed），
  绝不伪造成功。`models.backend: modeld` 切换；legacy backend 保留。
- Provider：moonshot（kimi-k3）、kimi-code（k3/k3-256k，OpenAI 兼容面）、
  ollama（本地）；凭据 env 引用或 modeld credential store（0600、
  sha256 指纹展示，secret 不出 API 响应）。
- 命令：`/providers /model /login /logout`（MODEL_CONTROL，经控制 API，
  不发给模型）；`rosclaw agent backend --set modeld` 迁移现有 Kimi
  配置（profile 与 env 凭据引用不变）。
- 验收：K7（真实 Kimi K3 经 modeld 链路 probe/chat/usage/strict tool
  call）、K8（backend=modeld 的 AgentService 完整 turn，usage 计量）——
  全部为真实路径，无 mock。

## rosclaw-tui（批次 C）

- `packages/rosclaw-tui`：精确锁定 `@earendil-works/pi-tui@0.83.0`
  （MIT，third_party/pi 有 NOTICE/LICENSE；lockfile 已提交），
  Node >= 22.19。薄 Presenter + 纯函数 reducer + 命令路由——不复制
  Pi interactive-mode.ts。
- 数据流：`GET snapshot`（权威对齐）→ SSE `Last-Event-ID` 增量 →
  reducer → 渲染效果；sequence 缺口自动重拉快照；event_id 去重；
  agent.settled 停止 spinner。
- 命令：`/help /hotkeys /quit /clear-screen /approve /deny /missions`
  本地处理；`/compact /cancel /rename /archive /status /tools` 经
  Batch B 控制 API；未知命令提示而不是发给模型。
- 卡片：tool / worker / approval / receipt 结构化渲染；状态行常显
  mode（REAL 红底）/body/mission/state/待批准数/阶段。
- `rosclaw chat` 默认启动 TUI（进程内 uvicorn + exec node）；
  Node/资源缺失时诚实回退 `rosclaw chat --basic`（input() 诊断模式）。
- 边界由架构测试锁定：TUI 不得含模型客户端，pi 包必须精确锁定
  （tests/architecture/test_pi_dependency_boundary.py）。

## UI 控制面（批次 B：命令/事件/快照/交互）

- **命令是控制协议，不是聊天文本**：`/compact /cancel /rename /archive
  /status /tools` 经 CommandService 注册表路由到控制 API，永不进入模型
  上下文；`/approve /estop` 属 SAFETY_CONTROL，走专用端点，不在通用
  注册表。`GET /v1/capabilities` 返回含 disabled_reason 的命令表；
  `POST /v1/missions/{id}/commands` 幂等执行（idempotency_key）。
- **事件全集（AgentEventV2）**：agent.started/settled/failed（settled
  是 TUI 停 spinner 的唯一可靠信号，失败路径也保证发出）、turn.*、
  message.*、model.selected/request.ended、context.usage、tool.*、
  worker 全生命周期（claimed→started→submitted→verifying→accepted/
  failed/expired）、approval.requested/decided、grant.revoked/consumed、
  mission.renamed/archived。per-mission sequence 单调无缺口。
- **断线恢复**：SSE 支持标准 `Last-Event-ID` 头；sequence 出现缺口时
  客户端拉 `GET /v1/missions/{id}/snapshot`（MissionSnapshotV1）重新
  对齐——快照只含可公开状态（grant 仅 public 字段，无 secret/Permit）。
- **通用交互**：InteractionRequestV1（select/confirm/input/editor）经
  `POST /v1/interactions/{id}/respond` 响应；masked 值不落 journal；
  generic confirm 永远不能伪造 approval。
- Mission 展示名/归档存 `mission_meta`（migration 011），不改
  MissionSessionV1 契约与状态机；归档 mission 拒绝新 turn。

## Tool/Capability Catalog（PR-05）

- **ToolDescriptorV2**（`rosclaw.tool_descriptor.v2`）在契约层强制：
  `PHYSICAL_ACTION` 永远 `model_callable=false` 且必须声明
  `requires_exact_action_grant=true`；OBSERVE 不得有副作用。违规构造直接
  ValidationError（fail closed）。
- **ToolResolver**：先确定性硬过滤（body 兼容 / mode / capability 在线 /
  SelfSnapshot 新鲜 / permission / policy / 预算 / verifier / quarantine /
  model_callable，全部可解释），再做相关性排序（语义、延迟、可靠性、
  成本、新鲜度、证据等级）。安全条件永不进入模型评分；每轮注入 ≤12 个工具。
- **MCP adapter**：`mcp_servers:` 配置的外部 MCP server（如 limo-ros-mcp）
  经 stdio 发现工具；分类 fail-closed——`readOnlyHint` 且无动作动词 →
  OBSERVE；动作动词 / destructive / 无注解歧义 → PHYSICAL_ACTION。
  发现失败 → 整个 source 隔离（quarantine），聊天不中断、观测不伪造。
- **证据封装**：每次观测产出 EvidenceEnvelope（timestamp/body/source/
  evidence_class/freshness/artifact_ref），内容以 `<untrusted_input>` 包裹；
  大输出落盘 artifact store（content-addressed），模型只见 ref+摘要。
- 配置示例：

```yaml
mcp_servers:
  - name: limo-ros-mcp
    command: /usr/local/bin/limo-ros-mcp
    args: ["--profile", "sim"]
    env_refs: ["LIMO_MCP_TOKEN"]   # 只引用环境变量名，值不入文件
    supported_modes: ["SIMULATION", "SHADOW"]
    required_body_types: ["agilex-limo"]
```

## 审计修复记录（对照总纲逐项审计后）

1. EXACT_ACTION verify 必须声明动作意图（broker 从批准卡片重算，不采信
   模型自报）；2. broker 签名密钥随机持久化（不可由公开 policy_hash
   推导）；3. 成员失联只重公告无副作用任务（team_tasks.side_effect_class）；
4. world merge 严格 latest-valid（后到旧观测忽略并告警）；5. 预算超限
   进入 WAIT_INPUT 且不执行决策（新增 PLAN/VALIDATE→WAIT_INPUT 边）。
归因链补齐：decisions/context_manifests（含 prompt hash）/work_results
落库、operator_events、transition trace_id。


## Kimi K3 两个产品面（2026-08-01 实测）

| 产品 | endpoint | 模型 | Key |
|---|---|---|---|
| Moonshot 开放平台 | `https://api.moonshot.cn/v1` | `kimi-k3` | `MOONSHOT_API_KEY` |
| Kimi Code（Coding Plan） | `https://api.kimi.com/coding/v1` | `k3`（1M ctx）/ `k3-256k` | `ROSCLAW_KIMI_API_KEY`（sk-kimi-*） |

两者 OpenAI 兼容、支持 strict tool call 与 `reasoning_effort`（low/high/max）。
Key/额度不互通，firstboot 必须按 Key 类型匹配 endpoint。

## 测试

```bash
pytest tests/agentd tests/contracts tests/architecture -q          # 单元/契约/不变量
ROSCLAW_KIMI_API_KEY=... pytest tests/agentd/test_kimi_live.py -m integration  # K0–K3 实网验收
```

Live 测试无 fixture 替代：API 不可达即失败，不伪造成功；密钥只走环境变量。
