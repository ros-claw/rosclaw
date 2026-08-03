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
