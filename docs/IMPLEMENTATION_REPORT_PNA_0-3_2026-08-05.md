# ROSClaw Pi 重构 PNA-0/1/2/3 实施与全链路深测报告

- 日期：2026-08-05
- 基线：`rosclaw_native_agent重构.md`（Pi-backed Native Agent 路线）
- 交付：PR #231（PNA-0/1）、#232（PNA-2/3）；main = `f6257e9`
- 同日合入独立测试方 PR：#221、#227、#228、#230（逐个审查 + CI 绿）
- **REAL 门禁：不变（关闭）；默认 engine：legacy（不变）。**

## 1. 交付内容

| PR | 内容 |
|---|---|
| #231 | `docs/audit/PI_CODING_AGENT_EMBEDDING_AUDIT.md`（上游 588915ec 精读，file:line 证据）；`packages/rosclaw-agent`（createAgentSessionRuntime + InteractiveMode 公开 SDK、ROSClaw 品牌 header/title/working、noTools:"all"、项目资源全关、`!` bash 功能级关闭、rosclaw_status 只读工具、`rosclaw chat --engine pi`）；pi-bridge UDS + SessionBinding（一 session 一 mission、单 writer lease 过期回收、migration 014）；架构边界测试改写（harness SDK 仅 rosclaw-agent 精确锁 0.83.0） |
| #232 | `native_agent_v2.md`；EmbodiedContextEnvelopeV1（TTL+hash，每次现取现算）；before_agent_start 每轮注入（stale 注入禁止动作警示）；PiToolRequestV1/ResultV1 + pi.tools.execute 全验证链（binding/mission/lease/allowlist/OBSERVE-only/idempotency/DecisionV1 镜像）；bridge 工具 observe/verify/memory_query/fail_safe；未开放工具诚实 TOOL_DEFERRED |

## 2. 冲突/逻辑复查发现（全部修复后合入）

1. **`rosclaw chat --engine pi` 不起 agentd 内核**——pi-bridge 永远不可达。
   修复：先起 AgentService + lifespan（pi-bridge/operator socket/token 文件），
   传 --mission，退出关闭。
2. **同 mission 双 ACTIVE binding**（两次启动各绑各的）——修复：新 session
   绑定时旧 ACTIVE 降 DETACHED（单认知写者，规格 §12）；lease 退出即释放
   （落库验证）。
3. **架构边界扫描误伤** dist/ 内的 package.json 副本——排除 dist。
4. 旧"全禁 pi-coding-agent"边界测试与新路线直接冲突——按重构规格 §1
   改写为"harness SDK 仅 rosclaw-agent + 精确锁版"。

## 3. 全链路闭环深测矩阵（main=f6257e9，本机实测）

| 验证 | 结果 |
|---|---|
| 全量回归 | **6244 passed**（8 个环境基线失败：firstboot×4+lerobot×4，origin/main 相同；2 个负载抖动项单独重跑全过：release build structure、modeld socket isolation） |
| K0–K9 live（真实 Kimi K3） | 10/11；K7 套件内空回复（模型负载抖动），单独 15s 通过——非代码回归 |
| Node 套件 | TUI 27/27、modeld 18/18、rosclaw-agent 7/7 |
| Pi 端到端冒烟 | agentd 内核（pi-bridge+token+operator.sock）→ bind+lease → 上下文注入 → 真实 K3 回合 `ok`；lease 退出释放、binding 落库 |
| PNA-1/2/3 专项 | binding/lease 4/4、tool 验证链 6/6、transcript 6/6 |
| 发布/证据 | packaging 14/14（含 T5 攻击矩阵）；验收 12/12 + 签名证据包 `rosclaw evidence verify` VERIFIED（E3_SIM_VERIFIED） |
| 安全套件 | shadow FTC-100 5/5、operatord PTY 8/8、operator 合约 34/34、architecture 21/22（1 skip） |

## 4. 诚实 deferred（规格后续批次）

- PNA-4 Worker 体验（/delegate、native pi worker、递归上限）；
- PNA-5 ApprovalCard 组件接 operatord（当前行级 y/n + /approve）；
- PNA-6 session 生命周期映射（fork 强制 SIM、authority 不复制、tree 不回滚物理事实）；
- PNA-7 Provider 收敛（Pi ModelRuntime 默认 + 凭据后端 + modeld 退役路径）；
- PNA-8 event mirror 不双写全文；PNA-9 Resource Security profiles；
- PNA-10 发布门禁（clean build/bundled Node/build-info/installed PTY）；
- PNA-11 默认切换（须全 Gate 通过）。
- 已知：内建 slash 命令不可 veto（薄 fork 点，PNA-9 前 ROBOT profile
  不启用）；Pi Session 绑定目前由 --mission 传入（/resume 自动绑定在 PNA-6）。

## 5. 证据

- 证据包：`/tmp/evidence-final2/acceptance/run_30e38cec384d44a18fa51eb9/`（signed）
- 上游审计：`docs/audit/PI_CODING_AGENT_EMBEDDING_AUDIT.md`
- CI：#231/#232 全绿（含 node-agent-unit、cross-uid-operator-e2e、evidence-pack-verify）
