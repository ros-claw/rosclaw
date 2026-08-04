# ROSClaw Native Agent 实施报告

**日期**：2026-08-04　**基线**：main @ d65dc5c + #209　**实施方**：Claude Code（Fable 5，k3[1m]）

---

## 1. 实施范围

本报告覆盖两份实施规格在 ros-claw/rosclaw 仓库的完整落地：

1. **《ROSClaw_Native_Agent_Pi_TUI_Provider完整实施大纲》**（PR-01 … PR-12）
2. **《rosclaw-native-agent-reuse-supplement》**（复用补充文档，批次 A–G）

前置已完成的《Native Agent Worker Team 实施总纲 v1.0》（Phase 0–8、Worker Fabric、Operator Broker、Team Fabric，PR #187–#193）不在本报告详述，但其成果（Worker/Team/授权体系）被本阶段全部复用并通过兼容性审查。

**指导原则（贯穿全部实施）**：
- 复用成熟轮子，不重造：pi-tui、pi-ai（均精确锁定 0.83.0，MIT）、官方 mcp SDK、官方 agent-client-protocol SDK；
- ROSClaw 拥有 Agent 控制权、命令语义、Mission 事实、工具权限、Worker 边界与物理安全语义；
- rosclawd 是唯一物理执行权威；agentd 无特权；E-Stop 永不依赖 LLM/云；
- API key 只在环境变量（`env:VAR` 引用），绝不进文件/提交/日志/事件/WorkOrder；
- 提交≠完成；SIM 证据永不冒充 SHADOW/REAL；不允许用 mock 成功替代真实路径。

## 2. 交付清单（14 个 PR，全部 CI 绿并 squash 合并）

| PR | 内容 | 关键验证 |
|---|---|---|
| #196 | AgentEventV2 事件流 + /v2 API、MissionRunner（唤醒条件）、决策意图、submit_decision 协议工具、持久化 Compaction | K0–K6 live |
| #197 | PR-05 Tool/Capability Catalog：ToolDescriptorV2（契约级 fail-closed 不变量）、ToolResolver 硬过滤+排序（≤12 注入）、MCP adapter（fail-closed 分类）、证据封装（untrusted+artifact 落盘） | LIMO MCP 观测可用、action 永不可直接执行（真实 stdio server 端到端） |
| #198 | 批次 A：append-only journal、entry_id/seq 稳定身份、reactive overflow 走持久压缩（修 §3.3 persisted_count 隐患）、wire 净化、CompactionEntry 审计字段、atomic_group 防拆 | 重启 view 一致；压缩后新消息必落盘 |
| #199 | 批次 B：CommandSpec/CommandRequest/CommandResult、MissionSnapshotV1、InteractionRequestV1；agent.settled 全路径保证；Last-Event-ID；worker 全生命周期事件；mission_meta | sequence 无缺口；secret 扫描 |
| #200 | 批次 C：rosclaw-tui（pi-tui@0.83.0）——Editor/IME/CJK、transcript、SSE 增量+快照对齐、命令路由（不发给模型）、卡片、状态行；rosclaw chat 默认 TUI，--basic 降级 | 13 node 测试；架构边界测试（禁 pi agent 运行时、精确锁定） |
| #201 | 批次 D：rosclaw-modeld（pi-ai@0.83.0）——UDS 0700/0600+bearer、providers/auth/probe/stream、凭据库指纹展示；Python ModeldGateway（AgentLoop 脱离 OpenAI 协议细节）；/providers /model /login /logout；backend 迁移 CLI | **K7/K8 真实 Kimi K3 经 modeld 全链路**（probe/chat/usage/strict tool/完整 turn） |
| #202 | 批次 E：全命令面——workers/grants/body/doctor/mode/context/session/new/retry/failover/thinking/scoped-models；export(.rcmission 脱敏包)/import(只读不恢复授权)/share(诚实指向)/settings(白名单+原子写)/reload(安全域拒绝) | 恶意 bundle 全拒（穿越/篡改/secret/非 zip） |
| #203 | PR-11：operator.sock（0600 JSONL）——SO_PEERCRED peer identity（principal 唯一来源）、display hash 换卡防护、CSRF/Origin 防线、/estop 直达 rosclawd | 伪造 root 忽略、hash 不匹配拒、重放拒、无 daemon 诚实 |
| #204 | PR-12：LIMO 完整闭环（§18 C1–C10）+ arm64 发布打包/安装/回滚；SimActionChannel（SIM 物理权威，SIMULATED receipt）；PersistentMcpClient（观测/执行同进程）；K9 live（真实 K3 自主完成闭环） | **K9：真实模型完成观测→两卡授权→SIM 执行→位姿验证→单次消费** |
| #205 | 批次 F 一阶段：ReasoningBranch 双时间线——/tree 只读、/fork 开新 SIMULATION mission（不复制任何 authority，动作进行中拒绝） | fork 后 grants/approvals 为空、mode 强制 SIM |
| #206 | 批次 G：ACP adapter（官方 SDK）——session→Mission、prompt→turn、事件→session update；approval 只呈现不代决 | 官方 SDK client 真实 stdio 往返 |
| #207 | K4/K5 live 健壮性：协议工具字段规则明示；hire_worker instructions 多层提取+兜底 | K4 委派闭环、K5 授权闭环 live |
| #208 | 深度复盘：ModeldGateway socket 冲突（每实例唯一 socket+锁）、TUI 入口路径、fork 读 canonical journal、operator.sock 请求上限、timingSafeEqual、mcp 禁内联 secret、TUI 审批走 operator.sock、ACP 尾巴补推 | §19.6 攻击清单 11 项全显式测试 |
| #209 | 冲突审查：旧 MCP server 只读工具注解（readOnlyHint，S0/S2 观测可用）、ur5 banner 改 stderr、HTTP principal 按 mission owner 解析、async delta 回调 await（/v2 delta 丢失修复）、PersistentMcpClient 按 loop 分会话；全链路 E2E 套件 | `test_full_chain_e2e.py` 单跑覆盖 15 个链路环节 |

## 3. 架构总览（实施完成后）

```text
┌ rosclaw-tui (pi-tui 0.83.0) ─┐   ┌ ACP clients (Zed…) ─┐
│ Editor/IME/卡片/命令路由      │   │ 官方 SDK stdio       │
└──────┬───────────────────────┘   └──────┬───────────────┘
       │ HTTP control + SSE(Last-Event-ID) │ JSON-RPC stdio
┌──────┴───────────────────────────────────┴───────────────┐
│ rosclaw-agentd（唯一 Native Agent，无特权）                │
│  AgentLoop ↔ MissionStore(journal) ↔ ContextCompiler     │
│  Tool/Capability Catalog（硬过滤）/ Worker / Team         │
│  CommandService（命令永不进模型）/ AgentEventV2           │
│  operator.sock（SO_PEERCRED + display hash + estop）      │
└──┬───────────┬──────────────┬───────────────┬────────────┘
   │ModelGateway│ToolCatalog    │OperatorBroker  │Worker Fabric
┌──┴──────┐ ┌──┴───────────┐ ┌─┴───────────┐ ┌─┴──────────┐
│rosclaw-  │ │ MCP servers  │ │ 审批卡/Grant │ │native/stdio│
│modeld    │ │(观测 OBSERVE)│ │EXACT_ACTION  │ │/外部 CLI   │
│(pi-ai)   │ │(动作=不可调用)│ │单次消费      │ │WorkerPack  │
└──────────┘ └──────┬───────┘ └──────┬──────┘ └────────────┘
                    │ SIM actuation  │ proposal/permit
              ┌─────┴───────┐   ┌────┴────────────────┐
              │limo-sim(SIM)│   │ rosclawd（唯一物理权威）│
              │SIMULATED    │   │ Policy/Permit/Receipt  │
              │receipt      │   │ E-Stop                 │
              └─────────────┘   └───────────────────────┘
```

## 4. 关键设计决策与依据

1. **物理动作永不模型可调用**：契约层（ToolDescriptorV2 validator）+ catalog 执行守卫 + resolver 硬过滤 + strict_schema 拒绝，四重独立防线；物理动作唯一路径 `REQUEST_APPROVAL → Operator → Grant（单次）→ REQUEST_ACTION → rosclawd`。
2. **命令是控制协议不是聊天**：CommandSpecV1 注册表（owner: LOCAL_UI/AGENT/MODEL/MISSION/SAFETY_CONTROL），SAFETY_CONTROL 永不在通用注册表；unknown 命令提示而非发送。
3. **双时间线**：推理分支可 fork；物理事实线 append-only，不可回滚、不被遮蔽——fork 永远开新 SIMULATION mission 且不复制任何 authority。
4. **Compaction 不毁证据**：canonical journal 只增不减；view 由 journal 投影；entry_id/seq 稳定身份；切点不拆 tool 对与 atomic_group；持久化 entry 记录 covered hashes/supersedes/provider/prompt version。
5. **身份只有一个来源**：operator.sock 的 SO_PEERCRED；HTTP 端点 principal 缺省按 mission owner 解析；请求体 principal 字段在 operator 通道永远被忽略。
6. **模型层可替换**：ModelGateway 协议下 legacy（OpenAICompat）与 modeld（pi-ai）并存，backend 一行配置切换，Kimi 现有配置零改动迁移。
7. **证据域诚实**：SIMULATED receipt 标注 `usable_for_real_execution=false`；无声学观测时只声称"驱动已执行"；SHADOW/REAL 无 daemon/硬件时明确拒绝。

## 5. 测试与验证矩阵

| 层 | 数量/状态 |
|---|---|
| Python 全量回归 | **6127 passed**（含契约 golden 22 份、架构不变量、攻击回归、全链路 E2E） |
| Node（rosclaw-tui / rosclaw-modeld） | 14 + 17 全绿 |
| Lint / 类型 | ruff 全净；mypy 1079 文件全净 |
| **真实 Kimi K3 live（K0–K9）** | **11/11**：API 探测、认知诚实、只读工具环、SIM 闭环+崩溃恢复、委派、授权单次消费、团队、modeld 后端、modeld 全 turn、**LIMO 自主闭环** |
| 真实 claude CLI WorkerPack | ACCEPTED 一致性 |
| 打包 | wheel 构建验证（agentd/acp/limo 文件齐全）；arm64 bundle 结构+安装/回滚语义测试 |
| 环境性失败（非回归，基线同样失败） | firstboot×3+1、lerobot×4（本机无 LeRobot 运行时装配） |

## 6. 复盘发现并已修复的代表性问题

1. **ModeldGateway UDS 冲突**：多实例共享 socket 路径互相杀伤（复现后修复：每实例唯一 socket+启动锁）。
2. **per-call MCP 会话丢状态**：set_initial_pose 对下一次 get_pose 不可见 → PersistentMcpClient（生产单 loop 同进程；跨 loop 按 loop 分会话）。
3. **OpenAI 函数名禁点号**：工具 id wire 上映射 `__`（catalog 保证单射），Kimi live 400 → 修复。
4. **/v2 delta 静默丢失**：async 回调未 await。
5. **旧 MCP 全被误判 PHYSICAL_ACTION**：源头补 readOnlyHint（S0/S2 观测恢复可用，动作保持不可调用）。
6. **hatch 打包遵守 .gitignore**：误加 `agentd/` 会把 agentd 排除出 wheel（CI Product Acceptance 拦截，锚定修复）。
7. **principal 三处不一致**（CI uid=1001 暴露）：mission owner 动态 uid、operator.sock peer uid、测试统一 LOCAL_PRINCIPAL、HTTP 缺省按 mission owner。
8. **协议工具悬空 tool_call**：Kimi 拒绝无响应 tool_call → 成功提交也回执 tool result。

## 7. 安全红线复核（§19.6 攻击清单全过）

模型文本 /approve、tool result 伪造授权、worker 伪造回执、伪造 principal、display hash 不匹配、grant 重放、body hash 漂移、mode 不匹配、客户端断开→待定→过期、OAuth token 不入日志/事件/WorkOrder、浏览器 CSRF——全部有显式回归测试且通过。全仓 secret 扫描仅合成测试值。

## 8. 已知边界（诚实清单，未伪造完成）

- 批次 F 二阶段（同 mission 分支切换、/clone）——第一阶段不变量已锁定，待扩展后开放；
- modeld OAuth 登录（返回 501 明示，本批 API Key 完整）；
- /copy（TUI 剪贴板 vendor）、AG-UI Web adapter（内部事件已稳定，可做）；
- 真机 REAL 闭环：需实体 LIMO + rosclawd 在场；SIM/SHADOW 门控已就位，绝不用 SIM 证据冒充；
- 性能目标（§12.6：p95 渲染/延迟）为验收目标非安全承诺，未做系统测量。

## 9. 复用登记（依据补充文档 §13 要求）

| 组件 | 方式 | 版本/commit | 许可 |
|---|---|---|---|
| @earendil-works/pi-tui | 直接依赖（精确锁定） | 0.83.0 / 0e633790 | MIT（third_party/pi 含 LICENSE+NOTICE） |
| @earendil-works/pi-ai | 直接依赖（精确锁定） | 0.83.0 / 0e633790 | MIT 同上 |
| mcp（Python SDK） | 直接依赖 | >=1.0.0,<2.0.0 | MIT |
| agent-client-protocol | 直接依赖 | >=0.12.0 | Apache-2.0 |
| Pi compaction 切点算法 | 行为移植（非 import） | 同上 | MIT，已注明 |
| Pi RPC settled/事件分类 | 概念借鉴 | 同上 | 不作 wire 协议 |
| 禁止进入运行路径 | pi-coding-agent、pi-agent-core、Hermes、OpenCode Agent | — | 架构测试锁定 |

## 10. 后续建议（优先级序）

1. AG-UI Web adapter（内部事件已稳定，增量小）；
2. modeld OAuth（pi-ai 已有流，接通交互即可）；
3. 批次 F 二阶段（先补不变量测试：分支切换中动作 fail-closed）；
4. 真机 REAL 验收（实体 LIMO 到位后按 §18 C1–C10 跑 SHADOW→REAL 门控）；
5. TUI 性能测量与 10k 行 transcript 窗口化。
