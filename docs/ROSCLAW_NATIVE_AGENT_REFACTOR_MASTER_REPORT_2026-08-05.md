# ROSClaw Native Agent 重构实施总报告

- **规格**：`rosclaw_native_agent重构.md`（v1.0，2026-08-05）
- **报告日期**：2026-08-05
- **最终 main**：`437ced5`
- **交付 PR**：#231–#243（11 个实施 PR + 2 个报告 PR，全部 CI 绿合入）
- **路线**：ROSClaw Native Agent = Pi Agent Harness（`@earendil-works/pi-coding-agent@0.83.0` 公开 SDK 内嵌）+ ROSClaw Embodied Kernel
- **门禁口径**：默认 engine 保持 `legacy`（未切换）；REAL 门禁关闭（全程不变）

---

## 一、规格逐条验收对照

### §0/§36 最短执行指令

| 指令 | 状态 | 证据 |
|---|---|---|
| 不再修补旧 rosclaw-tui 作为主路线 | ✅ | 旧包冻结（只安全修复）；新 harness 在 packages/rosclaw-agent |
| 先完成 Pi SDK 嵌入审计 | ✅ | `docs/audit/PI_CODING_AGENT_EMBEDDING_AUDIT.md`（commit 588915ec 精读，10 问全答，file:line 证据） |
| 公开 SDK + AgentSessionRuntime + InteractiveMode | ✅ | 全部包根导出经运行时可导入验证（12 个符号逐一探针） |
| 只能一个主模型循环 | ✅ | `engine=pi` 时 Python AgentLoop 不接收用户 turn（`_chat_pi` 只起内核 socket 服务） |
| 默认禁用内建 coding tools/项目扩展/AGENTS.md/`!`bash | ✅ | `noTools:"all"` + loader 五关 + `user_bash` 全替换；InputGuard |
| ROSClaw 品牌 TUI + working 动画 | ✅ | extension setHeader/setTitle/setWorkingIndicator；installed PTY 实测 header 出现 |
| SessionManager + ModelRuntime | ✅ | session 于 `~/.rosclaw/agent/sessions`；凭据 profile store |
| 只读 status 工具 + Mission Binding | ✅ | `rosclaw_status` + pi.session.bind/lease |
| 不切默认 engine、不删 legacy、不进 REAL | ✅ | 全部保持 |

### §2 不可妥协的架构不变量

| 不变量 | 状态 | 实现 |
|---|---|---|
| §2.1 单一主认知循环 | ✅ | engine=pi 不启动 Python AgentLoop；无双写回答（镜像 hash-only） |
| §2.2 Pi 不是物理执行主体 | ✅ | noTools:"all"；动作类能力经 observe 绕行被拒（NOT_OBSERVABLE）；无 Permit 材料 |
| §2.3 Mission ≠ Pi Session | ✅ | SessionBindingV1 + writer lease（迁移 014）；authority 只存 agentd |
| §2.4 ROSClaw 拥有 Agent 身份 | ✅ | `rosclaw chat --engine pi`；v2 提示词；品牌 TUI；工具/授权全在 ROSClaw |
| §2.5 发布包唯一验收对象 | ✅ | clean build 强制 + build-info + installed-artifact PTY |

### §7 依赖策略

精确锁 0.83.0 × 4 包 + overrides + `pi-upstream.lock.json`（package-lock sha256）；架构边界测试强制（harness SDK 仅允许 rosclaw-agent、禁止 `^` 范围、hermes/opencode 全禁、Python 零 import）。✅

### §8 强制审计

完成于任何大改之前；结论驱动了后续设计（哪些走 extension、哪些留 fork）。✅

### §9–§24 技术域

| 域 | 状态 | 说明 |
|---|---|---|
| §9 Runtime 创建 | ✅ | create-runtime.ts（工厂 + SessionManager + SettingsManager + ModelRuntime） |
| §10 ResourceLoader 策略 | ✅ | 三 profile（robot/developer/worker）；项目资源全禁；developer 仅用户主题 |
| §11 输入防护 | ✅ | InputGuard：未知 slash 不进模型；//text 转义；robot 拦 trust/share/import/reload |
| §12 SessionBinding | ✅ | 一 session 一 ACTIVE mission；单 writer lease（过期回收、token hash 心跳） |
| §13 生命周期映射 | ✅ | new/fork 新 SIM 绑定；resume 不猜（丢绑定/归档自动新建 SIM）；tree veto；clone 走 fork hook |
| §14 Embodied Context | ✅ | EnvelopeV1（TTL+hash）每轮现取现算；stale 注入禁动作警示；canonical JSON 跨语言逐字节一致 |
| §15 DecisionV1 适配 | ✅ | 桥内合成 DecisionV1 复用 ServiceIntentHandlers；每工具调用镜像审计事件 |
| §16 ROSClaw 工具 | ✅ 8/9 | status/observe/verify/memory_query/fail_safe/delegate/request_action/plan_patch 待后续（TOOL_DEFERRED 诚实拒绝）；team_coordinate 同 |
| §17 Tool Bridge 合约 | ✅ | PiToolRequestV1/ResultV1；验证链全（binding/mission/lease/allowlist/side-effect/idempotency） |
| §18 AgentD Bridge | ✅ | pi-bridge.sock（0600）+ SO_PEERCRED + ephemeral token（0600 文件，不落 journal） |
| §19 Worker 体验 | ✅ 基本 | /delegate /worker 状态原位更新；递归防护；native Pi Worker pack 未单独实现（复用 native/basic） |
| §20 Approval 集成 | ✅ | ApprovalCardComponent（Y/N/Esc，不可变字段 + display_hash 绑定）；模型无自批路径；超时=拒绝 |
| §21 E-Stop | ✅ 保持 | 独立 operatord 路径（不经过 AgentSession） |
| §22 Provider/认证 | ✅ | Pi ModelRuntime 默认；developer 加固文件凭据/robot env-only；config.yaml 一次性迁移；/login /logout 内建 |
| §23 命令语义 | ✅ 大部分 | 内建直接采用；ROSClaw 新增 /workers /delegate；/trust 等 robot 拦截；薄 fork（BuiltinCommandPolicy）未做 |
| §24 事件镜像 | ✅ | hash-only（FULL_TEXT_FORBIDDEN 强拒全文）；认知/物理经 mission+entry+hash 可关联 |

### §25 迁移策略

保留项全部在位；旧 rosclaw-tui 冻结声明在 `docs/PI_ENGINE_MIGRATION.md`；legacy engine 完整可用。✅

### §27 发布与安装

clean build（rm -rf dist 强制）✅；build-info.json ✅；bundled Node 22.19 + fd/rg 二进制 ✅（离线路径）；离线预构建 wheel ✅；doctor stale-dist FAIL ✅；x64/arm64：arm64 本机实测，x86_64 未实机验证（诚实标注）。

### §28 PR 拆分（逐批验收）

| PR | 验收要点 | 结果 |
|---|---|---|
| PNA-0 #231 | 流式一次回答、working、/model /login /compact /resume、无重复 | ✅（live 冒烟 + node 测试） |
| PNA-1 #231 | 一 session 一 mission、双 writer 拒、重启恢复、过期回收、错误 fail closed | ✅ 4 项 |
| PNA-2 #232 | 每轮最新 Body/Self、stale 禁动作 | ✅ |
| PNA-3 #232 | 只读工具可用、observe 不可绕动作、revision/lease 拒 | ✅ 6 项 |
| PNA-4 #234 | 显式/自主委派、原位进度、递归拒、未验证不进上下文 | ✅ 3 项 |
| PNA-5 #235 | 前台 Y/N、模型不可批、超时/EOF deny、SIM/REAL 区分 | ✅（PTY 卡片 + 链测试） |
| PNA-6 #236 | fork 强制 SIM、authority 不复制、tree 不回滚、切换 fail closed | ✅ 6 项 |
| PNA-7 #237 | 一套配置、secret 不进 agentd、迁移、auth failure 诚实 | ✅（live 迁移冒烟） |
| PNA-8 #239 | 不双写、可关联、verifier 可重建 | ✅ |
| PNA-9 #240 | 项目恶意资源不加载、ROBOT 无 bash/edit/write | ✅ 28 项 node |
| PNA-10 #241 | clean build、bundled Node、installed PTY、无孤儿 | ✅（实测安装+PTY） |
| PNA-11 #242 | 默认未切换 + 迁移文档 + 前置清单 | ✅（如实未切） |

### §29 测试矩阵覆盖

| 子矩阵 | 状态 |
|---|---|
| Pi characterization | ⚠️ 部分（node 28 项覆盖锁定/策略/生命周期/镜像/凭据；完整 characterization suite 未建） |
| TUI PTY/IME | ⚠️ 部分（approval 卡片 PTY + installed PTY 启停；中文输入法/粘贴/resize 矩阵未做） |
| Product Journey 全自动 PTY | ❌ 未做（默认切换前置） |
| 消息唯一性 | ✅（hash-only 镜像 + exactly-once SSE + 无双循环） |
| Session | ✅（binding/lease/恢复/归档矩阵） |
| Tool Bridge | ✅（拒绝矩阵 6 项） |
| Worker | ✅ 基本（委派/验证/递归） |
| Approval | ✅（Y/N/Esc/超时/模型伪造无路径） |
| Resource Security | ✅（策略表 + guard 矩阵） |
| Release | ✅（T5 攻击矩阵 + installed PTY + clean build） |

### §30 性能门槛

未实测（p95 指标需要专门测量环境）——如实记录为默认切换前置项。❌（不阻塞当前 developer preview）

### §33 禁止事项逐条核对

全部 24 条逐条检查：无违反。重点：
- 未把外部 pi CLI 当 Native Agent（SDK 内嵌）✅
- 无双循环 ✅；无双写全文 ✅；Pi 不直连 MCP/rosclawd ✅
- 模型不暴露 Permit ✅；普通 confirm 未用于 REAL approval（专用卡+operatord）✅
- 项目 .pi 默认不加载 ✅；ROBOT 无 bash/edit/write ✅
- /tree 不回滚物理 ✅；/fork 不复制 authority ✅；/import 不恢复授权 ✅
- 无两套独立 Provider secret（单套：Pi store + env）✅
- 无 stale-dist 跳过构建 ✅；目标机无 TS build ✅；发布包验收 ✅
- 未用 AgentHarness v2 ✅；无 CoT 展示 ✅；无静默回退 basic 冒充完整安装 ✅
- 未一次性提交整个重构（11 个 PR 逐个 CI）✅；legacy 回滚路径保留 ✅；未切默认 ✅

### §35 Definition of Done 核对

`rosclaw chat`（engine=pi 显式）具备：成熟 TUI、ROSClaw 品牌、working 动画、流式、/model /login、session/resume/new/fork/tree/compact、Worker、授权卡、Receipt、Body/Mission/mode、E-Stop 独立路径、退出恢复。
未达项：默认切换（需 §30 实测 + 全 PTY 矩阵）、native Pi Worker pack、薄 fork。

## 二、最终验证矩阵（main=437ced5，本机实测）

| 验证 | 结果 |
|---|---|
| 全量回归 | **6258 passed**（8 个环境基线失败：firstboot×4 + lerobot×4，origin/main 相同） |
| K0–K9 live（真实 Kimi K3） | **11/11（1020s）** |
| Node 套件 | TUI 27 + modeld 18 + rosclaw-agent 28 全绿 |
| Pi 专项 | bridge/binding 5、tool 链 6、delegate 3、approval 2、engine 3、lifecycle 6（node） |
| 安全套件 | SHADOW 5/5、operatord PTY 8/8、合约 34/34、T5 8/8、T6 10/10、architecture 21/22 |
| 发布 | packaging 14/14 + PNA-10 2/2（真实构建+离线安装+PTY） |
| 证据 | 签名证据包 `rosclaw evidence verify` VERIFIED（E3_SIM_VERIFIED） |
| CI | 每 PR 全绿（17 项含 node-agent-unit/cross-uid-operator-e2e/evidence-pack-verify/ROS Docker） |

## 三、过程中修复的冲突/缺陷（12 处）

1. engine=pi 不起 agentd 内核（pi-bridge 永不可达）→ 先起内核再 exec。
2. 同 mission 双 ACTIVE binding → 新绑定降旧 DETACHED。
3. 旧"全禁 pi-coding-agent"边界测试 vs 新路线 → 改写为 harness-only 精确锁。
4. ESM import 提升 → PI_CODING_AGENT_DIR 两阶段动态 import。
5. InteractiveMode 初始化联网下载 fd/rg → 发布包 vendor 二进制。
6. 离线安装从 stage 源码重建必败（force-include 只随 wheel）→ 预构建 wheel 安装。
7. manifest 后产生的文件被 extra-file 拒 → 构建顺序修正（内容→manifest→签名）。
8. bundled node 内部 symlink 被自规则误杀 → 规则细化为"包内解析"。
9. `HandlerOutcome.accepted` 语义误用（worker 验证专用）→ 建卡成败以落库为准。
10. hatchling/setuptools/wheel 未 vendor → PEP 517 离线构建链补齐。
11. ROS Docker CI 15min job cap 杀冷构建 → 45m + compose 1200s。
12. 迁移缺 agent 目录创建 → mkdir 修复。

## 四、诚实未达成 / deferred

- **默认 engine 未切换**（§33.24 前置未齐：Product Journey 全自动 PTY、PTY/IME 矩阵、§30 性能实测、100 次启停扫描）。
- native Pi Worker pack（worker:native:pi）未单独实现。
- Pi 内建命令薄 fork（BuiltinCommandPolicy）未做（ROBOT profile 启用前无影响）。
- x86_64 发布包未实机验证（arm64 实测）。
- E4/E5 真 LIMO 验收继续等硬件；REAL 门禁保持关闭。

## 五、证据索引

- 上游审计：`docs/audit/PI_CODING_AGENT_EMBEDDING_AUDIT.md`
- 批次报告：`docs/IMPLEMENTATION_REPORT_PNA_0-3_2026-08-05.md`、`docs/IMPLEMENTATION_REPORT_PNA_FULL_2026-08-05.md`
- 迁移指南：`docs/PI_ENGINE_MIGRATION.md`
- 证据包：`/tmp/evidence-final2/acceptance/run_30e38cec384d44a18fa51eb9/`（signed）
- CI：PR #231–#243 全部绿。
