# ROSClaw Pi 重构 PNA 全系列实施报告（PNA-0 ~ PNA-11）

- 日期：2026-08-05
- 基线：`rosclaw_native_agent重构.md`
- 交付：PR #231（PNA-0/1）、#232（PNA-2/3）、#233（报告）、#234（PNA-4）、#235（PNA-5）、#236（PNA-6）、#237（PNA-7）、#239（PNA-8）、#240（PNA-9）、#241（PNA-10）、#242（PNA-11）
- 最终 main：`0dc593a`
- **默认 engine：legacy（未切换——切换前置清单见 docs/PI_ENGINE_MIGRATION.md）；REAL 门禁：关闭（不变）。**

## 1. 批次总览

| PR | 批次 | 内容 |
|---|---|---|
| #231 | PNA-0/1 | Pi SDK 嵌入审计（588915ec 精读）；packages/rosclaw-agent（InteractiveMode + ROSClaw 品牌 + noTools:all + 资源锁定 + `!` bash 关闭 + rosclaw_status）；pi-bridge UDS + SessionBinding（一 session 一 mission、单 writer lease） |
| #232 | PNA-2/3 | native_agent_v2.md；EmbodiedContextEnvelopeV1（TTL+hash）每轮注入（stale 禁动作）；pi.tools.execute 全验证链（binding/mission/lease/allowlist/OBSERVE-only/idempotency/DecisionV1 镜像）；engine=pi 起内核修复 |
| #234 | PNA-4 | rosclaw_delegate（验证不过不进主上下文）+ /delegate /workers + 原位进度 + 递归防护 |
| #235 | PNA-5 | rosclaw_request_action 授权链（建卡→operatord 决定→单次 grant 执行→回执）+ ApprovalCardComponent（Y/N/Esc，display_hash 绑定） |
| #236 | PNA-6 | session 生命周期（new/fork 新 SIM 绑定、resume 不猜、tree veto、authority 结构性不复制） |
| #237 | PNA-7 | 凭据 profile store（developer 加固文件/robot env-only）+ config.yaml→Pi settings 迁移 + doctor pi_engine（stale dist FAIL） |
| #239 | PNA-8 | 认知事件 hash-only 镜像（pi_event_mirrors；FULL_TEXT_FORBIDDEN 强拒全文） |
| #240 | PNA-9 | 资源策略三 profile + InputGuard（未知 slash 不进模型；robot 拦 trust/share/import/reload） |
| #241 | PNA-10 | clean build 强制、build-info.json、bundled Node 22.19 + fd/rg、离线预构建 wheel、installed-artifact PTY |
| #242 | PNA-11 | agent.engine 配置 + 迁移文档（默认 legacy 不动） |

## 2. 架构不变量核对（规格 §2）

- ✅ 单一主认知循环：engine=pi 时 Python AgentLoop 不接收用户 turn（_chat_pi 只起内核 socket 服务）。
- ✅ Pi 不是物理执行主体：noTools:all；rosclaw_* 工具全经 agentd 验证链；无 Permit 材料。
- ✅ Mission ≠ Pi Session：SessionBinding + writer lease；物理事实只追加。
- ✅ ROSClaw 拥有 Agent 身份：rosclaw chat 入口、v2 提示词、品牌 TUI、工具面、授权链。
- ✅ 发布包唯一验收对象：clean build + bundled runtime + installed PTY。

## 3. 最终验证矩阵（main=0dc593a，本机实测）

| 验证 | 结果 |
|---|---|
| 全量回归 | **6258 passed**（8 个已知环境基线失败：firstboot×4+lerobot×4，origin/main 相同） |
| K0–K9 live（真实 Kimi K3） | **11/11（1020s）** |
| Node 套件 | TUI 27 + modeld 18 + rosclaw-agent 28 全绿 |
| CI（每 PR） | 全绿（含 node-agent-unit/cross-uid-operator-e2e/evidence-pack-verify/ROS Docker） |
| 安全套件 | SHADOW FTC-100 5/5、operatord PTY 8/8、合约 34/34、T5 8/8、T6 10/10 |
| Pi 端到端 | 内核→bind+lease→注入→真实 K3 回合；迁移冒烟（config.yaml→settings→K3 ok）；installed PTY（header→/quit→零孤儿） |

## 4. 诚实 deferred / 未达成

- **默认 engine 未切换**（规格 §33.24）：Product Journey 全自动 PTY、PTY/IME 矩阵、§30 性能实测、100 次启停零孤儿扫描未完成——清单在 docs/PI_ENGINE_MIGRATION.md。
- PNA-4 的 native Pi Worker pack（worker:native:pi）未单独实现（现有 native/basic worker 复用；delegate 链完整）。
- 内建 slash 命令薄 fork（BuiltinCommandPolicy）未做——ROBOT profile 不启用前无影响。
- E4/E5（真 LIMO SHADOW/REAL）继续关闭，等硬件验收。

## 5. 关键经验（已录记忆）

- ESM import 提升：PI_CODING_AGENT_DIR 必须两阶段动态 import。
- InteractiveMode 初始化下载 fd/rg——发布包必须 vendor。
- hatch force-include 数据只随 wheel：离线安装绝不经 stage 源码目录。
- bundle manifest 之后产生的文件必被 extra-file 拒——构建顺序硬约束。
