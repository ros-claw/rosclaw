# ROSClaw 二次审计（产品闭环方案）NA-FIX 实施报告

- 日期：2026-08-05
- 审计基线：`ROSClaw_NATIVE_AGENT_REFACTOR_二次审计与产品闭环方案_2026-08-05.md`
- 交付：PR #246（main=`74f1ed6`）
- 方法：**红测试先行**——每个缺陷先写稳定复现的失败测试，修复后转绿。
- 门禁：默认 engine 仍 legacy；REAL 门禁不变。

## 1. P0 缺陷处置对照

| 审计发现 | 修复 | 验证 |
|---|---|---|
| P0-1 新会话必现 context hash mismatch（Python 30.0 vs TS 30） | RFC 8785 canonical JSON（`contracts/pi/canonical.py`，对齐 JS Number#toString：整数折叠/-0→0/指数去前导零）；envelope hash 全面切换 | golden corpus（浮点/Unicode/嵌套/key 序）+ 完整真实 envelope Python→Node 字节一致；生产冒烟 `hash match: true` |
| P0-2 Session/Mission/lease split-brain | ActiveSessionContext 单一动态源（工具/镜像/审批执行时读取，不再捕获启动值）；SessionLeaseManager 唯一 bind+heartbeat 管理点（lease_token 不丢、旧 heartbeat 停）；--continue/--resume 经 SessionManager.open 不预建无用 Mission | node 32/32；完整 100 次切换事务验证在后续批次（如实标注） |
| P0-3 退出泄露 `pi --session` 旁路 | 薄补丁 patch-01：退出提示 `rosclaw chat --resume <id>`（patch-package 风格 postinstall 应用，锚点漂移硬失败） | 补丁测试（红→绿） |
| P0-4 request_action 未注册 | buildRequestActionTool 真实注册进运行时；prompt 按真实工具面重写（删未注册工具） | 注册/prompt 红测试转绿 |
| P0-5 Approval 监听顺序颠倒 | 两阶段 ActionCoordinator：execute→propose→approval_id onUpdate→TUI 按 approval_id 精确展卡（不再 tool_execution_start 盲等、不再取 pending[0]） | 生产冒烟 propose/status 通过 |
| P0-6 全局 grant 猜测 + 中文串判成功 | grant 按 request_id 精确匹配（SQL WHERE request_id=?，consumed/revoked 拒）；结构化 ExecutionReceipt（approval_id/grant_id/terminal_receipt/status/error_code） | test_pi_approval 全绿（含 decline 无 grant） |
| P0-7 context_revision 全 0 不校验 | Node 工具携带已验证 envelope 的 revision/body/mode；dispatcher 对动作类硬校验 exact revision（CONTEXT_REVISION_MISMATCH） | 红测试（不再出现 context_revision: 0） |
| P0-8 安全命令未注册 + ROBOT 内建拦截太晚 | 全量命令注册（status/mission/body/tools/approvals/revoke/doctor/estop/cancel/evidence/memory）；patch-02 BuiltinCommandPolicy 内建 dispatch 前置拦截；/estop dedicated operatord 通道 | 命令注册测试；estop 无 daemon 诚实不可用 |
| P0-9 SHADOW/REAL 未强制 ROBOT | `_chat_pi` 按 mission mode 强制 profile（SIM=developer，其余=robot） | cli 层静态验证 |

## 2. P1 处置

- P1-1：hideThinkingBlock=true（默认隐藏 raw reasoning；debug 可在 /settings 开）。
- P1-2：envelope 完整投影（task graph/OBSERVE capabilities/近 3 回执/worker 单/真实 tool policy）。
- P1-3：prompt 与真实工具面一致（plan_patch/team_coordinate 仍诚实 TOOL_DEFERRED 不出现在 prompt）。
- P1-4：双凭据路径收敛留待 NA-FIX-7（broker 统一）——如实未做。
- P1-5：`handlers.request_context()` 请求级不可变 mode/principal（退出恢复，不跨请求残留）。
- P1-6：header 显示 Mission/Body/revision/Operator ready（不再显示 engine=pi 实现细节）。
- P1-7：报告口径修正——本报告逐项标注"完整事务验证/IME 矩阵/性能实测"为未做。

## 3. 验证矩阵（main=74f1ed6）

| 验证 | 结果 |
|---|---|
| 红测试（9 项） | 全部红→绿 |
| agentd 套件 | 422/422 |
| node 套件 | 32/32 |
| 生产冒烟（真实内核） | pi.context hash match ✓、pi.action.propose 精确 approval_id ✓、status 轮询 ✓ |
| 全量回归 | 见 CI（PR #246 全绿，含 node-agent-unit/cross-uid-e2e/evidence-pack-verify） |
| K0–K9 live | 见本轮终验（与合并前基线一致） |

## 4. 审计结论的当前状态

- 动作与审批闭环：Native Agent 中**已可用**（request_action 注册 + 两阶段 + 精确绑定 + 结构化回执）。
- 具身上下文：stale 风险**已消除**（跨语言一致 + TTL/篡改 fail closed）；投影**已非骨架**。
- Session 生命周期：split-brain 主因**已修**（动态上下文 + lease 管理点 + resume 不预建）；100 次切换事务验证**未做**。
- 产品身份：退出 resume hint 已收口；header/命令已收口；`pi` 仅存在于 license/SBOM/build-info 允许域。
- SIM 默认切换：**仍暂缓**（T-PRODUCT 全旅程 PTY、T-TUI IME 矩阵、T-SOAK、性能实测未做）。
- SHADOW/REAL：继续关闭。
