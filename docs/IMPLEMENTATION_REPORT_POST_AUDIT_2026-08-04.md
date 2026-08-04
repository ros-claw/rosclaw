# ROSClaw 审计后冲突复查与全链路闭环深测报告

- 日期：2026-08-04
- 基线：`main = 6060464`（审计五线 #213/#214/#217/#220 全部合入后）
- 修复 PR：#222（`agent/post-audit-conflict-fixes`）
- 结论：**发现 3 处真实冲突/缺陷，全部修复并验证；全链路闭环深测全绿。**

---

## 1. 复查范围与方法

针对审计五线（operatord 授权拆分 / TUI 实时化 / modeld provider 层 /
签名打包 / SHADOW+证据包）与**既有功能**的交界做逐面审查：

| 审查面 | 方法 |
|---|---|
| agentd 决策面（decide/revoke/estop） | 全仓 grep 所有 `decide(` / `revoke` 调用点，逐一核对新旧口径 |
| rosclawd ACL 与 REAL 不变量 | 通读 permit/proposal 发放路径，确认 SHADOW 放松未波及 REAL |
| TUI ↔ operatord/agentd 通道 | 审查 `/approve` `/estop` 错误呈现与降级行为 |
| modeld ↔ ModeldGateway/ failover | 审查 socket 生命周期、权限、唯一性 |
| 命令面（CommandService） | 审查 `_ARGS_SCHEMAS` 与 mutating 命令的授权路径 |
| MCP / ACP 适配器 | grep 确认无任何 decide/approve 能力残留 |
| 证据包 | 实跑验收，检查 secret scan 与包完整性 |

## 2. 发现的冲突与修复（PR #222）

### 2.1 REPL `/approve` `/deny` 死命令（P0-01 冲突）— 已修复

- **现象**：`agentd/cli.py::_chat_repl` 的 `/approve` 直接调用
  `service.decide_approval(...)`（无 `_from_operatord`）。自 #217 起，
  该方法对非 operatord 调用一律抛错——**命令名存实亡**，用户每次
  只得到"授权操作失败"。
- **根因**：#217 收紧 `decide_approval` 时漏改了 REPL 这个调用点
  （TUI 已走 operatord.sock，REPL 是 fallback 路径，无测试覆盖该交互）。
- **修复**：`/approve|/deny` 改经 `operatord.sock` 的 `approvals.decide`
  （`display_hash_for` 绑定卡片 + `operator_call`），与 TUI 同一通道；
  operatord 未运行时打印接线指引
  （`rosclaw operatord enroll && rosclaw operatord start`）。
  `/approvals` 列表补打印 `display_hash` 供操作员核对卡片内容指纹。

### 2.2 `/revoke` UI 命令面口径不一致（P0-01 冲突）— 已修复

- **现象**：HTTP `POST /grants/{id}/revoke` 自 #217 起默认 403，但
  CommandService 的 `/revoke` 命令仍直连 `service.revoke_grant`——
  **同一操作、两个面、两种策略**，构成决策面旁路。
- **修复**：`/revoke` 改经 `operatord.sock` 的 `grants.revoke`
  （operatord 用 enrollment 私钥签 proof 后转发 agentd `apply_revoke`）；
  operatord 缺席时诚实拒绝并给指引。撤销属 fail-safe 方向，operatord
  侧不要求 human presence（与 #217 设计一致）。

### 2.3 modeld UDS 权限竞态（flaky `test_uds_permissions`）— 已修复

- **现象**：全量回归（`-x`）中 `test_uds_permissions` 失败一次；
  单跑、整目录跑、重跑均通过——典型负载相关抖动。
- **根因**：`packages/rosclaw-modeld/src/server.ts` 的
  `chmodSync(sock, 0o600)` 在 `server.listen` **回调**里执行，晚于
  socket 文件出现；Python gateway 以 `Path.exists()` 轮询等待，
  会在"文件已 bind、chmod 未执行"的窗口内 stat 到 umask 默认权限。
  （安全性影响小：socket 父目录在 listen 前已 0700，但测试断言
  socket 本体 0600 会抖动失败。）
- **修复**：listen 前 `process.umask(0o077)`，socket 创建即 0600；
  回调与 `error` 路径恢复 umask（chmod 保留作兜底）。
- **验证**：修复后连跑 8 次全绿；modeld 18/18 node 测试绿。

## 3. 审查未发现问题的面（记录结论）

| 面 | 结论 |
|---|---|
| rosclawd REAL 不变量 | 完好。permit 发放仍要求 ARMED + 无 recovery 待审 + ledger 健康 + 无 E-Stop 锁存 + REAL executor 已注册；SHADOW 走独立 executor 键（`capability:SHADOW`），与 REAL 严格分离（FTC-100）。 |
| daemon `proposal.decide` ACL | 仅 daemon euid 或有效 enrollment proof；proof 绑定 request_id+approve+nonce+decided_at+enrollment_id+display_hash；篡改任一项即拒（FTC-050 测试覆盖）。 |
| agentd 决策面 | 生产代码中除 operatord 通道外无任何 `decide` 调用点残留；MCP 工具面、ACP 适配器均无可决定授权的能力。 |
| TUI `/approve` `/estop` | 错误以红色/黄色文本诚实呈现；operatord 缺席时 fallback 到 agent operator.sock 会得到明确拒绝文案，不会假装成功。 |
| operatord `/estop` | 无 daemon 连接时明确回报 "nothing was stopped (honest)"；有 daemon 时直连 rosclawd，不依赖 LLM/云。 |
| evidence pack | 实跑验收产出 11 件 artifact；`secret_scan_clean=true`；`run_manifest` 含 commit/dirty/level/test_ids/operator。 |
| modeld failover 链 | 每实例唯一 socket（`modeld-{pid}-{id}.sock`）+ 启动锁，多 gateway 共存无 unlink 冲突（#208 修复仍有效）。 |

## 4. 全链路闭环深测矩阵

| 验证 | 结果 |
|---|---|
| 全量回归（pytest，含修复） | **6154 passed**（8 个已知环境性失败，见 §5） |
| ruff | clean |
| mypy | clean，1090 source files |
| TUI node 测试 | 25/25 |
| modeld node 测试 | 18/18 |
| **K0–K9 live**（真实 Kimi K3 + 真实 operatord 决策链） | **11/11，947s** |
| LIMO 验收（真实 OperatorDaemon 审批） | **12/12 checks**（C1–C10 + T6/T7） |
| E3 证据包 | 11 artifacts，secret scan 干净，`E3_SIM_VERIFIED` |
| `test_uds_permissions` 稳定性 | 修复后 8/8 |
| 冲突修复专项（operator_socket/shadow/commands/operator） | 42 passed |

闭环链路覆盖：用户输入 → mission → 模型回合（SSE 流式）→ 工具调用
（LIMO SIM MCP）→ 授权卡片 → **operatord enrollment proof 决策** →
grant（EXACT_ACTION 单次）→ 执行 → 终态回执 → journal/证据包；
外加 daemon 侧 SHADOW 全链（proposal→proof 决策→permit→SHADOW receipt，
actuation 硬阻断）。

## 5. 已知环境性失败（非回归，origin/main 基线相同）

- `tests/test_firstboot.py` ×3、`tests/feedback/test_firstboot_integration.py` ×1
  （本机 firstboot 子进程环境）。
- `tests/integrations/test_lerobot_*` ×4 —— 本机无 LeRobot runtime
  （"No LeRobot runtime configured and current interpreter cannot
  import LeRobot"）。

以上 8 项与审计改动零交集（五个 PR 未触碰这些模块），为基线性环境失败。

## 6. 证据位置

- 证据包：`/tmp/evidence-final/acceptance/run_d1b1cf7b1377443bb3fea589/`
  （11 artifacts + run_manifest，E3_SIM_VERIFIED）
- 修复 PR：#222（CI 13 项）
- 本报告：`docs/IMPLEMENTATION_REPORT_POST_AUDIT_2026-08-04.md`

## 7. 剩余诚实 deferred（与审计口径一致）

- FTC-110 真机有界 REAL（需物理 LIMO + 目标侧 operatord/rosclawd）。
- FTC-130 8/24h soak。
- Batch F 阶段二（同 mission 分支切换、/clone）。
- modeld OAuth/device-code（当前诚实 501）。
- AG-UI Web adapter、/copy 剪贴板。
- PTY/IME/终端矩阵、p95 性能实测、E6 独立第三方复跑。
- 已知限制：`apply_revoke` 的 proof 在 agentd 侧只做存在性检查
  （HMAC 私钥仅在 operatord，agentd 无法本地验签；socket 0600 本机
  边界兜底）。如需更强保证，可将 revoke proof 也经 rosclawd ACL 验证。
