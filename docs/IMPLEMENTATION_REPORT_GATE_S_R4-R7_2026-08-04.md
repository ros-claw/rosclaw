# ROSClaw 二次复核实施报告：Gate S 安全协议 + R4/R6/R7

- 日期：2026-08-04
- 输入：`ROSClaw_POST_AUDIT_二次复核与全链路验收方案_2026-08-04.md`
- 交付：PR #224（Gate S 协议重建）、PR #225（R4 + R6 + R7 + Node CI）
- 最终 main：`53a7a42`
- **REAL 门禁：继续关闭**（本轮修协议正确性与可信发布，不宣称 REAL-ready）。

---

## 1. 复核结论的处置对照

| 复核发现 | 处置 | 证据 |
|---|---|---|
| P0-1 human presence 只开 /dev/tty | **已修**：不可变卡片 + 显式 Y/N（默认/超时/EOF/无 tty 全 deny）+ 请求方前台进程组校验（/proc tpgid） | `operatord/human.py`；T2 真实 PTY 8 项 |
| P0-3 proof nonce ≠ daemon challenge nonce | **已修**：`DecisionChallengeV1` 由 daemon 签发、nonce 同源，operatord 原样签回 | `contracts/operator/decision.py`；合约篡改矩阵 30+ 项 |
| P0-4 decide 后续被 daemon-UID 检查挡住 | **已修**：proof 验证后走不暴露 socket 的内部 `_arm_after_operator_decision` / `_issue_permit_after_operator_decision`；`operator.proposals.list`/`challenge.get` 对已登记 operator UID 放行 | T0 静态边界（内部方法不经 IPC）；T1 e2e |
| P0-5 enrollment 内存态 + 首调抢注 | **已修**：Ed25519 取代共享 HMAC；daemon 持久化 registry（0600、原子写、双 fsync）；空表全拒；register/revoke/list 仅管理员；已焚毁 nonce 持久化 | registry 重启持久化测试；T1 `enrollment_survives_restart` |
| P0-6 agentd 只查 proof 存在 | **已修**：agentd 只接受 daemon 签名的 ACCEPT `DecisionReceiptV1`（签名校验 + 全字段精确匹配 + 未过期 + sqlite UNIQUE 防重放）；DECLINE 只关闭请求 | `service.verify_decision_receipt`；receipt 篡改矩阵 |
| P0-7 包内公钥自证 | **已修**：包外 trust anchor（--trusted-key/预置锚/指纹钉住）；额外文件/symlink 拒绝；rollback 重验；CycloneDX SBOM；`rosclaw release verify` | T5 攻击矩阵 8/8 |
| P1-1 resume 重复/遗漏 | **已修**：服务端 transcript projection + 分页 + latest_sequence；SSE afterSequence exactly-once；有界去重窗；同步路径补用户可见事件 | R4 测试 6/6；SSE node 测试 |
| P1-3 生命周期不关闭 | **已修**：`AgentService.open()` + lifespan finally close + token 文件随关闭删除 | lifecycle 测试 |
| P1-4 控制 token 可选 | **已修**：ephemeral token 全覆盖（除 /health 与 console 壳） | 401/200 端点测试 |
| P1-5 readiness 只看 socket 出现 | **已修**：`/v1/health/ready` 真实就绪 + gateway 轮询 + socket mode 验证 + 脱敏 stderr 尾 | gateway 11/11 |
| P1-6 凭据耐久性 | **已修**：O_NOFOLLOW/双 fsync/单链接校验/真实 quarantine | modeld 18/18 |
| P1-7 Node 非 required CI | **已修**：node-tui-unit、node-modeld-unit、cross-uid-operator-e2e、evidence-pack-verify 全部 required 且绿 | PR #225 CI 17/17 |
| P1-8 projection 省略参数扩权 | **已修**：无 mission_id 时按 peer owner 过滤 | owner filter 测试 |
| P1-9 revoke 只查 proof 存在 | **已修**：revoke 也要 Ed25519 验签（TOFU 钉住公钥） | socket 测试 |
| 测试数脱离 commit/命令 | **已修**：本报告与 PR 描述附 commit、命令、分项计数；CI 全部 job 绿 | 见 §4 |

## 2. 协议架构（Operator Decision Protocol v1）

```text
agentd ──proposal(+client_reference)──▶ rosclawd
operatord ◀──DecisionChallengeV1（nonce 同源）── rosclawd
operatord: 前台进程组校验 → /dev/tty 不可变卡片 → 显式 Y/N
operatord ──OperatorDecisionProofV1（Ed25519）──▶ rosclawd
rosclawd: registry(active?) → UID 匹配 → 验签 → challenge 逐字段比对
        → nonce 焚毁（持久化）→ 内部 arm/permit → 签 DecisionReceiptV1
operatord ──receipt──▶ agentd: daemon 公钥验签 + ACCEPT + 全字段匹配
        + 未过期 + 未重放 → broker 铸 grant（EXACT_ACTION 单次）
```

关键不变量：

- daemon 决定路径**没有 daemon-UID 直通**——同 UID 也要 proof；
  且调用方 UID 必须 == enrollment 登记的 operator UID。
- daemon 被读不泄露可伪造材料（只存公钥）。
- agentd 信任根 = daemon 签名 + 全字段精确匹配，不是字符串存在。
- SIM（无 daemon）剖面：operatord Ed25519 + agentd TOFU 钉住公钥，
  明确标记 DEV_SIM_ONLY。

## 3. 验证矩阵（commit 53a7a42，本机实测）

| 验证 | 命令 | 结果 |
|---|---|---|
| 全量回归 | `pytest -q --ignore=tests/agentd/test_kimi_live.py` | **6225 passed**, 82 skipped, 19 deselected；8 failed 全为已知环境基线（firstboot×4 + lerobot×4，origin/main 相同） |
| ruff | `ruff check src tests` | clean |
| CI（PR #224 + #225） | 17 checks | 全绿：Lint/Type/Test 3.11-3.13/Full Regression/Cross-UID Boundary/**Cross-UID Operator E2E**/Product/First Boot/Hub/Integration/Build/**ROS Docker**（45m 超时修复后）/**Node TUI**/**Node modeld**/**Evidence Pack Verify** |
| K0–K9 live | `pytest tests/agentd/test_kimi_live.py -m integration`（真实 Kimi K3） | 10/11；K1 注入抵抗断言套件并发下措辞波动（单独 4/4 通过；live LLM 行为抖动，非代码回归） |
| T1 四 UID e2e | docker（CI job） | 13/13（正向唯一成功，负向全 fail closed） |
| T2 PTY human | `pytest tests/operatord` | 8/8 |
| T5 发布攻击 | `pytest tests/agentd/test_release_packaging.py` | 14/14 |
| T6 证据包 | `pytest tests/agentd/test_evidence_pack.py` | 10/10 |
| LIMO 验收 | `run_acceptance(..., evidence_root=...)` | 12/12 + E3 签名证据包（signed=true，secret_scan_clean，`rosclaw evidence verify` VERIFIED） |

## 4. 已知环境性失败与 live 抖动（如实登记）

- firstboot×4、lerobot×4：本机环境（无 LeRobot、firstboot 子进程），
  origin/main 基线相同，与本两轮改动零交集。
- K1（live）：注入抵抗回复的措辞断言对模型非确定性敏感；
  代码路径未变，单独运行 4/4 通过。后续可把断言改为语义匹配。
- ROS Docker job：冷缓存 Humble 构建超时（15m job cap + 300s compose
  超时）——已修（45m + 1200s），现绿。

## 5. 诚实 deferred（与二次复核口径一致）

- **R5**：provider contract matrix（五家）、OAuth/device-code（现诚实
  501）、cost/finish-reason 归一 schema、OS keyring。
- **R8**：真 LIMO E4 SHADOW / E5 REAL——需物理 LIMO + 目标侧三进程
  部署（REAL 门禁因此继续关闭）。
- **T3** PTY/IME/中文输入/resize 矩阵与性能实测（首 delta p95 等）。
- **T4** provider 全矩阵 live canary；**T9** 故障注入与 24h soak。
- **P1-2** key-level approval overlay（当前为行级 y/n 快捷决定 +
  /approve 命令；卡片焦点 overlay 待做）。
- 同 UID 一体运行的 SIM 剖面信任根仍是 TOFU（已在报告与 doctor 中
  如实标记 DEV_SIM_ONLY）。

## 6. 证据位置

- 证据包：`/tmp/evidence-r7/acceptance/run_84d74722de1b4278a44da308/`
  （11 artifacts，signed，verify 通过）
- PR：#224（Gate S）、#225（R4/R6/R7/CI）；CI run 全绿。
- 本报告：`docs/IMPLEMENTATION_REPORT_GATE_S_R4-R7_2026-08-04.md`
