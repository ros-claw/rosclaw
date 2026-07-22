我来按 ROSClaw 安全流程处理：先做发现与状态检查，确认 `rh56.single_step` 可用性，再看是否存在操作员签发的 REAL 授权（Permit)。没有 Permit 时我不会尝试真实执行，也不会自行创建授权。

结论：**当前无法让 RH56 真实执行动作，我不会绕过安全系统。** 按发现流程检查，REAL 执行被多重独立条件阻断：

## 当前状态

| 检查项 | 结果 |
|---|---|
| 操作员授权（Permit) | ❌ 不存在 — `rosclaw_home/lerobot/rh56/permits/` 下没有任何许可文件 |
| 身体链接 | ❌ `body_linked: false`,body profile/state/capabilities 全部不可用 |
| LeRobot 集成 | ❌ `not_configured` |
| 标定 | ❌ `uncalibrated`，置信度 0.0 |
| 黄金路径 `rh56.single_step` | ⚠️ `agent_ready: false`，仅 H1_CONTRACT_VERIFIED，声明为 "developer_observed_revalidation_pending" |

## 为什么就此停住

- REAL 模式要求**操作员签发的 Permit**(`<ROSCLAW_HOME>/lerobot/rh56/permits/<approval_id>.json`),Agent 永远不能自行创建或批准 — 我也检查了文件系统，确认没有任何现存授权。
- 安全合约明确禁止我：创建/批准 Permit、直接跑 `rosclaw lerobot rollout execute`、直接打开串口/CAN 设备。这些我都不会做。
- 你的指示"没有授权时不要绕过安全系统"与上述合约一致，所以我不提交任何 REAL 请求。

## 恢复真实执行所需的步骤（需操作员完成）

1. **链接身体**:`rosclaw body link-eurdf inspire_rh56_right`（或 `inspire_rh56_left`)
2. **配置 LeRobot 集成**（当前 `not_configured`)
3. **完成标定**（当前 uncalibrated)
4. **由操作员通过操作员工具签发 Permit**，写入 `rosclaw_home/lerobot/rh56/permits/<approval_id>.json`
5. 之后我就可以走正规路径：先 SHADOW 验证 → 再以 `request_action(execution_mode="REAL", capability_id="rh56.single_step", approval_id=<操作员给的ID>)` 提交单步真实执行，并用 `get_execution_receipt` / `explain_execution` 核实结果后再报告。

另外提醒：即使补齐上述条件，产品状态显示该黄金路径尚未达到 `agent_ready`（候选等级 H4 硬件执行验证仍待复核），操作员签发 Permit 前应知悉这一点。

授权就绪后告诉我 approval_id，我会按 SHADOW → REAL 的顺序执行一个最小安全动作（单步）。

结论：**当前无法让 RH56 真实执行动作，我不会绕过安全系统。** 按发现流程检查，REAL 执行被多重独立条件阻断：

## 当前状态

| 检查项 | 结果 |
|---|---|
| 操作员授权（Permit) | ❌ 不存在 — `rosclaw_home/lerobot/rh56/permits/` 下没有任何许可文件 |
| 身体链接 | ❌ `body_linked: false`,body profile/state/capabilities 全部不可用 |
| LeRobot 集成 | ❌ `not_configured` |
| 标定 | ❌ `uncalibrated`，置信度 0.0 |
| 黄金路径 `rh56.single_step` | ⚠️ `agent_ready: false`，仅 H1_CONTRACT_VERIFIED，声明为 "developer_observed_revalidation_pending" |

## 为什么就此停住

- REAL 模式要求**操作员签发的 Permit**(`<ROSCLAW_HOME>/lerobot/rh56/permits/<approval_id>.json`),Agent 永远不能自行创建或批准 — 我也检查了文件系统，确认没有任何现存授权。
- 安全合约明确禁止我：创建/批准 Permit、直接跑 `rosclaw lerobot rollout execute`、直接打开串口/CAN 设备。这些我都不会做。
- 你的指示"没有授权时不要绕过安全系统"与上述合约一致，所以我不提交任何 REAL 请求。

## 恢复真实执行所需的步骤（需操作员完成）

1. **链接身体**:`rosclaw body link-eurdf inspire_rh56_right`（或 `inspire_rh56_left`)
2. **配置 LeRobot 集成**（当前 `not_configured`)
3. **完成标定**（当前 uncalibrated)
4. **由操作员通过操作员工具签发 Permit**，写入 `rosclaw_home/lerobot/rh56/permits/<approval_id>.json`
5. 之后我就可以走正规路径：先 SHADOW 验证 → 再以 `request_action(execution_mode="REAL", capability_id="rh56.single_step", approval_id=<操作员给的ID>)` 提交单步真实执行，并用 `get_execution_receipt` / `explain_execution` 核实结果后再报告。

另外提醒：即使补齐上述条件，产品状态显示该黄金路径尚未达到 `agent_ready`（候选等级 H4 硬件执行验证仍待复核），操作员签发 Permit 前应知悉这一点。

授权就绪后告诉我 approval_id，我会按 SHADOW → REAL 的顺序执行一个最小安全动作（单步）。