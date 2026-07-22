请求在正确通道被 **BLOCKED** 了。让我取一下该回执的官方解释，然后停止并汇报。

已停止。REAL 请求通过正确通道（MCP `request_action`）提交后被安全系统拒绝，**没有执行任何硬件动作**(`hardware_actions_executed=0`)，我没有、也不会绕过这些关卡。

## 执行摘要

| 步骤 | 结果 |
|------|------|
| 发现（product/runtime/body/calibration) | 完成，发现多项前置条件缺失 |
| Permit 目录检查 | 无任何操作员签发的 Permit |
| `request_action` REAL(`rh56.single_step` / `inspire_rh56_right`) | **BLOCKED** — `BODY_SNAPSHOT_REQUIRED` |
| 回执 `action_4d66ac4f491a42aaacbd2028476172c7` | final_state=BLOCKED,trust=UNVERIFIED,usable_for_real_execution=false |

## 缺少的授权与前置条件（按阻断顺序）

1. **身体未链接（最直接的阻断）**:`body_linked: false`。SHADOW/REAL 动作要求不可变的 body snapshot hash，而快照只能来自已链接的身体。需由操作员运行 `rosclaw body link-eurdf <profile_id>`。
2. **LeRobot 集成未配置**:`lerobot.state = not_configured`,golden path `rh56_single_step` 标记为 `agent_ready: false`。
3. **未标定**：校准状态 `uncalibrated`,confidence 0.0。
4. **操作员 Permit(REAL 的硬性授权）**:REAL 模式要求 `~/.rosclaw/lerobot/rh56/permits/<approval_id>.json`，必须由**人类操作员**签发。当前该目录不存在任何 Permit，且安全契约明确规定 Agent 永远不能自行创建或批准 Permit——所以即使前 3 项补齐，没有操作员签发的 Permit,REAL 仍会被拒。

## 需要谁做什么

只有人类操作员能解锁：链接身体 → 配置 LeRobot worker → 完成标定 → 先用 SHADOW 验证 → 最后签发一次性 Permit 并把 `approval_id` 交给我。拿到有效的 `approval_id` 后，我可以通过同一通道重新提交。

已停止。REAL 请求通过正确通道（MCP `request_action`）提交后被安全系统拒绝，**没有执行任何硬件动作**(`hardware_actions_executed=0`)，我没有、也不会绕过这些关卡。

## 执行摘要

| 步骤 | 结果 |
|------|------|
| 发现（product/runtime/body/calibration) | 完成，发现多项前置条件缺失 |
| Permit 目录检查 | 无任何操作员签发的 Permit |
| `request_action` REAL(`rh56.single_step` / `inspire_rh56_right`) | **BLOCKED** — `BODY_SNAPSHOT_REQUIRED` |
| 回执 `action_4d66ac4f491a42aaacbd2028476172c7` | final_state=BLOCKED,trust=UNVERIFIED,usable_for_real_execution=false |

## 缺少的授权与前置条件（按阻断顺序）

1. **身体未链接（最直接的阻断）**:`body_linked: false`。SHADOW/REAL 动作要求不可变的 body snapshot hash，而快照只能来自已链接的身体。需由操作员运行 `rosclaw body link-eurdf <profile_id>`。
2. **LeRobot 集成未配置**:`lerobot.state = not_configured`,golden path `rh56_single_step` 标记为 `agent_ready: false`。
3. **未标定**：校准状态 `uncalibrated`,confidence 0.0。
4. **操作员 Permit(REAL 的硬性授权）**:REAL 模式要求 `~/.rosclaw/lerobot/rh56/permits/<approval_id>.json`，必须由**人类操作员**签发。当前该目录不存在任何 Permit，且安全契约明确规定 Agent 永远不能自行创建或批准 Permit——所以即使前 3 项补齐，没有操作员签发的 Permit,REAL 仍会被拒。

## 需要谁做什么

只有人类操作员能解锁：链接身体 → 配置 LeRobot worker → 完成标定 → 先用 SHADOW 验证 → 最后签发一次性 Permit 并把 `approval_id` 交给我。拿到有效的 `approval_id` 后，我可以通过同一通道重新提交。