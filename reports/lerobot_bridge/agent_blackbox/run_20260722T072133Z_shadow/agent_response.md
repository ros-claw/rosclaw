我来按照 ROSClaw 的发现流程先了解当前运行时和身体状态，然后再做 SHADOW/仿真预演。

预演完成。**未执行任何真实硬件动作**（`hardware_actions_executed=0`，且 SHADOW 模式本身永不派发硬件命令）。以下是四部分报告：

## 1. 策略（Policy）

- **参考策略**：`rosclaw_rh56_reference`（LeRobot Bridge v1.0.1，golden path `rh56_single_step`，能力 `rh56.single_step`）
- **支持身体/模式**：`inspire_rh56_left` / `inspire_rh56_right`；`proposal_only`、`shadow`、`single_step_receding_horizon`
- **实际可用性**：该 golden path 当前 `agent_ready=false`（仅 H1_CONTRACT_VERIFIED，硬件执行验证待定），且 **LeRobot 集成未配置**（`not_configured`）、**无已注册技能**（`list_skills` 为空）、**无历史经验**（`query_memory` 为 0）
- **结论**：策略后端不可用，OK 手势轨迹由静态规划生成（假设 RH56 六关节归一化坐标 [小指， 无名指， 中指， 食指， 拇指屈曲， 拇指旋转]：三指伸展 = 0，食指 0.65，拇指屈曲 0.55，拇指旋转 0.5，从张开手 4 步插值到位），**非策略模型推理产物**

## 2. Sandbox

- `sandbox_run`（MuJoCo 单步）：❌ **不可用** — `RUNTIME_UNAVAILABLE`，"Sandbox initialized but no physics state was returned; MuJoCo model may be unavailable"
- `validate_trajectory`（4 点轨迹：张开 → OK）：✅ 通过，但仅为 **StaticPolicyValidation**（`simulation_executed=false`）
  - `is_safe=true`、`risk_score=0.0`、无碰撞、无违规、理由 "Within limits"
  - **即只完成静态限位检查，动力学级仿真预演未能执行**

## 3. 执行状态

- SHADOW 模式经 MCP `request_action` 提交，action_id `action_e939f562cc7c4e75a69ad3e7d1b45768`
- 状态转换：`PROPOSED` → **`BLOCKED`**（原因：`BODY_SNAPSHOT_REQUIRED` — 未链接身体，无法提供不可变身体快照哈希）
- 防护链按预期工作：无命令被派发，`verified=false`，`usable_for_real_execution=false`

## 4. 证据

| 项目 | 值 |
|---|---|
| 轨迹校验 replay | `sandbox://replay/dac7beaf04e0`（trace `05bd73d3-…`） |
| SHADOW 收据 | `action_e939f562cc7c4e75a69ad3e7d1b45768`（trace `6a283a56-…`），final_state=BLOCKED，evidence_level=REQUESTED |
| explain_execution | "SHADOW never dispatches hardware commands; it is a preview. hardware_actions_executed=0" |
| 环境根因 | `body_linked=false`、LeRobot `not_configured`、校准 `uncalibrated` |

**要完成一次真正的仿真预演，需要先**：① `rosclaw body link-eurdf <profile_id>` 链接 RH56 身体（解锁 body profile/快照哈希）；② 配置 LeRobot 集成；③ 恢复 MuJoCo 模型使 sandbox 可用。这几步都需要操作员在 Agent 外完成，我可以在此之后重跑完整链路。

预演完成。**未执行任何真实硬件动作**（`hardware_actions_executed=0`，且 SHADOW 模式本身永不派发硬件命令）。以下是四部分报告：

## 1. 策略（Policy）

- **参考策略**：`rosclaw_rh56_reference`（LeRobot Bridge v1.0.1，golden path `rh56_single_step`，能力 `rh56.single_step`）
- **支持身体/模式**：`inspire_rh56_left` / `inspire_rh56_right`；`proposal_only`、`shadow`、`single_step_receding_horizon`
- **实际可用性**：该 golden path 当前 `agent_ready=false`（仅 H1_CONTRACT_VERIFIED，硬件执行验证待定），且 **LeRobot 集成未配置**（`not_configured`）、**无已注册技能**（`list_skills` 为空）、**无历史经验**（`query_memory` 为 0）
- **结论**：策略后端不可用，OK 手势轨迹由静态规划生成（假设 RH56 六关节归一化坐标 [小指， 无名指， 中指， 食指， 拇指屈曲， 拇指旋转]：三指伸展 = 0，食指 0.65，拇指屈曲 0.55，拇指旋转 0.5，从张开手 4 步插值到位），**非策略模型推理产物**

## 2. Sandbox

- `sandbox_run`（MuJoCo 单步）：❌ **不可用** — `RUNTIME_UNAVAILABLE`，"Sandbox initialized but no physics state was returned; MuJoCo model may be unavailable"
- `validate_trajectory`（4 点轨迹：张开 → OK）：✅ 通过，但仅为 **StaticPolicyValidation**（`simulation_executed=false`）
  - `is_safe=true`、`risk_score=0.0`、无碰撞、无违规、理由 "Within limits"
  - **即只完成静态限位检查，动力学级仿真预演未能执行**

## 3. 执行状态

- SHADOW 模式经 MCP `request_action` 提交，action_id `action_e939f562cc7c4e75a69ad3e7d1b45768`
- 状态转换：`PROPOSED` → **`BLOCKED`**（原因：`BODY_SNAPSHOT_REQUIRED` — 未链接身体，无法提供不可变身体快照哈希）
- 防护链按预期工作：无命令被派发，`verified=false`，`usable_for_real_execution=false`

## 4. 证据

| 项目 | 值 |
|---|---|
| 轨迹校验 replay | `sandbox://replay/dac7beaf04e0`（trace `05bd73d3-…`） |
| SHADOW 收据 | `action_e939f562cc7c4e75a69ad3e7d1b45768`（trace `6a283a56-…`），final_state=BLOCKED，evidence_level=REQUESTED |
| explain_execution | "SHADOW never dispatches hardware commands; it is a preview. hardware_actions_executed=0" |
| 环境根因 | `body_linked=false`、LeRobot `not_configured`、校准 `uncalibrated` |

**要完成一次真正的仿真预演，需要先**：① `rosclaw body link-eurdf <profile_id>` 链接 RH56 身体（解锁 body profile/快照哈希）；② 配置 LeRobot 集成；③ 恢复 MuJoCo 模型使 sandbox 可用。这几步都需要操作员在 Agent 外完成，我可以在此之后重跑完整链路。