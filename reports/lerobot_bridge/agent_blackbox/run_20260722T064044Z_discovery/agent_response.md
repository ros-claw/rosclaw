子代理完成了只读文件系统检查。我直接核实一下关键的上下文快照文件，确保报告内容准确。

检查完成，未对系统做任何修改（仅只读查询）。说明一下方法：MCP 服务器已连接，但本会话中 `mcp__rosclaw__*` 工具未出现在可调用列表里，因此我改用只读文件系统核查（`.rosclaw/agent/context.snapshot.json`、遥测日志、SKILL.md），以下为核实后的结果。

## 结论：已安装，支持 LeRobot，但集成尚未配置

| 项目 | 状态 |
|---|---|
| ROSClaw | **已安装，v1.0.1**（CLI 1.0.1，今日 06:40 UTC 有成功运行记录） |
| LeRobot Bridge | **v1.0.1**（作为 ROSClaw 内置 policy backend） |
| LeRobot 集成状态 | **`not_configured`**（`configured: false`，`direct_execution_allowed: false`） |
| 默认机器人 | 无（`robot_id: null`） |

环境：Linux x86_64、Python 3.11、ROS 2 Humble、CUDA 可用（RTX A6000）。

## 支持的机器人（bodies）

- `inspire_rh56_left` — 因时 RH56 灵巧手（左）
- `inspire_rh56_right` — 因时 RH56 灵巧手（右）

## 支持的 policy

- `rosclaw_rh56_reference` —— 唯一参考策略路径，v1.0.1 不支持任意 policy 到机器人的映射。

## 运行模式

1. **proposal_only** — 仅提案，不执行
2. **shadow (SHADOW)** — 影子模式，REAL 前必须先走 SHADOW
3. **single_step_receding_horizon** — 单步 REAL（滚动时域），只能经 MCP `request_action`（S3）提交

## 主要限制

- **P0 无真实执行工具**（`no_real_execution: true`）：Agent 不能直接驱动硬件，REAL 动作只能提交给 rosclawd 等待授权；Agent **不得自行签发或批准 Permit**，遇 `AUTHORIZATION_REQUIRED` 必须停止并说明。
- `validate_trajectory`（S2）通过 ≠ 执行授权；`sandbox_run`（S1）仅 MuJoCo 仿真。
- **LeRobot 集成当前未配置**：需操作员先运行 `rosclaw setup lerobot --reference-policy rh56` 和 `rosclaw lerobot doctor --json`（这是 operator 工作流，我未执行）。
- **v1.0.1 明确不支持**：训练、DAgger、reward model、Hub 发布、任意 policy 映射、**CAN 方式驱动 RH56**（fail-closed）、开环 chunk、无人值守执行。
- 禁止：从 Agent 进程运行 `rosclaw lerobot rollout execute`、直接打开串口/CAN 设备、直接发布 ROS topic（如 `/cmd_vel`）。

## 两点无法确认

- **`rh56.single_step` 技能可用性**：`list_skills` 工具本会话不可调用，磁盘上也没有技能注册表，无法确认。
- **校准状态**：`get_calibration_status` 同样不可调用，无本地数据。

如需实时确认这两项，需要在 MCP 工具可调用的会话中运行 Discovery 流程（`get_product_status` → `get_runtime_status` → `get_body_profile` → `get_body_state` → `get_calibration_status`）。

检查完成，未对系统做任何修改（仅只读查询）。说明一下方法：MCP 服务器已连接，但本会话中 `mcp__rosclaw__*` 工具未出现在可调用列表里，因此我改用只读文件系统核查（`.rosclaw/agent/context.snapshot.json`、遥测日志、SKILL.md），以下为核实后的结果。

## 结论：已安装，支持 LeRobot，但集成尚未配置

| 项目 | 状态 |
|---|---|
| ROSClaw | **已安装，v1.0.1**（CLI 1.0.1，今日 06:40 UTC 有成功运行记录） |
| LeRobot Bridge | **v1.0.1**（作为 ROSClaw 内置 policy backend） |
| LeRobot 集成状态 | **`not_configured`**（`configured: false`，`direct_execution_allowed: false`） |
| 默认机器人 | 无（`robot_id: null`） |

环境：Linux x86_64、Python 3.11、ROS 2 Humble、CUDA 可用（RTX A6000）。

## 支持的机器人（bodies）

- `inspire_rh56_left` — 因时 RH56 灵巧手（左）
- `inspire_rh56_right` — 因时 RH56 灵巧手（右）

## 支持的 policy

- `rosclaw_rh56_reference` —— 唯一参考策略路径，v1.0.1 不支持任意 policy 到机器人的映射。

## 运行模式

1. **proposal_only** — 仅提案，不执行
2. **shadow (SHADOW)** — 影子模式，REAL 前必须先走 SHADOW
3. **single_step_receding_horizon** — 单步 REAL（滚动时域），只能经 MCP `request_action`（S3）提交

## 主要限制

- **P0 无真实执行工具**（`no_real_execution: true`）：Agent 不能直接驱动硬件，REAL 动作只能提交给 rosclawd 等待授权；Agent **不得自行签发或批准 Permit**，遇 `AUTHORIZATION_REQUIRED` 必须停止并说明。
- `validate_trajectory`（S2）通过 ≠ 执行授权；`sandbox_run`（S1）仅 MuJoCo 仿真。
- **LeRobot 集成当前未配置**：需操作员先运行 `rosclaw setup lerobot --reference-policy rh56` 和 `rosclaw lerobot doctor --json`（这是 operator 工作流，我未执行）。
- **v1.0.1 明确不支持**：训练、DAgger、reward model、Hub 发布、任意 policy 映射、**CAN 方式驱动 RH56**（fail-closed）、开环 chunk、无人值守执行。
- 禁止：从 Agent 进程运行 `rosclaw lerobot rollout execute`、直接打开串口/CAN 设备、直接发布 ROS topic（如 `/cmd_vel`）。

## 两点无法确认

- **`rh56.single_step` 技能可用性**：`list_skills` 工具本会话不可调用，磁盘上也没有技能注册表，无法确认。
- **校准状态**：`get_calibration_status` 同样不可调用，无本地数据。

如需实时确认这两项，需要在 MCP 工具可调用的会话中运行 Discovery 流程（`get_product_status` → `get_runtime_status` → `get_body_profile` → `get_body_state` → `get_calibration_status`）。