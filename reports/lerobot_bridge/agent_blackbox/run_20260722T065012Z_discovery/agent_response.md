子代理完成了只读检查。让我直接核验最关键的一项——项目快照中的 LeRobot 集成状态。

核验完毕。本次检查全程只读（读项目清单、遥测、canonical status.yaml / context.snapshot.json / dist-info)，未修改系统任何内容。以下是结论：

## ROSClaw 安装状态：已安装 ✅

| 项 | 状态 |
|---|---|
| 与本项目匹配的版本 | **rosclaw 1.0.1**(`/home/dell/rosclaw_eval/rt/venv`，与今日遥测 `rosclaw_version: 1.0.1` 一致，`agent` 命令 success) |
| 其他安装 | 主机上还有多个 1.0.0(`~/.local`、anaconda、`~/rosclaw-v1.0` venvs)——裸 `rosclaw` 在 PATH 上指向不唯一 |
| 运行环境 | Python 3.11/3.12,ROS 2 Humble 在位，CUDA 可用（RTX A6000) |
| MCP 接入 | `.mcp.json` 已配置 `rosclaw mcp serve`(stdio),19 个工具面已声明 |

## LeRobot 支持：桥接存在，但当前**未配置** ⚠️

- 架构层面支持：**LeRobot Bridge v1.0.1**，参考 policy 插件 `lerobot_policy_rosclaw_rh56 0.1.0` 已装入 runtime venv（要求 `lerobot>=0.6,<0.7`)。
- 实际状态（项目快照确认）:`integrations.lerobot = { configured: false, state: "not_configured", direct_execution_allowed: false }`。
- 原因：`lerobot` 包本身未装入 runtime venv，也未装入隔离 worker 环境 `~/.rosclaw/envs/lerobot`（该 env 只有 pip)。需运营方完成 setup 后集成才可用。

## 支持的机器人（LeRobot 参考路径）

| Body | 硬件 | 说明 |
|---|---|---|
| `inspire_rh56_left` | Inspire RH56 灵巧手，6 自由度，RS485 Modbus-RTU | 左手 rig(golden robot id `rh56_left_01`)，标定数据在位 |
| `inspire_rh56_right` | 同上 | 右手，标定数据在位 |

产品层面另有其他 golden path:`sim_ur5e`(H2 仿真验证）、`realsense_inspect`(H1 实验性）、`turtlesim_guarded_motion`（未验证）、`mobile_base_deadman`(H2 仿真验证）。**当前项目 `robot_id = null`，未检测到机器人。**

## Policy 与运行模式

- **唯一参考 policy:** `rosclaw_rh56_reference`
- **支持模式：** `proposal_only`（仅提案）、`shadow`（影子模式）、`single_step_receding_horizon`（单步 REAL)
- **REAL 提交入口：** 仅通过 MCP `request_action`（经 rosclawd)，禁止任何直连

## 主要限制

1. **P0 MCP 无真实执行器**:`no_real_execution: true`,REAL 请求在无已验证执行器时必须拒绝。
2. **执行链未就绪**:`real_robot_execution` 组件状态为 `revalidation_pending`;`rh56_single_step` golden path 仅达 H1 契约验证，agent 黑盒路径未跑过，`agent_ready: false`。
3. **Bridge v1.0.1 范围外**：训练、DAgger、reward model、Hub 发布、任意 policy 映射均不支持。
4. **禁止项**：不得运行 `rosclaw lerobot rollout execute`；不得直接打开串口/CAN 设备；不得创建或批准 Permit;REAL 前必须先 SHADOW；不支持开环 action chunk、多活动会话、无人值守执行、运动中 USB 热插拔（RH56 无 CAN 执行路径，fail-closed)。
5. **安全链在位**:`validate_before_motion`、`emergency_stop`（S4）可用；要求 `require_sim_before_real` 与 firewall 模式。

**一句话总结：** ROSClaw 1.0.1 已安装且声明了 LeRobot Bridge v1.0.1 支持（RH56 左右手 + `rosclaw_rh56_reference` policy + 三种模式），但当前集成状态为 `not_configured`(worker 环境缺 `lerobot` 包）、无机器人连接、无真实执行器——现阶段只能做只读状态查询与仿真/验证类操作。

附注：本会话中 ROSClaw MCP 工具未实际加载到工具面（服务器已配置但工具未暴露），以上状态取自其 canonical 磁盘来源（`status.yaml`、`context.snapshot.json`、dist-info、遥测），与 `get_product_status`/`get_runtime_status` 等工具报告的内容一致。

核验完毕。本次检查全程只读（读项目清单、遥测、canonical status.yaml / context.snapshot.json / dist-info)，未修改系统任何内容。以下是结论：

## ROSClaw 安装状态：已安装 ✅

| 项 | 状态 |
|---|---|
| 与本项目匹配的版本 | **rosclaw 1.0.1**(`/home/dell/rosclaw_eval/rt/venv`，与今日遥测 `rosclaw_version: 1.0.1` 一致，`agent` 命令 success) |
| 其他安装 | 主机上还有多个 1.0.0(`~/.local`、anaconda、`~/rosclaw-v1.0` venvs)——裸 `rosclaw` 在 PATH 上指向不唯一 |
| 运行环境 | Python 3.11/3.12,ROS 2 Humble 在位，CUDA 可用（RTX A6000) |
| MCP 接入 | `.mcp.json` 已配置 `rosclaw mcp serve`(stdio),19 个工具面已声明 |

## LeRobot 支持：桥接存在，但当前**未配置** ⚠️

- 架构层面支持：**LeRobot Bridge v1.0.1**，参考 policy 插件 `lerobot_policy_rosclaw_rh56 0.1.0` 已装入 runtime venv（要求 `lerobot>=0.6,<0.7`)。
- 实际状态（项目快照确认）:`integrations.lerobot = { configured: false, state: "not_configured", direct_execution_allowed: false }`。
- 原因：`lerobot` 包本身未装入 runtime venv，也未装入隔离 worker 环境 `~/.rosclaw/envs/lerobot`（该 env 只有 pip)。需运营方完成 setup 后集成才可用。

## 支持的机器人（LeRobot 参考路径）

| Body | 硬件 | 说明 |
|---|---|---|
| `inspire_rh56_left` | Inspire RH56 灵巧手，6 自由度，RS485 Modbus-RTU | 左手 rig(golden robot id `rh56_left_01`)，标定数据在位 |
| `inspire_rh56_right` | 同上 | 右手，标定数据在位 |

产品层面另有其他 golden path:`sim_ur5e`(H2 仿真验证）、`realsense_inspect`(H1 实验性）、`turtlesim_guarded_motion`（未验证）、`mobile_base_deadman`(H2 仿真验证）。**当前项目 `robot_id = null`，未检测到机器人。**

## Policy 与运行模式

- **唯一参考 policy:** `rosclaw_rh56_reference`
- **支持模式：** `proposal_only`（仅提案）、`shadow`（影子模式）、`single_step_receding_horizon`（单步 REAL)
- **REAL 提交入口：** 仅通过 MCP `request_action`（经 rosclawd)，禁止任何直连

## 主要限制

1. **P0 MCP 无真实执行器**:`no_real_execution: true`,REAL 请求在无已验证执行器时必须拒绝。
2. **执行链未就绪**:`real_robot_execution` 组件状态为 `revalidation_pending`;`rh56_single_step` golden path 仅达 H1 契约验证，agent 黑盒路径未跑过，`agent_ready: false`。
3. **Bridge v1.0.1 范围外**：训练、DAgger、reward model、Hub 发布、任意 policy 映射均不支持。
4. **禁止项**：不得运行 `rosclaw lerobot rollout execute`；不得直接打开串口/CAN 设备；不得创建或批准 Permit;REAL 前必须先 SHADOW；不支持开环 action chunk、多活动会话、无人值守执行、运动中 USB 热插拔（RH56 无 CAN 执行路径，fail-closed)。
5. **安全链在位**:`validate_before_motion`、`emergency_stop`（S4）可用；要求 `require_sim_before_real` 与 firewall 模式。

**一句话总结：** ROSClaw 1.0.1 已安装且声明了 LeRobot Bridge v1.0.1 支持（RH56 左右手 + `rosclaw_rh56_reference` policy + 三种模式），但当前集成状态为 `not_configured`(worker 环境缺 `lerobot` 包）、无机器人连接、无真实执行器——现阶段只能做只读状态查询与仿真/验证类操作。

附注：本会话中 ROSClaw MCP 工具未实际加载到工具面（服务器已配置但工具未暴露），以上状态取自其 canonical 磁盘来源（`status.yaml`、`context.snapshot.json`、dist-info、遥测），与 `get_product_status`/`get_runtime_status` 等工具报告的内容一致。