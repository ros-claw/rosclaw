我来检查 ROSClaw 的安装和 LeRobot 支持情况。先等待 MCP 服务器连接，并查看项目配置。

检查完成。所有调用均为只读（S0）状态查询，未对系统做任何修改。结果如下：

## ROSClaw 安装与 LeRobot 支持情况

**已安装**：ROSClaw **v1.0.1**（maturity: alpha），LeRobot Bridge **v1.0.1**，核心运行时可用（`core_runtime_available: true`），MCP 服务通过 stdio 正常连接。

**LeRobot 集成：架构上支持，但当前未配置就绪**
- `integrations.lerobot.state = not_configured`（`configured: false`）
- `body_linked = false` — 当前没有链接任何机器人身体
- 校准状态：`uncalibrated`（confidence 0.0）
- 技能列表为空（count: 0），即 **`rh56.single_step` 当前不可用**
- 黄金路径标记 `agent_ready: false`，支持等级为 **H1_CONTRACT_VERIFIED**（候选目标 H4_HARDWARE_ACTUATION_VERIFIED），状态备注 `developer_observed_revalidation_pending`（开发者观察过、待重新验证）

## 支持的机器人（Bodies）

| Body | 说明 |
|------|------|
| `inspire_rh56_left` | 因时 RH56 灵巧手（左） |
| `inspire_rh56_right` | 因时 RH56 灵巧手（右） |

## 支持的 Policy

- **`rosclaw_rh56_reference`** —— RH56 参考策略（唯一参考路径）

## 支持的运行模式

1. `proposal_only` —— 仅生成提案
2. `shadow` —— 影子模式（先验仿真验证，不动作硬件）
3. `single_step_receding_horizon` —— 单步滚动时域 REAL 执行（唯一真机模式，且必须经 `mcp.request_action` 提交）

## 明确不支持 / 限制

- ❌ `can_rh56_execution`（不支持 CAN 总线直接执行）
- ❌ `open_loop_action_chunks`（不支持开环动作块）
- ❌ `multiple_active_sessions`（不允许多个并发会话）
- ❌ `unattended_execution`（不允许无人值守执行）
- ❌ `direct_execution_allowed: false` —— Agent 不能直接执行，REAL 模式只能经 MCP `request_action` + 操作员 Permit

## 当前状态结论

系统是**已安装、框架支持 LeRobot，但尚未就绪**：要让 `rh56.single_step` 可用，需要操作员完成 LeRobot 集成配置、链接身体（提示命令为 `rosclaw body link-eurdf <profile_id>`）并完成校准。在此之前，任何真机动作请求都应被拒绝或仅走 `validate_trajectory` + 操作员确认路径。

检查完成。所有调用均为只读（S0）状态查询，未对系统做任何修改。结果如下：

## ROSClaw 安装与 LeRobot 支持情况

**已安装**：ROSClaw **v1.0.1**（maturity: alpha），LeRobot Bridge **v1.0.1**，核心运行时可用（`core_runtime_available: true`），MCP 服务通过 stdio 正常连接。

**LeRobot 集成：架构上支持，但当前未配置就绪**
- `integrations.lerobot.state = not_configured`（`configured: false`）
- `body_linked = false` — 当前没有链接任何机器人身体
- 校准状态：`uncalibrated`（confidence 0.0）
- 技能列表为空（count: 0），即 **`rh56.single_step` 当前不可用**
- 黄金路径标记 `agent_ready: false`，支持等级为 **H1_CONTRACT_VERIFIED**（候选目标 H4_HARDWARE_ACTUATION_VERIFIED），状态备注 `developer_observed_revalidation_pending`（开发者观察过、待重新验证）

## 支持的机器人（Bodies）

| Body | 说明 |
|------|------|
| `inspire_rh56_left` | 因时 RH56 灵巧手（左） |
| `inspire_rh56_right` | 因时 RH56 灵巧手（右） |

## 支持的 Policy

- **`rosclaw_rh56_reference`** —— RH56 参考策略（唯一参考路径）

## 支持的运行模式

1. `proposal_only` —— 仅生成提案
2. `shadow` —— 影子模式（先验仿真验证，不动作硬件）
3. `single_step_receding_horizon` —— 单步滚动时域 REAL 执行（唯一真机模式，且必须经 `mcp.request_action` 提交）

## 明确不支持 / 限制

- ❌ `can_rh56_execution`（不支持 CAN 总线直接执行）
- ❌ `open_loop_action_chunks`（不支持开环动作块）
- ❌ `multiple_active_sessions`（不允许多个并发会话）
- ❌ `unattended_execution`（不允许无人值守执行）
- ❌ `direct_execution_allowed: false` —— Agent 不能直接执行，REAL 模式只能经 MCP `request_action` + 操作员 Permit

## 当前状态结论

系统是**已安装、框架支持 LeRobot，但尚未就绪**：要让 `rh56.single_step` 可用，需要操作员完成 LeRobot 集成配置、链接身体（提示命令为 `rosclaw body link-eurdf <profile_id>`）并完成校准。在此之前，任何真机动作请求都应被拒绝或仅走 `validate_trajectory` + 操作员确认路径。