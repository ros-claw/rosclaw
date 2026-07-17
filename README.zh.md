<div align="center">

# ROSClaw

### 面向具身 Agent 的可信物理执行运行时与控制平面

**将动作绑定到身体，失败时关闭，凭证据执行，并返回可审计回执。**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python)](https://www.python.org/)
[![ROS 2](https://img.shields.io/badge/ROS_2-Humble_|_Jazzy-FF3E00?logo=ros)](https://docs.ros.org/)
[![Simulation](https://img.shields.io/badge/Verified_Simulation-MuJoCo-black?logo=mujoco)](https://mujoco.org/)
[![MCP](https://img.shields.io/badge/Protocol-MCP-8A2BE2)](https://modelcontextprotocol.io/)
[![Status](https://img.shields.io/badge/Status-Alpha-orange)](https://github.com/ros-claw/rosclaw)

[English](README.md) • **中文** • [架构](ARCHITECTURE.md) • [快速开始](QUICKSTART.md) • [文档](docs/)

</div>

```bash
curl -sSL https://rosclaw.io/get | bash
rosclaw firstboot --yes --profile offline --no-telemetry
rosclaw doctor --level verified
```

---

## ROSClaw 是什么？

ROSClaw 不是另一个 Agent Framework，不是 ROS 2 的替代品，也不是简单的“大模型调用 ROS”工具。

ROSClaw 是**面向具身 Agent 的可信物理执行运行时与控制平面**。Codex、Claude Code、OpenClaw、VLA 服务等是可替换的北向客户端；ROS 2、MCP、厂商 SDK、仿真器和机器人控制器是南向系统。

它的统一动作路径把意图绑定到 Body 和 Capability，执行策略与授权检查，仲裁物理资源，向 Driver 分发，并返回带证据等级的 `ExecutionReceipt`。Memory 和自进化模块异步消费这些证据，不能代替执行证据。

### 当前成熟度

ROSClaw 当前是 Alpha 软件。下表描述真实能力边界，不用路线图冒充已完成功能。

| 范围 | 状态 | 当前证据 |
|---|---|---|
| UR5e 桌面 Reach | **仿真已验证** | 真 MuJoCo 模型和 `mj_step`、笛卡尔成功条件、碰撞/越界阻止、轨迹 Artifact、Trace、Receipt。 |
| Action Contract / Gateway | **组件与系统验证** | 版本化动作/回执、证据等级、Action ID 幂等、Body 独占租约、Executor 缺失时失败关闭。 |
| E-Stop 控制路径 | **组件验证** | Fan-out、超时、部分 ACK、幂等、锁存、物理观测字段；本环境未验证硬件急停。 |
| Mock Sense、Mock Provider、Fixture Driver | **仅 Fixture** | 明确标记 `FIXTURE` / `SYNTHETIC`，不能用于安全判断或验收。 |
| RH56 LeRobot 单步闭环 | **Fixture 已验证** | 必须显式 `--fixture`；验证合成 Modbus 反馈、Permit/Hash/Watchdog，且 `hardware_actions_executed=0`；Serial backend 仍是 stub。 |
| ROS Connector、LeRobot、Hardware MCP、真实 Provider | **实验性** | 契约和组件覆盖不等；注册成功或能 import 不代表可执行。 |
| ROS 2 Turtlesim 运动闭环 | **本环境未验证** | 当前验证环境没有安装 ROS 2。 |
| 真实机器人执行 | **未运行** | 必须按本体验证 Driver ACK 和物理反馈。 |

核心包支持 Python 3.11+。隔离的 LeRobot 0.6 运行时和内置 RH56
参考 policy 插件需要 Python 3.12+。

### 执行模式

| Mode | 含义 |
|---|---|
| `FIXTURE` | 用户显式启用的合成数据；永远不算 verified，不能进入真实执行验收。 |
| `DRY_RUN` | Schema / 静态策略检查；没有物理仿真和分发。 |
| `REPLAY` | 已记录证据的回放；不会产生新的物理副作用。 |
| `SIMULATION` | 物理引擎执行；当前已验证 UR5e Reach。 |
| `SHADOW` | 契约已定义，尚无通过验收的黄金闭环。 |
| `REAL` | 契约已定义并失败关闭；当前不声明仓库级真实硬件可用。 |

---

## 为什么物理 AI 需要运行时基础设施？

今天的大模型已经很会推理、写代码、规划任务，但它们本质上还活在“文字和 Token 世界”里。

如果让一个大模型直接控制机器人，它会遇到很多问题：

- 不知道机器人身体的真实限制；
- 不知道机械臂能转多少度；
- 不知道某个动作会不会撞到桌子；
- 不知道上一次抓取为什么失败；
- 不知道失败之后该如何恢复；
- 也不知道如何把失败经验变成新的技能。

**物理世界不是聊天窗口。** 它有重力、摩擦、碰撞、延迟、传感噪声、力矩限制、关节限制和安全边界。ROSClaw 要做的，就是把大模型的认知能力真正接到物理世界中。

---

## 运行时闭环

```text
动作意图 → Body/Capability → Policy/Authorization → Resource Lease
         → Dispatch/ACK → 物理观测 → 任务验证 → Execution Receipt

Receipt → Trace/Practice → Memory/How/Auto/Darwin（异步）
```

> **请求不等于执行，分发不等于完成，完成必须有证据。**

Auto 可以提出改变，但不能独自批准改变。沙盒验证、Darwin 评测、晋升门和人类确认共同决定改变是否能进入真实世界。

---

## 首次具身 / 快速开始

安装 CLI 并运行交互式首次启动向导：

```bash
curl -sSL https://rosclaw.io/get | bash
rosclaw firstboot
rosclaw doctor --level verified
```

无需硬件即可运行本地仿真演示：

```bash
rosclaw sandbox run --robot sim_ur5e --world tabletop --task reach
```

无头或 CI 环境：

```bash
rosclaw firstboot --yes --profile offline --no-telemetry
```

查看 [QUICKSTART.md](QUICKSTART.md) 获取四条路径的详细指南：本地仿真、Agent 集成、机器人本体设置和开发者设置。

---

## 核心运行时模块

| 模块 | 职责 |
|---|---|
| **Runtime** | 生命周期、配置、插件注册、依赖注入。 |
| **EventBus** | 模块通信、主题路由、Trace 关联。 |
| **Provider** | 能力路由、Schema 约束、安全边界。 |
| **Sandbox** | 安全验证、防火墙、MuJoCo 预演。 |
| **Practice** | 时间线捕获、MCAP、JSONL、执行记录。 |
| **Memory** | 经验图谱、失败/成功模式、记忆召回。 |
| **Know** | TaskCard、Pattern、EvidenceTrace、故障分类。 |
| **How** | 运行时干预、证据引用、最小修复建议。 |
| **Auto** | 提案、补丁、实验、Champion、DeadEnd 跟踪。 |
| **Darwin** | 多种子基准测试、压力场景、回归评测。 |
| **Skill Registry** | 版本、血缘、Champion、回滚。 |
| **Dashboard** | 可观测性、进化轨迹、血缘可视化。 |

---

## Hub 与资产

ROSClaw Hub 是一个物理 AI 资产中心，用于管理技能、Provider、硬件 MCP 服务器、数字孪生和认知 Wiki。资产可以完全本地化，也可以与注册表同步。

支持的资产类型：

- `skill` — 可复用的物理 AI 技能
- `provider` — 运行时能力 Provider
- `hardware_mcp` — 封装真实硬件的 MCP 服务器
- `digital_twin` — 仿真资产 / e-URDF 孪生
- `cognitive_wiki` — 结构化运维知识

```bash
rosclaw hub validate tests/fixtures/hub_assets/hardware_mcp_valid/manifest.yaml
rosclaw hub search g1
rosclaw hub install rosclaw://hardware_mcp/rosclaw/unitree-g1@1.0.0 --yes
```

详见 [docs/ASSETS.md](docs/ASSETS.md) 和 [docs/hub/README.md](docs/hub/README.md)。

---

## 硬件 MCP 自动接入

ROSClaw 可以从声明式 manifest 自动安装硬件 MCP Server，并持续检查其健康状态。Manifest 按以下顺序解析：

1. 本地缓存 `~/.rosclaw/mcp/cache/`。
2. 内置离线 registry（`unitree-g1`、`realsense-d455` 等）。
3. 远程 ROSClaw Hub：`https://www.rosclaw.io/api/registry`。

网络不可用时自动回退到本地缓存和内置 registry。

### 快速开始

```bash
# 预览安装计划（不写任何文件）
./rosclaw mcp install unitree-g1 --dry-run --offline

# 离线安装内置硬件 MCP
./rosclaw mcp install unitree-g1 --offline

# 预览公共 Hub 包（去掉 --offline 即可从网络获取）
./rosclaw mcp install ros-claw/g1-mcp --dry-run --offline

# 使用自定义/私有 Hub 端点
ROSCLAW_MCP_HUB=https://my-hub.example.com ./rosclaw mcp install ros-claw/g1-mcp --dry-run --offline

# 列出已安装/可用 Server
./rosclaw mcp list --offline
./rosclaw mcp list --offline --json

# 健康检查
./rosclaw mcp health
./rosclaw mcp health unitree-g1
./rosclaw mcp health unitree-g1 --full --json
```

`--dry-run` 可以在写入前查看解析出的 manifest、版本、artifact、body patch、权限和 Claude `.mcp.json` 合并计划；`--offline` 强制只使用本地缓存和内置 registry。

完整生命周期、状态文件、权限和排错指南见 `docs/HARDWARE_MCP_ONBOARDING.md`。

---

## 为任意 Agent 安装并配置 ROSClaw

把下面这段 setup prompt 粘贴给 Codex、Claude Code、OpenClaw 或其他支持 MCP 的 Agent：

> 为这个仓库安装并配置 ROSClaw Agent 集成。请运行
> `rosclaw agent install --project-root . --skip-secrets`，然后阅读
> `ROSCLAW.md`、`AGENTS.md` 和 `.agents/skills/rosclaw/SKILL.md`。再用
> `rosclaw agent test universal --project-root . --quick --mcp-probe` 验证配置。
> 之后通过 ROSClaw 的 CLI 和 MCP tools 使用机器人状态、技能、记忆、
> sandbox 仿真、practice 记录和安全检查能力。

如果你自己手动操作，核心安装命令是：

```bash
rosclaw agent install --project-root . --skip-secrets
```

这会安装并配置 ROSClaw 面向 Agent 的集成文件：`.mcp.json`、`AGENTS.md`、
`ROSCLAW.md`、`CLAUDE.md`、`.agents/skills/rosclaw/SKILL.md` 和
`.rosclaw/agent/context.snapshot.json`。

它不会安装 ROSClaw 本体，也不是把 ROSClaw 作为某个 Agent 框架的原生插件安装进去。
ROSClaw 仍然是独立的 CLI、Python package、MCP server 和机器人基础设施层；这个命令的作用是让 Agent harness 能发现并使用它。

---

## 将 Practice 持久化到 SeekDB

本地开发可使用 SQLite；真实 SeekDB/OceanBase server 使用 MySQL-compatible DSN：

未提供 `--data-root` 时，ROSClaw 优先使用 `ROSCLAW_PRACTICE_DATA_ROOT`，
否则使用 `$ROSCLAW_HOME/data/practice`（默认为
`~/.rosclaw/data/practice`）。容器部署可以显式将该环境变量设置为
`/data/rosclaw/practice`。

```bash
# 本地文件
rosclaw practice ingest-seekdb <practice_id> \
  --seekdb-path ~/.rosclaw/memory/seekdb.sqlite

# 真实 SeekDB server
rosclaw practice ingest-seekdb <practice_id> \
  --seekdb-url mysql://root@127.0.0.1:2881/rosclaw

rosclaw practice query failures \
  --robot-id rh56 \
  --seekdb-url mysql://root@127.0.0.1:2881/rosclaw \
  --json
```

SeekDB 的 `2881` 端口使用 MySQL-compatible SQL 协议，不是 HTTP API。
重复执行 Practice ingest 会按 episode 和 evidence ID 幂等更新。

---

## 安全模型

ROSClaw 的核心安全规则：

> **任何模型输出都不应该直接控制机器人。**

新的统一物理动作路径是：

1. Provider 生成结构化的动作提案。
2. Sandbox / Firewall 根据有效本体模型和安全策略检查。
3. 决策结果为 `ALLOW`、`MODIFY`、`BLOCK` 或 `REQUIRE_HUMAN_CONFIRMATION`。
4. `ActionGateway` 获取独占资源租约并分发 Executor。
5. Driver ACK、观测、验证、Trace 和 Artifact 被组装为 Receipt。
6. Practice、Memory、How、Auto、Darwin 异步消费 Receipt。

历史执行适配器仍在逐步迁移到 Action Gateway。当前不能假定每条旧 CLI、Skill、ROS Connector 或厂商路径都已经不可绕过。`REAL` 部署必须通过本体专项验收，并将南向凭证隔离在 Agent 进程之外。

已知的 MCPHub 低层动作、独立 UR5 MCP 运动工具和 ROS Connector capability 执行现在都会失败关闭，不再直接下发。执行器迁移期间，ROS Connector 仍支持发现、验证、显式 dry-run，以及如实返回证据的急停请求。

ROSClaw 是研究基础设施，不能替代经过认证的工业安全系统。务必先在仿真中测试，保持急停系统可用，并在人类监督下运行。

完整安全模型见 [docs/SAFETY.md](docs/SAFETY.md)。

---

## 文档

- [QUICKSTART.md](QUICKSTART.md) — 5 分钟快速开始。
- [INSTALL.md](INSTALL.md) — 详细安装与故障排查。
- [docs/FIRSTBOOT.md](docs/FIRSTBOOT.md) — 安装与首次启动完整参考。
- [docs/CLI.md](docs/CLI.md) — CLI 命令参考。
- [docs/SAFETY.md](docs/SAFETY.md) — 安全模型与部署规则。
- [docs/ASSETS.md](docs/ASSETS.md) — 物理 AI 资产中心。
- [ARCHITECTURE.md](ARCHITECTURE.md) — 运行时架构。
- [CONTRIBUTING.md](CONTRIBUTING.md) — 开发规范。

---

## 路线图

| 阶段 | 重点 |
|---|---|
| **当前 / Alpha** | 可信 Action/Receipt 契约、Fixture/Provider/Driver 失败关闭、MuJoCo UR5e Reach 黄金闭环、分级 Doctor。 |
| **下一步** | 所有物理副作用迁移至 Gateway；取消/抢占；ROS 2 Turtlesim 真运动观测闭环。 |
| **硬件验收** | 先做 RealSense 只读采集，再做带 ACK、反馈、停止验证和 Receipt 的受限执行任务。 |
| **后续** | 基于 Receipt 的 Memory/How/Auto/Darwin 晋升，配套独立评测与回滚。 |

---

## 贡献

欢迎贡献。开发规范、PR 流程和代码风格见 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## 联系方式

- 邮箱：[ai@rosclaw.io](mailto:ai@rosclaw.io)
- Issues：[GitHub Issues](https://github.com/ros-claw/rosclaw/issues)
- Discussions：[GitHub Discussions](https://github.com/ros-claw/rosclaw/discussions)

---

## 许可证

[MIT](LICENSE)
