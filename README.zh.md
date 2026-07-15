<div align="center">

# ROSClaw

### 面向物理 AI 与具身 Agent 的自进化运行时基础设施

**让 AI Agent 进入机器人身体，让每一次物理行动都可验证、可记忆、可修复、可进化。**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python)](https://www.python.org/)
[![ROS 2](https://img.shields.io/badge/ROS_2-Humble_|_Jazzy-FF3E00?logo=ros)](https://docs.ros.org/)
[![Simulation](https://img.shields.io/badge/Simulation-MuJoCo_|_Isaac--Sim-black?logo=mujoco)](https://mujoco.org/)
[![MCP](https://img.shields.io/badge/Protocol-MCP-8A2BE2)](https://modelcontextprotocol.io/)
[![Status](https://img.shields.io/badge/Release-v1.0-purple)](https://github.com/ros-claw/rosclaw/releases)

[English](README.md) • **中文** • [架构](ARCHITECTURE.md) • [快速开始](QUICKSTART.md) • [文档](docs/)

</div>

```bash
curl -sSL https://rosclaw.io/get | bash
rosclaw firstboot
```

---

## ROSClaw 是什么？

ROSClaw 不是另一个聊天机器人框架，不是简单的“大模型调用 ROS”工具，也不是一堆零散的机器人脚本集合。

ROSClaw 是一套面向物理 AI 与具身 Agent 的运行时基础设施层。它把 AI Agent、机器人本体、仿真沙盒、能力 Provider、物理记忆、实践捕获、运行时干预和技能进化连接到一起，形成统一的物理 AI 运行时。

它为下一代具身智能体而设计：这些智能体不仅要会推理，还要能安全行动、记住经验、失败恢复、持续进化。

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
意图 → 本体上下文 → 能力路由 → 沙盒验证 → 真实执行
     → 轨迹记录 → 物理记忆 → 运行时干预 → 技能进化 → 更安全的技能
```

> **每一次物理行动，都应该被约束、被验证、被记录、被记住，并最终变成更好的技能。**

Auto 可以提出改变，但不能独自批准改变。沙盒验证、Darwin 评测、晋升门和人类确认共同决定改变是否能进入真实世界。

---

## 首次具身 / 快速开始

安装 CLI 并运行交互式首次启动向导：

```bash
curl -sSL https://rosclaw.io/get | bash
rosclaw firstboot
rosclaw doctor
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

## 安装 ROSClaw Agent 集成

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

```bash
# 本地文件
rosclaw practice ingest-seekdb <practice_id> \
  --data-root /data/rosclaw/practice \
  --seekdb-path ~/.rosclaw/memory/seekdb.sqlite

# 真实 SeekDB server
rosclaw practice ingest-seekdb <practice_id> \
  --data-root /data/rosclaw/practice \
  --seekdb-url mysql://root@127.0.0.1:2881/rosclaw

rosclaw practice query failures \
  --robot-id rh56 \
  --data-root /data/rosclaw/practice \
  --seekdb-url mysql://root@127.0.0.1:2881/rosclaw \
  --json
```

SeekDB 的 `2881` 端口使用 MySQL-compatible SQL 协议，不是 HTTP API。
重复执行 Practice ingest 会按 episode 和 evidence ID 幂等更新。

---

## 安全模型

ROSClaw 的核心安全规则：

> **任何模型输出都不应该直接控制机器人。**

每一个物理动作都要经过验证流水线：

1. Provider 生成结构化的动作提案。
2. Sandbox / Firewall 根据有效本体模型和安全策略检查。
3. 决策结果为 `ALLOW`、`MODIFY`、`BLOCK` 或 `REQUIRE_HUMAN_CONFIRMATION`。
4. Practice 记录执行过程。
5. Memory 和 Know 保留证据供后续审计。
6. How 和 Auto 可以提出改进，但只有晋升门能改变当前生效的技能。

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
| **当前 / v1.0** | Runtime、EventBus、Sandbox、Practice、Memory、How、MCP 服务、First Boot、Hub 验证与搜索。 |
| **进行中** | Provider 路由、真实本体上的技能执行、Auto 进化工作流、Darwin 评测。 |
| **研究方向** | 多 Agent 集群协作、生产环境持续自进化、跨机器人技能迁移。 |

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
