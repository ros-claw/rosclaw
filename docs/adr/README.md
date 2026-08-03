# ROSClaw 架构决策记录（ADR）

本目录记录 ROSClaw 的架构级决策。流程、命名规范与成熟度等级见 [ADR-0000](0000-adr-process-naming-maturity.md)。

| ADR | 标题 | 状态 |
|---|---|---|
| [0000](0000-adr-process-naming-maturity.md) | ADR 流程、命名规范与功能成熟度等级 | Accepted |
| [0001](0001-native-agent-process-boundary.md) | Native Agent 独立进程边界（rosclaw-agentd） | Accepted |
| [0002](0002-mission-vs-physical-session.md) | MissionSession 与物理 AgentSession 分离 | Accepted |
| [0003](0003-cognitive-worker-fabric.md) | 认知 Worker Fabric 与 daemon 硬件 adapter 分离 | Accepted |
| [0004](0004-team-fabric-planes-legacy-swarm.md) | Team Fabric 平面分离与 legacy swarm 冻结 | Accepted |
| [0005](0005-embodied-context-compiler.md) | Embodied Context Compiler | Accepted |
| [0006](0006-authorization-invariant-operator-broker.md) | 授权不变量与 Operator Broker / MissionGrant | Accepted |
| [0007](0007-dual-layer-operator-consent.md) | 双层 Operator Consent 集成路径（agentd grants ↔ daemon proposals） | Accepted |
| [0008](0008-pi-dependency-boundary.md) | Pi 依赖边界（pi-tui/pi-ai 可 import，Pi Agent 生态禁止） | Accepted |

依据文档：《ROSClaw Native Agent、Worker Fabric 与多机器人 Team Fabric 实施总纲 v1.0》（2026-08-01）。
