# ADR-0004：Team Fabric 控制/数据平面分离与 legacy swarm 冻结

- 状态：Accepted
- 日期：2026-08-01
- 依据：实施总纲 §2.2、§10

## 背景

`rosclaw.swarm` 是内存 registry + 简单距离投标 + “Raft-like”骨架：无网络传输、无 epoch、无日志复制、无分区语义、无故障证明，且默认关闭。以此为基础扩展多机器人协作会继承伪一致性假设。

## 决策

1. `rosclaw.swarm` 标记 `experimental_legacy`：冻结新增功能；文档与 CLI 标注；新建 `rosclaw.team`；完成兼容迁移后删除或保留只读 adapter。**不得**扩展 `swarm/consensus.py` 的 Raft-like 代码；P2 需要强一致时采用成熟协调服务，不在 Python 自研共识。
2. 新建 `src/rosclaw/team/`（membership / roles / task_graph / world_model / allocator / transport / recovery），四平面分离：

   | 平面 | 内容 | 协议 | 频率 |
   |---|---|---|---|
   | Agent/Task Plane | AgentCard、Task、协商 | A2A / ROSClaw contract | 秒级 |
   | Team Control Plane | membership、epoch、role lease、TaskGraph patch | 可靠服务 / Zenoh query | 100 ms–秒级 |
   | World Data Plane | pose、对象状态、map delta | ROS 2/DDS 或 Zenoh | 10–100 Hz |
   | Local Motion Plane | 伺服/安全回路 | 本地控制器 | 50 Hz–kHz |

3. 每台机器人是自治安全单元：团队分配是建议/契约，本地 Native Agent + `rosclawd` 保留拒绝权；LLM 对话永不作为实时控制环。
4. 成员、角色、任务所有权全部带 `team_epoch` + lease；共享世界状态带时间/frame/来源/置信度/revision/freshness 与 tombstone 语义。
5. P0/P1 使用单一逻辑 Coordinator（term/epoch、award/role lease 持久化可幂等重放）；Coordinator 失联时停止产生新团队任务，本地按 degraded policy 继续或安全停止——机器人不因 Coordinator 丢失失去本地安全。
6. 任务分配用可解释 Contract Net/auction（确定性特征排序），不用“模型投票”。
7. 网络分区是正常状态，按总纲 §10.8 降级矩阵执行（世界过期降速/停队形；epoch 不一致拒绝旧 award；A2A 与数据面互相不可替代）。

## 后果

- Team Fabric 从 `experimental` 起步，经 T-SIM-1/2/3 阶梯验证后晋升。
- `rosclaw.swarm` 现有公开 import 保留但 docstring/CLI 增加 legacy 警告。
