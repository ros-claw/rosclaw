# ADR-0009：Data Plane —— Structured Store / Retrieval Store 分层与 SeekDB 定位

- 状态：Accepted
- 日期：2026-08-12
- 依据：《当前 rosclaw seekdb 情况》代码审查、《ROSClaw SeekDB-Centric Data
  Flywheel 重构实施总纲》§17/§20-22/§47-48、ADR-0010

## 背景

ROSClaw 的数据闭环方向（Practice 事实 → Memory 长期回忆 → Know 工程知识 →
How 运行时干预 → Evolution 自进化 → Darwin 评估压力 → Skill Registry）在
`ARCHITECTURE.md` 中已正确定义，且多条闭环已在代码中存在。但实现处于过渡态：

- `SeekDBClient` 一名多义（InMemory / SQLite / MySQL-compatible / native），
  Runtime 默认 `seekdb_backend="sqlite"` —— 概念上是 SeekDB Core，工程上不是；
- Runtime 用 `seekdb = ...` 局部变量依次塞给 Memory/Skill/Knowledge/How/Auto，
  无法判断后端语义；
- Knowledge v2 `memory_store_shared=false`，Auto 主状态在 `LocalStore`、
  SeekDB 仅 best-effort 旁路镜像；
- `rosclaw.memory.insight` 只有 subscriber（Auto），没有 publisher。

## 决策

### 1. 两层存储正式分层

**Structured Store**（结构化事实与状态的 source of truth）：

```text
事务状态 / 结构化事实 / Memory / Knowledge metadata / Evolution / Darwin / Lineage
生产：SeekDB SQL（MySQL-compatible）    边缘/离线：SQLite
```

**Retrieval Store**（检索投影，可删除、可重建）：

```text
Vector / BM25 / Hybrid / Metadata filter / RRF / Versioned ACTIVE collection
实现：SeekDB Native（pyseekdb，embedded 或 server）
```

部署形态：

| 场景 | Structured | Retrieval |
|---|---|---|
| Edge / Offline | SQLite Structured Store | SeekDB Native embedded |
| Server / Cluster | SeekDB SQL | SeekDB Native server |

`SQLite → Outbox → SeekDB native projection` 的 source-of-truth 与检索投影
分离设计保留；Retrieval projection 永远可删除、可重建，**绝不是唯一
source of truth**。

### 2. DataPlaneContext 单次初始化

Runtime 初始化一次 `DataPlaneContext`（structured_store / retrieval_store /
outbox / memory_projection / memory_retrieval），各模块依赖注入自己需要的部分。
禁止 `seekdb = ...` 式模糊变量。

### 3. Heavy Artifact 红线

视频、MCAP、深度图、高频 telemetry、模型权重**绝不进入 SeekDB**。SeekDB 只保存
`artifact_id / type / uri / sha256 / size / time_range / schema / episode_id /
practice_id` 等索引与意义层字段。MemoryWriteGate 对高频 frame/IMU/telemetry
与完整 CoT 的拒绝规则继续保持。

### 4. rosclawd 安全权威边界不变

permit ledger、action ledger、E-stop latch、REAL execution authorization 继续由
daemon 本地持久化并拥有权威。SeekDB 只做异步投影/索引/分析/学习：

- 禁止 REAL dispatch 前等待远端 SeekDB；
- 禁止 E-stop 等数据库响应；
- 禁止 permit 校验依赖任何数据库。

SeekDB outage 时机器人基本执行不受影响、Practice 落盘、Outbox 累积、恢复后
catch-up（P0 验收，E2E 六）。

### 5. 迁移原则

兼容层优先、禁止 Big Bang rename、行为切换与 rename 分 PR（ADR-0010 §纪律）。
完整 PR 序列见《实施总纲》§49（PR-DF-00 … PR-DF-16）。

## 后果

- 命名与后端语义一一对应，代码审查与配置排障不再歧义；
- SQLite 边缘部署与 SeekDB 生产部署共享同一抽象，迁移路径清晰；
- 检索索引可随时重建，数据可恢复性明确；
- 安全控制面与数据平面的耦合被明文禁止并有故障注入验收。
