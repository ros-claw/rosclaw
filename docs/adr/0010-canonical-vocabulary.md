# ADR-0010：Canonical Vocabulary（数据平面命名冻结）

- 状态：Accepted
- 日期：2026-08-12
- 依据：《ROSClaw SeekDB-Centric Data Flywheel 重构实施总纲》§2/§3/§51-52、ADR-0009

## 背景

当前代码中 `SeekDBClient` 可能是 InMemory / SQLite / MySQL-compatible /
pyseekdb native 中的任何一种；`knowledge_store` 实际被 Memory、Skill、How、
Auto、Legacy Know 共用；`rosclaw.know` 与 `rosclaw.knowledge` 双包并存；
`auto` / `evolution` / `continual` / `darwin` 命名分裂。命名语义污染已经
造成代码审查与运维的实际误解。

## 决策

以下词汇为唯一 canonical，所有源码、文档、CLI、EventBus、Dashboard 以其为准：

| 概念 | Canonical 名称 | 职责 |
|---|---|---|
| 原始事实 | **Practice** | 发生了什么 |
| 长期经验 | **Memory** | 应该记住什么 |
| 工程知识 | **Knowledge** | 可以泛化出什么 |
| 运行时干预 | **How** | 当前应该怎么修 |
| 自进化编排 | **Evolution** | 应该尝试改变什么 |
| 评估压力 | **Darwin** | 改变究竟好不好 |
| 模型训练 | **Learning** | 真正更新模型/权重 |
| 重型文件 | **Artifact** | 视频、MCAP、图片、模型等 |
| 物理执行证据 | **ExecutionReceipt** | 真实执行发生到了哪一步 |
| 数据血缘 | **Lineage** | 一个结果从哪里演化而来 |
| 结构化存储 | **Structured Store** | 表、状态、事实、关系 |
| 检索存储 | **Retrieval Store** | Vector/BM25/Hybrid |
| 数据基础设施 | **Storage / Data Plane** | Store、Outbox、Projection |

### 类/包名映射（兼容层优先，禁止 Big Bang rename）

| 当前 | Canonical | 迁移方式 |
|---|---|---|
| `SeekDBClient` | `StructuredStore` | 别名兼容，逐步替换内部 import |
| `InMemoryKnowledgeStore` | `InMemoryStructuredStore` | 同上 |
| `SQLiteKnowledgeStore` | `SQLiteStructuredStore` | 同上 |
| `SeekDBMySQLClient` | `SeekDBSQLStore` | 同上 |
| `SEEKDB_SCHEMAS` | `ROSCLAW_STRUCTURED_SCHEMAS` | 同上 |
| `StorageFactory.create_knowledge_store()` | `StoreFactory.create_structured_store()` | 同上 |
| `SeekDBNativeStore` | `SeekDBRetrievalStore` | 同上 |
| `SeekDBEmbeddedStore` / `SeekDBServerStore` | `SeekDBEmbeddedRetrievalStore` / `SeekDBServerRetrievalStore` | 同上 |
| `SeekDBProjection` | `MemoryRetrievalProjection` | 同上 |
| `PracticeDistiller` | `EpisodeFactExtractor` | 行为迁移期保留旧名入口 |
| `SeekDBIngestor` | `PracticeFactIngestor` | 同上 |
| `SeekDBBridge` | `PracticeEventSink`（Protocol）+ `SeekDBHttpPracticeSink` | 别名兼容 |
| `rosclaw.know` | `rosclaw.knowledge`（canonical） | `know` 保留 compatibility shim ≥1 minor release |
| `rosclaw.auto` | `rosclaw.evolution.orchestrator` | 别名兼容；CLI `auto` → `evolve` 别名 |
| `rosclaw.continual` | `rosclaw.learning`（P2，晚于数据平面） | 本轮不动 |

### Event topics

新增 canonical topic 通过 `EventTopics` + `_TOPIC_COMPAT` 归一化机制扩展：

```text
rosclaw.evolution.proposal.created      (compat: rosclaw.auto.proposal.created)
rosclaw.evolution.experiment.completed  (compat: rosclaw.auto.experiment.completed)
rosclaw.evolution.champion.promoted     (compat: rosclaw.auto.champion.promoted)
rosclaw.memory.insight.created
```

核心事件一律使用 `EventTopics.*`，禁止散写裸字符串。

### 工程纪律

1. 兼容层优先：`SeekDBClient = StructuredStore` 式别名保留至内部引用清零。
2. 不允许同一 PR 同时改行为 + 大规模 rename。
3. 数据库 schema 版本 `memory.v2` 是数据协议版本，保留；源码包的 `memory.v2`
   目录在行为主线收敛后才物理迁移（PR-DF-16）。
4. CLI 用户词汇 `rosclaw know` / `rosclaw how` 保留（用户认知合理），Python
   package 只用 `rosclaw.knowledge`。
