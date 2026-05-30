# KNOW 模块反思报告

## 一、知识图谱一致性

### 现状
- 存储层：SeekDB `knowledge_graph` 表（subject-predicate-object-confidence-source-timestamp）
- 缓存层：KNOW 在 `_do_initialize()` 时加载 capabilities/symptoms/patterns 到内存字典
- 写入路径：`load_eurdf_profile()` 直接 INSERT，无事务；`record_knowledge_usage()` 直接 INSERT
- 无版本控制：同一 robot_id 的 e-URDF 重复加载会产生重复记录，无 upsert/去重逻辑

### 问题
1. **写冲突**：KNOW 和 Memory 各自持有 SeekDB client 引用，并发写入同一表无锁保护
2. **缓存失效**：内存缓存 `_capabilities` / `_patterns` 初始化后不再刷新，SeekDB 更新后 KNOW 读的是旧数据
3. **重复数据**：`load_eurdf_profile()` 每次调用都 INSERT 新记录，无 idempotency

### 建议
- 添加 `ON CONFLICT REPLACE` 或先 query 后 upsert 的逻辑
- 增加 `_refresh_from_seekdb()` 方法，在 EventBus 收到 `knowledge.updated` 时刷新缓存
- KNOW 和 Memory 共享同一个 SeekDB client 实例（Runtime 已部分实现，但需保证线程安全）

## 二、与 Memory 的查询接口

### 现状
- **Memory → KNOW**：MemoryInterface 有 `query_knowledge_graph()` 和 `retrieve_robot_capability()`，直接操作 SeekDB 的 knowledge_graph 表
- **KNOW → Memory**：KNOW 无直接调用 Memory 的接口；通过 EventBus 发布 `knowledge.pre_check` / `knowledge.ingest_complete`，Memory 作为 subscriber 被动接收
- **API 冗余**：Memory 和 KNOW 都能查询 capabilities，但 API 不同（Memory 返回 triple 列表，KNOW 返回结构化 dict）

### 问题
1. **单向耦合**：KNOW 不能查询 Memory 中的经验数据（如"这个机器人在类似任务中失败过几次"），导致推理缺少上下文
2. **API 分裂**：同一功能（查机器人能力）有两个入口，调用方无所适从
3. **EventBus 延迟**：KNOW 通过事件异步通知 Memory，不适合需要即时查询结果的场景

### 建议
- 在 Runtime 中建立统一入口：`runtime.query_knowledge(query_type, ...)`，由 Runtime 决定路由到 KNOW 或 Memory
- KNOW 增加 `query_memory_context(task, robot_id)` 方法，直接调用 MemoryInterface 的 `find_similar_experiences()`
- 合并 capability 查询 API：KNOW 作为"结构化推理层"，Memory 作为"原始数据层"，统一返回 schema

## 三、实时更新机制

### 现状
- **初始化加载**：`_do_initialize()` → `_load_from_seekdb()` 一次性加载
- **事件触发**：KNOW 订阅了 `provider.inference.requested`、`sandbox.episode.started`、`runtime.execution.completed`
- **更新路径**：事件处理函数中只发布新事件，不更新知识库本身
- **e-URDF 加载**：`load_eurdf_profile()` 是显式调用，无自动触发

### 问题
1. **静态知识**：运行时新技能注册、新机器人加入、失败模式更新都不会自动刷新 KNOW 缓存
2. **无知识演化**：Practice 积累了大量 failure/success 数据，但 KNOW 不从中学习更新 heuristic rules
3. **更新盲区**：多个 Runtime 实例共享 SeekDB 时，A 实例的更新对 B 实例不可见

### 建议
- 增加 `knowledge.update` EventBus topic，当 e-URDF 加载、技能注册、规则更新时发布
- KNOW 订阅 `knowledge.update`，调用 `_refresh_from_seekdb()` 刷新缓存
- 增加离线"知识蒸馏"：定期（或 episode 结束时）从 Memory 的 experience_graph 中提取新的 success/failure patterns，更新到 KNOW 的 `_patterns` 和 SeekDB 的 `heuristic_rules`
- 参考 HOW 的 `HeuristicEngine.seed_defaults()`，为 KNOW 增加 `learn_from_experiences()` 方法

## 总结

KNOW 从"孤岛"到"接入主流程"已完成（EventBus + Runtime 钩子 + Practice 记录），但要达到 v1.0 GA 的"知识可演化、可推理、可自愈"标准，还需要：

| 优先级 | 改进项 | 预计工时 |
|--------|--------|----------|
| P1 | 缓存刷新 + 写去重 | 2h |
| P1 | KNOW ↔ Memory 双向查询 | 3h |
| P2 | 知识蒸馏（从经验学习） | 4h |
| P2 | 统一查询 API | 2h |
