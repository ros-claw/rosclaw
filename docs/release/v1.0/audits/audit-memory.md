# Memory + SeekDB Audit Report

> **Auditor**: Memory domain owner session
> **Date**: 2026-05-28
> **Scope**: `src/rosclaw/memory/` (v1.0 integrated) vs `part/rosclaw-memory/` (independent, 95K+ lines)

---

## Executive Summary

v1.0 的 Memory 模块是一个 **621 行的薄适配层**（3 个 Python 文件），通过 SeekDB 提供经验存储和关键词搜索。它**正确地实现了 RFC-0005 定义的最小可行闭环**：`PraxisEvent → EventBus → SeekDB → Memory 查询`。

与独立模块 `part/rosclaw-memory`（56,505 行核心代码 + 37 个具身记忆子模块文件）相比，v1.0 集成度是 **~1%**，但这是合理的 —— RFC-0001 明确将 rosclaw-memory 标记为 P1 集成优先级，v1.0 只需 "Basic event storage"。

**结论**：v1.0 Memory 满足 P0 验收门控，但有 5 个 P1 问题需要在 RC1 前修复，3 个 P2 建议留给 v1.1。

---

## 1. Current Memory API Inventory

### 1.1 MemoryInterface（322 行）

| API | 签名 | 后端 | 状态 |
|-----|------|------|------|
| `store_experience()` | 11 params → str | SeekDB `experience_graph` | ✅ 工作 |
| `find_similar_experiences()` | instruction, limit, outcome_filter → list[dict] | 关键词匹配（内存扫描） | ⚠️ ISSUE-002 |
| `get_experience()` | experience_id → dict | SeekDB 查询 | ✅ 工作 |
| `get_statistics()` | → dict | SeekDB count | ✅ 工作 |
| `add_world_object()` | obj → str | EmbodiedMemory 代理 | ⚠️ ISSUE-001 |
| `get_world_object()` | obj_id → Any | EmbodiedMemory 代理 | ⚠️ ISSUE-001 |
| `update_world_object_pose()` | obj_id, pose, state → bool | EmbodiedMemory 代理 | ⚠️ ISSUE-001 |
| `search_world_objects()` | center, radius, scene_id → list | EmbodiedMemory 代理 | ⚠️ ISSUE-001 |
| `get_scene_graph()` | scene_id → tuple | EmbodiedMemory 代理 | ⚠️ ISSUE-001 |
| `compute_relations()` | scene_id, spatial_tolerance → list | EmbodiedMemory 代理 | ⚠️ ISSUE-001 |
| `sync_scene_objects()` | scene_id, detections, timestamp_sec → Report | EmbodiedMemory 代理 | ⚠️ ISSUE-001 |
| `record_trajectory()` | content, waypoints → int | EmbodiedMemory 代理 | ⚠️ ISSUE-001 |
| `search_similar_trajectories()` | query_waypoints, top_k, max_dtw → list | EmbodiedMemory 代理 | ⚠️ ISSUE-001 |
| `cognitive_search()` | query, spatial, temporal → list | EmbodiedMemory 代理 | ⚠️ ISSUE-001 |
| `run_meditation()` | phases → dict | EmbodiedMemory 代理 | ⚠️ ISSUE-001 |

### 1.2 SeekDBClient（273 行）

| 实现 | 类型 | 后端 | Schema 数量 |
|------|------|------|------------|
| `SeekDBClient` | 抽象基类 | — | 定义 4 表 |
| `SeekDBMemoryClient` | 内存 | dict of dicts | 4 表 |
| `SeekDBSQLiteClient` | 持久化 | SQLite 文件 | 4 表 |

**SeekDB Schema 定义**：

| 表名 | 列数 | 用途 | v1.0 使用状态 |
|------|------|------|-------------|
| `experience_graph` | 12 | 经验存储 | ✅ Memory + Practice 写入 |
| `skill_metadata` | 11 | 技能元数据 | ❌ 未使用（ISSUE-004） |
| `knowledge_graph` | 6 | 知识图谱三元组 | ❌ 未使用 |
| `heuristic_rules` | 7 | How 模块启发式规则 | ❌ 未使用 |

### 1.3 Runtime 集成点

`src/rosclaw/core/runtime.py`（557 行）中有 **48 处 Memory 引用**：
- 生命周期管理：`_memory` 属性初始化、start/stop
- 配置驱动：`enable_memory` 配置项、`memory_backend` 选择
- 物理世界 API 代理：`add_world_object()`、`record_trajectory()` 等委托给 `_memory`

---

## 2. Current SeekDB Usage Patterns

### 2.1 RFC-0001 合规性：SeekDB 定位为 Knowledge Plane

| 要求 | 实际 | 评估 |
|------|------|------|
| "SeekDB is Infrastructure Layer, not Memory Layer" | Schema 定义了 4 个跨模块表 | ✅ 设计正确 |
| "Memory: experience graph" | `experience_graph` 表由 Memory 写入 | ✅ 实现 |
| "Practice: event index" | PraxisEvent → `_on_praxis_recorded` → `experience_graph` | ✅ 实现 |
| "Know: knowledge graph" | `knowledge_graph` 表已定义但无写入方 | ⚠️ 空表（P2） |
| "How: heuristic rules" | `heuristic_rules` 表已定义但无写入方 | ⚠️ 空表（P2） |
| "Skill: skill metadata" | `skill_metadata` 表已定义但无写入方 | ⚠️ ISSUE-004 |
| "v1.0 minimum: Support Memory + Practice event storage" | Memory + Practice 路径完整 | ✅ 满足 |

### 2.2 查询模式分析

| 查询类型 | 实现方式 | 性能 | 风险 |
|----------|----------|------|------|
| 按 ID 查询 | `filters={"id": ...}` | O(N) 内存 / O(1) SQLite | 低 |
| 按 robot_id 过滤 | `filters={"robot_id": ...}` | 全表扫描 | ⚠️ ISSUE-003 |
| 按 outcome 过滤 | `filters={"outcome": ...}` | 全表扫描 | 中（数据量小可接受） |
| 时间排序 | `order_by="-timestamp"` | Python sort | ⚠️ ISSUE-003 |
| 相似性搜索 | 关键词集合交集 | O(N*M) | ⚠️ ISSUE-002 |
| 空间查询 | 无（需 EmbodiedMemory） | N/A | P2 |
| 向量搜索 | 无 | N/A | P2 |

---

## 3. Minimum Closed Loop Verification

### 3.1 端到端路径：Practice → SeekDB → Memory

```
Practice 执行任务
    │
    ▼
EventBus.publish("praxis.recorded", {event_id, event_type, instruction, ...})
    │
    ▼
MemoryInterface._on_praxis_recorded(event)  ← 订阅
    │
    ▼
MemoryInterface.store_experience()
    │
    ▼
SeekDBClient.insert("experience_graph", record)
    │
    ▼
Agent 调用 MemoryInterface.find_similar_experiences(instruction)
    │
    ▼
SeekDBClient.query("experience_graph", filters={"robot_id": ...})
    │
    ▼
关键词匹配 → 返回排序后的经验列表
```

**验证结果**：✅ **闭环路径存在且可测试**

证据：`tests/test_seekdb.py::test_praxis_auto_ingestion` 验证了完整路径。

### 3.2 缺失环节

| 环节 | 状态 | 影响 |
|------|------|------|
| Agent 主动查询经验 | 无代码路径 | Agent 无法在任务规划时利用历史经验 |
| 经验反馈到 Practice | 无实现 | 无法基于历史成功/失败调整策略 |
| 多 Agent 经验共享 | `robot_id` 隔离 | 不同机器人经验不互通 |

---

## 4. Integration Gap Analysis（part/ vs v1.0）

### 4.1 代码规模对比

| 模块 | 文件数 | 代码行数 | v1.0 集成度 |
|------|--------|----------|------------|
| `part/rosclaw-memory` 总计 | 212 | 56,505 | 0%（未引入） |
| `part/rosclaw-memory/embodied/` | 37 | 12,682 | 1%（代理 API 定义） |
| v1.0 `src/rosclaw/memory/` | 3 | 621 | 100%（自包含） |

### 4.2 功能对比

| 功能 | 独立模块 | v1.0 集成 | v1.0 必要性 |
|------|----------|----------|------------|
| 经验存储/查询 | ✅ 向量嵌入 + BM25 | ⚠️ 仅关键词匹配 | P1（ISSUE-002） |
| 世界对象 CRUD | ✅ 完整（含 occlusion） | ✅ 代理 API | P0（✅ 已有） |
| 轨迹记忆 + DTW | ✅ 完整（numpy 加速） | ✅ 代理 API | P1（✅ 已有） |
| 场景图 | ✅ 完整 | ✅ 代理 API | P1（✅ 已有） |
| 认知路由 | ✅ Tri-Route | ❌ 无 | P2 |
| 冥想管线 | ✅ 3 阶段 | ✅ 代理 API | P2 |
| 对象恒存 | ✅ 含运动预测 | ✅ 代理 API | P1（✅ 已有） |
| 物理约束 | ✅ 完整 | ❌ 无 | P2 |
| gRPC 服务 | ✅ 完整 | ❌ 无 | P2 |
| 记忆遗忘 | ✅ 年龄+显著性 | ❌ 无 | P1（ISSUE-005） |
| 空间/时间索引 | ✅ 体素哈希+时间树 | ❌ 无 | P2 |

### 4.3 v1.0 关键决策：代理模式 vs 直接集成

v1.0 选择了**代理模式**（`MemoryInterface._embodied` 可选注入），这意味着：
- powermem **不是 v1.0 的依赖**（`pyproject.toml` 无 powermem 条目）
- EmbodiedMemory 的所有代理方法都有 `if self._embodied is None: return None/[]/False` 守卫
- 用户可以选择性启用具身记忆，不影响核心运行时

**评估**：这是正确的架构决策。v1.0 聚焦框架层稳定性，具身记忆作为可选增强。

---

## 5. Issue List

### P0 Issues（Release Blockers）

**无 P0 问题。** v1.0 Memory 满足 RFC-0005 Gate #8（"Memory can query a stored event"）。

### P1 Issues（Should Fix Before Release）

---

#### ISSUE-001: EmbodiedMemory 代理方法无类型安全

**Severity**: P1
**Module**: memory
**Detected by**: memory-auditor
**Status**: partial-fix (powermem side complete; v1.0 proxy annotations pending)
**Fix**: `powermem.embodied.protocols` now exposes `Vec3Like`, `PoseLike`, `WorldObjectLike`,
  `TemporalIntervalLike`, `PermanenceReportLike`, `SceneGraphLike`, `SpatialRelationLike`,
  `MemoryAtomLike`, `TelemetryLike`, `EmbodiedMemoryLike` — all `@runtime_checkable` Protocols.
  v1.0 `MemoryInterface` should replace `Any` annotations using the conditional-import pattern
  documented in `tests/unit/test_protocols.py::TestV10IntegrationPattern`.

**Problem**: `MemoryInterface` 的 11 个 EmbodiedMemory 代理方法全部使用 `Any` 类型注解，无输入验证。

**Evidence**:
- file: `src/rosclaw/memory/interface.py`
- lines: 175-275（`add_world_object` 到 `run_meditation`）
- 示例: `def add_world_object(self, obj: Any) -> Optional[str]`
- 示例: `def record_trajectory(self, content: str, waypoints: list[tuple[Any, float]]) -> Optional[int]`

**Why it matters**:
1. Agent 或其他调用方无法获得类型提示，容易传错参数
2. 如果 `powermem.types.WorldObject` 与 v1.0 期望的 `obj` 格式不匹配，运行时会静默返回 `None`
3. 代理方法的返回值类型（如 `PermanenceReport`）对外部调用方完全不可见

**Expected behavior**: 代理方法应导入 powermem 类型或使用 Protocol 定义接口契约。

**Suggested fix**:
```python
# 方案 A：类型导入（推荐，powermem 为可选依赖）
try:
    from powermem.embodied.types import WorldObject, Vec3, Pose, TemporalInterval
    HAS_POWERMEM = True
except ImportError:
    HAS_POWERMEM = False
    WorldObject = Any  # type: ignore
    Vec3 = Any  # type: ignore
    Pose = Any  # type: ignore
```

**Verification**:
```bash
python -c "from rosclaw.memory import MemoryInterface; import inspect; sig = inspect.signature(MemoryInterface.add_world_object); print(sig)"
```

---

#### ISSUE-002: `find_similar_experiences` 仅使用关键词匹配

**Severity**: P1
**Module**: memory
**Detected by**: memory-auditor
**Status**: open

**Problem**: 相似性搜索使用简单的关键词集合交集，无语义理解能力。

**Evidence**:
- file: `src/rosclaw/memory/interface.py`
- lines: 144-166
- 算法: `overlap = len(keywords & exp_words)` — 纯词袋匹配
- 无嵌入向量、无 BM25、无语义搜索

**Why it matters**:
1. "pick up the cup" 和 "grasp the mug" 零匹配，但语义完全等价
2. Agent 无法找到相关历史经验来辅助决策
3. RFC-0001 第 5 节 Known Gaps 已标注 "Memory only in-memory backend" 为 P1

**Expected behavior**: 至少支持 BM25 或 TF-IDF 排序，理想情况下支持嵌入向量搜索。

**Suggested fix**:
```python
# 短期：引入 rank-bm25（powermem 已依赖此包）
from rank_bm25 import BM25Okapi

# 中期：使用 powermem.integrations 的 embedding provider
# 长期：接入 powermem.storage 的向量搜索
```

**Verification**:
```bash
cd ~/rosclaw/rosclaw/rosclaw-v1.0
python -c "
from rosclaw.memory import MemoryInterface
m = MemoryInterface('bot'); m.initialize()
m.store_experience('1', 'test', 'pick up red cup', outcome='success')
m.store_experience('2', 'test', 'grasp the mug', outcome='success')
r = m.find_similar_experiences('grasp mug')
print([x['instruction'] for x in r])  # 应包含 'grasp the mug'
"
```

---

#### ISSUE-003: SeekDB 查询无索引加速，`query()` 全表扫描

**Severity**: P1
**Module**: memory
**Detected by**: memory-auditor
**Status**: open

**Problem**: `SeekDBMemoryClient.query()` 对内存 dict 做线性扫描。`SeekDBSQLiteClient` 虽定义了索引但 `query()` 方法未利用。

**Evidence**:
- file: `src/rosclaw/memory/seekdb_client.py`
- lines: 117-131（MemoryClient 全表扫描）
- lines: 167-185（SQLiteClient 拼接 SQL，但索引仅用于单列等值过滤）
- 无复合索引、无范围查询优化

**Why it matters**:
1. 随经验数量增长（>1000 条），查询延迟线性增加
2. RFC-0005 P1 Gate #8 要求 "Memory buffer growth has size limits"，但无性能退化防护

**Expected behavior**: SQLite 后端应利用已建索引；内存后端应维护二级索引或设置最大查询量。

**Suggested fix**:
```python
# SeekDBMemoryClient: 为高频过滤字段维护倒排索引
# SeekDBSQLiteClient: 验证 EXPLAIN QUERY PLAN 确认索引命中
```

**Verification**:
```bash
cd ~/rosclaw/rosclaw/rosclaw-v1.0
python -c "
from rosclaw.memory import SeekDBSQLiteClient
c = SeekDBSQLiteClient('/tmp/seekdb_bench.sqlite'); c.connect()
for i in range(10000):
    c.insert('experience_graph', {'id': f'exp_{i}', 'event_type': 'test', 'robot_id': 'r1', 'timestamp': float(i)})
import time; t0 = time.time()
r = c.query('experience_graph', filters={'robot_id': 'r1'}, order_by='-timestamp', limit=10)
print(f'10K rows query: {(time.time()-t0)*1000:.1f}ms, results: {len(r)}')
c.disconnect()
"
```

---

#### ISSUE-004: `skill_metadata` 表未被 Skill Manager 写入

**Severity**: P1
**Module**: memory + skill_manager
**Detected by**: memory-auditor
**Status**: open

**Problem**: SeekDB 定义了 `skill_metadata` 表，但 `src/rosclaw/skill_manager/` 中无任何代码写入此表。

**Evidence**:
- file: `src/rosclaw/memory/seekdb_client.py`
- lines: 36-54（`skill_metadata` schema 定义）
- file: `src/rosclaw/skill_manager/executor.py`
- line: 112（executor 仅读取 `self.memory` 引用，无写入 `skill_metadata` 的代码）

**Why it matters**:
1. RFC-0001 要求 "Skill: skill metadata" 作为 SeekDB Knowledge Plane 的一部分
2. Skill 执行统计（成功/失败计数、平均耗时）无法持久化
3. Dashboard 无法展示技能执行历史

**Expected behavior**: SkillManager 在执行完成后应调用 `SeekDBClient.insert("skill_metadata", ...)` 更新统计。

**Suggested fix**: 在 `skill_manager/executor.py` 的 `execute()` 方法中添加 post-execution hook，写入/更新 `skill_metadata` 表。

**Verification**:
```bash
grep -r "skill_metadata" ~/rosclaw/rosclaw/rosclaw-v1.0/src/rosclaw/skill_manager/
# 应有至少一处写入
```

---

#### ISSUE-005: 无记忆容量管理和遗忘机制

**Severity**: P1
**Module**: memory
**Detected by**: memory-auditor
**Status**: open

**Problem**: `MemoryInterface` 只有 `store_experience()` 和 `get_experience()`，无 `delete_experience()` 或容量限制。

**Evidence**:
- file: `src/rosclaw/memory/interface.py`
- 全文无 `delete` 方法
- 全文无 max_size、max_age、capacity 等概念
- `SeekDBClient` 接口无 `delete()` 方法

**Why it matters**:
1. 长时间运行的机器人会无限累积经验记录
2. RFC-0005 P1 Gate #8 要求 "Memory buffer growth has size limits"
3. 独立模块 `part/rosclaw-memory` 已实现 `forget_old_memories()` 和 `vacuum_indexes()`

**Expected behavior**: 至少提供 `delete_experience(id)` 和 `get_capacity_info()` 方法。

**Suggested fix**:
```python
# 1. SeekDBClient 添加 delete() 方法
# 2. MemoryInterface 添加 forget_old_experiences(max_age_days=30)
# 3. Runtime 定期调用（或在 store_experience 时检查容量）
```

**Verification**:
```bash
python -c "
from rosclaw.memory import MemoryInterface
m = MemoryInterface('bot'); m.initialize()
m.store_experience('del1', 'test', 'to delete')
# m.delete_experience('del1')  # 应可用
assert m.get_experience('del1') is not None
print('delete_experience not implemented')
"
```

---

### P2 Issues（Defer to v1.1）

---

#### ISSUE-006: `knowledge_graph` 和 `heuristic_rules` 表为空

**Severity**: P2
**Module**: memory
**Detected by**: memory-auditor
**Status**: deferred

**Problem**: SeekDB Schema 定义了 `knowledge_graph`（Know 模块用）和 `heuristic_rules`（How 模块用）表，但 v1.0 无对应模块集成。

**Evidence**:
- file: `src/rosclaw/memory/seekdb_client.py`
- lines: 55-77
- RFC-0001 第 3 节将 Know 和 How 标记为 P2（v1.1）

**Why it matters**: 不影响 v1.0 发布，但 Schema 存在但无人使用的"幽灵表"可能误导开发者。

**Suggested fix**: 在 Schema 定义中添加注释标记 `# v1.1: used by Know/How module`，或在 `_create_tables()` 中延迟创建。

**Verification**: N/A（v1.1 再验证）

---

#### ISSUE-007: 无空间/时间查询能力（需 EmbodiedMemory）

**Severity**: P2
**Module**: memory
**Detected by**: memory-auditor
**Status**: deferred

**Problem**: `MemoryInterface` 的空间查询（`search_world_objects`）和时间查询（`search_similar_trajectories`）完全依赖可选的 EmbodiedMemory。未启用时返回空列表。

**Evidence**:
- file: `src/rosclaw/memory/interface.py`
- lines: 200-260（所有空间/时间方法都有 `if self._embodied is None: return []` 守卫）

**Why it matters**: 不影响 v1.0（RFC-0001 将空间/时间记忆标记为 P1 集成优先级），但 v1.1 需要原生支持。

**Suggested fix**: v1.1 将 `part/rosclaw-memory` 的 spatial_index 和 temporal_index 抽取为独立模块，不依赖完整 powermem。

**Verification**: N/A（v1.1）

---

#### ISSUE-008: SeekDB 无 Schema 版本控制

**Severity**: P2
**Module**: memory
**Detected by**: memory-auditor
**Status**: deferred

**Problem**: `SEEKDB_SCHEMAS` 字典无版本号，`_create_tables()` 使用 `CREATE TABLE IF NOT EXISTS`，无法处理 Schema 迁移。

**Evidence**:
- file: `src/rosclaw/memory/seekdb_client.py`
- lines: 15-80（无 `schema_version` 字段）
- RFC-0005 P1 Gate #4 要求 "Schema versioning on PraxisEvent and MemorySchema"

**Why it matters**: 如果 v1.1 修改 Schema（如添加向量列），无法自动迁移已有数据。

**Suggested fix**: 添加 `SEEKDB_SCHEMA_VERSION = "1.0.0"` 常量，在 `_create_tables()` 中写入 `schema_meta` 表。

**Verification**: N/A（v1.1）

---

## 6. Recommended v1.0 Minimum Memory Contract

基于审计发现，v1.0 的 Memory Contract 应正式定义为：

### 6.1 MUST HAVE（P0，已有）

| 契约 | 描述 |
|------|------|
| `store_experience(event_id, event_type, instruction, ...)` | 存储经验到 SeekDB |
| `get_experience(experience_id)` | 按 ID 检索经验 |
| `find_similar_experiences(instruction, limit)` | 返回相似经验（关键词匹配可接受） |
| `get_statistics()` | 返回经验统计 |
| EventBus 订阅 `praxis.recorded` | 自动摄入 PraxisEvent |
| EventBus 发布 `memory.experience.stored` | 通知其他模块 |
| SeekDB SQLite 后端 | 持久化存储 |

### 6.2 SHOULD HAVE（P1，建议 RC1 前修复）

| 契约 | 描述 | 对应 Issue |
|------|------|----------|
| 类型安全的代理方法 | 导入 powermem 类型或使用 Protocol | ISSUE-001 |
| 语义搜索（BM25+） | 超越关键词匹配 | ISSUE-002 |
| 查询性能保障 | 索引命中 + 结果上限 | ISSUE-003 |
| Skill 元数据写入 | SkillManager → skill_metadata | ISSUE-004 |
| 容量管理 | delete + forget + 容量查询 | ISSUE-005 |

### 6.3 NICE TO HAVE（P2，v1.1）

| 契约 | 描述 | 对应 Issue |
|------|------|----------|
| 知识图谱读写 | knowledge_graph 表 | ISSUE-006 |
| 空间/时间原生查询 | 不依赖 EmbodiedMemory | ISSUE-007 |
| Schema 版本控制 | 迁移支持 | ISSUE-008 |

---

## 7. SeekDB Positioning Assessment

### RFC-0001 要求
> "SeekDB is Infrastructure Layer, not Memory Layer. Any module may read/write via its contract."

### 实际评估

| 维度 | 评分 | 说明 |
|------|------|------|
| Schema 广度 | ✅ 4 表覆盖 4 模块 | experience, skill, knowledge, heuristic |
| 模块接入数 | ⚠️ 仅 2/4 | Memory + Practice 写入，Skill/Know/How 未接入 |
| 抽象层质量 | ✅ ABC + 2 实现 | 内存 + SQLite，可扩展 |
| 查询灵活性 | ⚠️ 基础 | 仅等值过滤 + 单列排序，无范围/复合/全文 |
| 性能保障 | ⚠️ 弱 | 无连接池、无查询超时、无批量操作 |
| 文档完整度 | ⚠️ 无独立文档 | 无 SeekDB API 文档或 Schema 说明 |

**结论**：SeekDB 在 v1.0 中**设计上是 Knowledge Plane，实现上更像 Memory DB**。Schema 预留了其他模块的表，但实际只有 Memory 和 Practice 使用。这是可接受的 v1.0 范围，但应在 v1.1 前将 Skill/Know/How 模块接入。

---

## 8. Testing Assessment

### 8.1 现有测试覆盖

| 测试文件 | 测试数 | 覆盖范围 |
|----------|--------|----------|
| `tests/test_memory.py` | 4 | store/query, get, missing, statistics |
| `tests/test_seekdb.py` | 6 | CRUD, SQLite, similarity, auto-ingestion, statistics |
| **总计** | **10** | 基础功能路径 |

### 8.2 缺失的测试

| 缺失测试 | 影响 | 优先级 |
|----------|------|--------|
| EmbodiedMemory 代理方法测试 | 代理路径未验证 | P1 |
| 并发写入安全性 | 多线程下 SQLite 可能损坏 | P1 |
| 大数据量性能测试 | 无性能基线 | P2 |
| SeekDB 断连/重连恢复 | 异常路径未覆盖 | P2 |
| 多 robot_id 隔离验证 | 跨机器人数据泄露风险 | P2 |

---

## 9. Summary Table

| 类别 | 数量 | 详情 |
|------|------|------|
| P0 Issues | **0** | 无发布阻塞 |
| P1 Issues | **5** | ISSUE-001 到 ISSUE-005 |
| P2 Issues | **3** | ISSUE-006 到 ISSUE-008 |
| 缺失测试 | **5** | 见 §8.2 |
| Memory API 方法 | **15** | 4 核心 + 11 代理 |
| SeekDB Schema 表 | **4** | 2 活跃 + 2 预留 |
| 测试用例 | **10** | 基础覆盖 |
| v1.0 代码行数 | **621** | 3 文件 |
| 独立模块代码行数 | **56,505** | 212 文件 |
| 集成比例 | **~1%** | 薄适配层设计合理 |

---

*Audit completed by Memory domain owner. No v1.0 source code was modified.*
