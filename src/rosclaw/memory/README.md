# ROSClaw Memory Module

## Architecture

```
MemoryInterface (体验存储 + 语义搜索)
    ├── SeekDBClient (抽象接口)
    │       ├── SeekDBMemoryClient (内存版，测试用)
    │       └── SeekDBSQLiteClient (SQLite版，生产用)
    ├── KnowledgeGraphWrapper (KNOW 三元组)
    ├── HeuristicRuleWrapper (HOW 启发规则)
    └── EmbodiedMemory bridge (可选，物理智能)
            ├── WorldObjectStore (物体持久化)
            ├── TrajectoryMemory (轨迹 + DTW)
            ├── CognitiveRouter (语义/空间/时序)
            └── ObjectPermanenceTracker (遮挡推理)
```

## EventBus 集成

**订阅:**
- `praxis.recorded` → 自动摄入 experience
- `rosclaw.practice.event.created` → 写入 praxis_events 表
- `rosclaw.sandbox.episode.failed` → 写入 failures 表
- `rosclaw.sandbox.episode.succeeded` → 写入 success_patterns 表
- `rosclaw.how.recovery_hint.generated` → 关联 recovery_hint
- `firewall.action_blocked` → 写入 safety failure

**发布:**
- `memory.experience.stored`
- `rosclaw.memory.write.completed`

## Sprint 8/9 新增表

| 表名 | 用途 |
|------|------|
| `robots` | 机器人注册信息 |
| `providers` | 能力提供商 |
| `skills` | 技能元数据 |
| `tasks` | 任务队列 |
| `episodes` | 执行片段 |
| `praxis_events` | 实践事件 |
| `failures` | 故障案例 |
| `success_patterns` | 成功模式 |
| `benchmarks` | 评测结果 |
| `artifacts` | 大文件引用 |
| `retries` | 重试状态 (Sprint 9) |

## 类型导出

```python
from rosclaw.memory import (
    MemoryInterface,
    SeekDBClient, SeekDBMemoryClient, SeekDBSQLiteClient,
    PraxisEvent, FailureMemory, ArtifactRef,
)
```

## 常量说明

- `DEFAULT_MAX_EXPERIENCES = 10_000` — experience_graph 容量上限，超过时自动驱逐最旧记录
- `DEFAULT_MAX_AGE_DAYS = 30` — `forget_old_experiences` 默认保留天数
