# RH56 RPS 本地 SeekDB / Practice 数据查看教程

这份文档说明 RH56 猜拳 demo 的本地数据闭环存在哪里、表结构是什么、以及怎么用 `rosclaw practice` CLI 和 `sqlite3` 直接查看。

## 1. 数据存在哪里

配置文件：`examples/rh56_rps/configs/dual/rps_rosclaw.yaml`

```yaml
practice:
  data_root: ~/.rosclaw/practice/runs/rh56_rps

memory:
  backend: sqlite
  db_path: ~/.rosclaw/practice/runs/rh56_rps/seekdb.sqlite
```

实际路径（`~` 会展开为 `/home/nvidia`）：

```text
~/.rosclaw/practice/runs/rh56_rps/
├── indexes/
│   └── practice_catalog.sqlite      # PracticeCatalog v2（运行后实时写入）
├── seekdb.sqlite                    # SeekDB SQLite 后端（默认不自动写入，需 ingest）
├── sessions/
│   ├── prac_<timestamp>_<hash>/     # 运行时主目录（practice_id 命名）
│   │   ├── manifest.yaml            # practice 元信息、session_id、body_id、outcome
│   │   ├── episode.json             # episode 摘要
│   │   ├── timeline.jsonl           # 按时间线展开的事件
│   │   ├── raw/
│   │   │   ├── events.jsonl         # 原始事件流
│   │   │   └── events.mcap          # MCAP 录像（含帧、telemetry）
│   │   ├── keyframes/               # 关键帧图
│   │   └── ...                      # 其他运行时产物
│   └── sess_<timestamp>_<hash>/     # ArtifactStore 生成的 summary artifact 目录
│       └── episodes/<episode_id>/artifacts/summary/
│           ├── summary.yaml
│           └── artifact_manifest.yaml
```

> 注意：
> - 默认 `practice.seekdb.enabled: false`，所以运行结束后 **只有 PracticeCatalog 有数据**，SeekDB 是空的。需要执行 `rosclaw practice ingest-seekdb` 才会把 episode / body_cognition 等蒸馏结果写进 SeekDB。
> - `manifest.yaml` 里的 `session_id` 是 `sess_...`，但文件系统上的运行时目录以 `prac_...`（practice_id）命名。`practice_artifacts.path` 可能仍指向 `sess_...` 路径，而原始事件/MCAP/帧图等都在 `prac_...` 路径下。

## 2. 用 `rosclaw practice` CLI 查看（推荐）

先设置 PYTHONPATH（如果在仓库外运行）：

```bash
export PYTHONPATH="/home/nvidia/workspace/rosclaw/rosclaw_test/rosclaw/src:/home/nvidia/workspace/rosclaw_rh56_real/rosclaw-rh56-runtime/src:/home/nvidia/workspace/rosclaw/rosclaw_test/examples/rh56_rps/src:${PYTHONPATH}"
PYTHON=/home/nvidia/workspace/rosclaw/rosclaw_test/.venv/bin/python
```

### 2.1 校验一次运行是否完整

```bash
${PYTHON} -m rosclaw.cli practice verify \
  prac_20260714T093707Z_212a83 \
  --data-root ~/.rosclaw/practice/runs/rh56_rps \
  --strict
```

示例输出：

```json
{
  "practice_id": "prac_20260714T093707Z_212a83",
  "passed": true,
  "strict": true,
  "checked": [
    "catalog_exists", "practice_record", "manifest_exists",
    "events_jsonl_exists", "session_record", "episode_records",
    "event_types", "event_count", "artifact_records"
  ],
  "issues": []
}
```

### 2.2 查询 episode

```bash
${PYTHON} -m rosclaw.cli practice query episodes \
  --body rh56_rps_robot \
  --data-root ~/.rosclaw/practice/runs/rh56_rps \
  --json | head -40
```

你会看到类似：

```json
{
  "episode_id": "ep_20260714T093707Z_c27576c6",
  "session_id": "sess_20260714T093707Z_477cb614c8f8",
  "body_id": "rh56_rps_robot",
  "skill_id": "rh56_rps",
  "outcome": "success",
  "success": true,
  "metrics_json": {
    "duration_ms": 54392.055896,
    "source_event_count": 1504
  }
}
```

### 2.3 把最新运行灌入 SeekDB

```bash
${PYTHON} -m rosclaw.cli practice ingest-seekdb \
  prac_20260714T093707Z_212a83 \
  --data-root ~/.rosclaw/practice/runs/rh56_rps \
  --seekdb-path ~/.rosclaw/practice/runs/rh56_rps/seekdb.sqlite
```

示例输出：

```json
{
  "practice_id": "prac_20260714T093707Z_212a83",
  "episode_id": "ep_20260714T093707Z_c27576c6",
  "success": true,
  "table_counts": {
    "episodes": 1,
    "body_cognition": 1
  },
  "total_records": 2
}
```

## 3. 直接用 `sqlite3` 看 SeekDB

### 3.1 连接

```bash
sqlite3 ~/.rosclaw/practice/runs/rh56_rps/seekdb.sqlite
```

### 3.2 常用查询

查看有哪些表：

```sql
.tables
.schema episodes
.schema events
.schema body_cognition
```

最新 5 个 episode：

```sql
SELECT
  id,
  robot_id,
  task_id,
  datetime(started_at, 'unixepoch') AS started,
  outcome
FROM episodes
ORDER BY started_at DESC
LIMIT 5;
```

查看 events 表（RPS 原始事件都先写这里）：

```sql
SELECT
  event_type,
  COUNT(*) AS cnt
FROM events
GROUP BY event_type
ORDER BY cnt DESC
LIMIT 20;
```

查看 body_cognition：

```sql
SELECT
  id,
  body_id,
  cognition_type,
  datetime(timestamp, 'unixepoch') AS ts,
  data
FROM body_cognition
WHERE body_id = 'rh56_rps_robot'
ORDER BY timestamp DESC
LIMIT 3;
```

查看 failures / how_interventions（当前 RPS 没产生 failure，所以是空的）：

```sql
SELECT * FROM failures ORDER BY timestamp DESC LIMIT 5;
SELECT * FROM how_interventions ORDER BY timestamp DESC LIMIT 5;
```

## 4. 直接看 PracticeCatalog

PracticeCatalog 是运行时的主索引，episodes 等数据先写到这里，再被 ingest 进 SeekDB。
当前 RPS 运行会把事件写入 `events` 表，`practice_event_index` 表可能为空（取决于 recorder 是否启用索引写入）。

```bash
sqlite3 ~/.rosclaw/practice/runs/rh56_rps/indexes/practice_catalog.sqlite
```

常用表：

```sql
-- 所有 practice session
SELECT practice_id, session_id, start_time, outcome
FROM practices
ORDER BY start_time DESC
LIMIT 10;

-- episode 表（v2 catalog）
SELECT episode_id, session_id, body_id, skill_id, outcome, success
FROM practice_episodes
ORDER BY started_at DESC
LIMIT 10;

-- artifact 索引
SELECT artifact_id, artifact_type, path, sha256
FROM practice_artifacts
ORDER BY created_at DESC
LIMIT 10;

-- 事件索引（当前 RPS 数据主要写入 events 表；practice_event_index 表可能为空）
SELECT event_id, event_type, timestamp_utc
FROM events
ORDER BY timestamp_ns DESC
LIMIT 10;
```

## 5. 查看原始事件与视频

### 5.1 events.jsonl

```bash
ls ~/.rosclaw/practice/runs/rh56_rps/sessions/prac_20260714T093707Z_212a83/
head -3 ~/.rosclaw/practice/runs/rh56_rps/sessions/prac_20260714T093707Z_212a83/raw/events.jsonl
```

每行是一个 `PracticeEventEnvelope`，包含 event_type、payload、timestamp、tags 等。

### 5.2 MCAP

可以用 `mcap` 工具或 Python 读取：

```bash
# 列出 topic
mcap info ~/.rosclaw/practice/runs/rh56_rps/sessions/prac_20260714T093707Z_212a83/raw/events.mcap

# 导出某 topic
mcap cat ~/.rosclaw/practice/runs/rh56_rps/sessions/prac_20260714T093707Z_212a83/raw/events.mcap --topic /camera/camera/color/image_raw
```

### 5.3 帧图

keyframes 存在 `prac_.../keyframes/` 目录下，通常是 JPEG/PNG 序列，可用任意图片查看器打开。

## 6. 关键表结构速查

### SeekDB

> 说明：实际 SeekDB 远不止下面列出的这几张表（执行 `.tables` 可看到约 25 张，包括 `auto_experiments`、`auto_patches`、`knowledge_graph`、`providers`、`robots`、`tasks` 等）。下面只列出与 **Practice 闭环** 最相关的表。

| 表 | 用途 |
|---|---|
| `episodes` | 一次 episode（RPS 一局或一次完整练习） |
| `praxis_events` | 运行时事件（默认 `ingest-seekdb` **不灌入**，所以通常是空的） |
| `body_cognition` | 身体认知（力基线、接触分布、热限等；RPS 当前可能只写入空占位） |
| `sim2real_deltas` | 仿真 vs 真机差异 |
| `failures` | 失败记录 |
| `how_interventions` | 失败后的干预/经验教训 |
| `skill_candidates` | 候选策略 |
| `promotion_results` | 策略晋升结果 |
| `artifacts` | 数据产物（MCAP、Parquet、视频等） |

### PracticeCatalog v2

| 表 | 用途 |
|---|---|
| `practices` | practice session 主记录 |
| `practice_sessions` | session 级索引（body_id、状态、统计） |
| `practice_episodes` | episode 级索引 |
| `practice_artifacts` | artifact 索引（含 sha256） |
| `practice_event_index` | 事件索引（用于快速定位） |
| `events` / `failures` / `artifacts` | 旧版兼容表 |

## 7. 小贴士

- `body_id: rh56_rps_robot` 是在 `rps_rosclaw.yaml` 里显式指定的，这样 SeekDB 和 Catalog 都能按 body 查询。
- 如果 `seekdb.enabled: false`，运行后 SeekDB 不会自动更新；想长期自动写入，把配置改成 `seekdb.enabled: true`。
- 当前 `rosclaw practice ingest-seekdb` 默认只写入 `episodes` 和 `body_cognition`，**不会自动灌入 `praxis_events`**。需要事件进 SeekDB 时要额外配置或调用 MemoryConsumer。
- RPS demo 目前没有 `physical_feedback_event` 事件类型，所以 **不能直接用 `--format lerobot` 导出**。想导出 RPS 数据请用 `jsonl` / `json` / `parquet`。

想看 LeRobot/Parquet 导出（仅适用于有力反馈事件的 practice）：

```bash
${PYTHON} -m rosclaw.cli practice export \
  prac_20260714T093707Z_212a83 \
  --format parquet \
  --data-root ~/.rosclaw/practice/runs/rh56_rps
```

```bash
# 注意：RPS 数据不支持 LeRobot，因为缺少 physical_feedback_event
${PYTHON} -m rosclaw.cli practice export \
  prac_20260714T093707Z_212a83 \
  --format lerobot \
  --data-root ~/.rosclaw/practice/runs/rh56_rps
```

- 如果 sqlite3 输出乱码，可以加 `.mode columns` / `.headers on` 美化。

## 8. 文档路径

本文件位于：

```text
examples/rh56_rps/docs/seekdb_tutorial.md
```

对应的本地数据根目录：

```text
/home/nvidia/.rosclaw/practice/runs/rh56_rps/
```
