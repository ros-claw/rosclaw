# ROSClaw SeekDB 实践事件集成

本文档说明两种不同的 SeekDB 集成：

1. `SeekDBMySQLClient` 通过 MySQL-compatible SQL 协议直连 SeekDB/OceanBase，
   用于 `practice ingest-seekdb` 和 `practice query`。
2. 旧 `SeekDBBridge` 通过独立的 HTTP adapter 转发单个 `PraxisEvent`。

SeekDB 原生 `2881` 端口不是 HTTP API，不能使用
`http://localhost:2881/api/v1/insert`。

---

## 目标

`EpisodeRecorder` 在每次实践episode结束时已经组装出完整的 `PraxisEvent`，并写入本地 artifact 目录、发布 `praxis.recorded` 事件。本集成在此基础上，把同样的事件对象经 `SeekDBBridge` 转发到 SeekDB，使经验数据进入统一存储，便于后续检索、回放与模型训练。

---

## 前置条件

- ROSClaw v1.0 主仓库已安装。
- 直连 server 时，SeekDB 可通过
  `mysql://root@127.0.0.1:2881/rosclaw` 访问。
- 仅使用旧 HTTP bridge 时，可选依赖 `rosclaw[practice]` 已安装
  （内部指向 `rosclaw-practice` 包）：

  ```bash
  pip install -e ".[practice]"
  ```

---

## 直接写入真实 SeekDB

```bash
rosclaw practice ingest-seekdb <practice_id> \
  --data-root /data/rosclaw/practice \
  --seekdb-url mysql://root@127.0.0.1:2881/rosclaw \
  --json

rosclaw practice query failures \
  --robot-id rh56 \
  --data-root /data/rosclaw/practice \
  --seekdb-url mysql://root@127.0.0.1:2881/rosclaw \
  --json
```

客户端会创建数据库、Knowledge Plane 表和索引。重复 ingest 使用稳定 evidence ID
做幂等 upsert。

---

## 启用旧 HTTP Bridge

### 1. 安装可选依赖

```bash
python3 -m pip install -e ".[practice]"
```

### 2. 配置环境变量

```bash
export ROSCLAW_SEEKDB_URL=http://seekdb-adapter.example:8080
# 可选：离线失败时的 JSON 落盘目录
export ROSCLAW_SEEKDB_FALLBACK_DIR=/data/rosclaw/fallback
```

### 3. 启动 Runtime

```python
from rosclaw.core import Runtime, RuntimeConfig

config = RuntimeConfig(
    robot_id="ur5e_lab_01",
    enable_practice=True,
    seekdb_url="http://seekdb-adapter.example:8080",
    seekdb_fallback_dir="/data/rosclaw/fallback",
)
runtime = Runtime(config)
runtime.initialize()
```

当 `seekdb_url` 为空或未设置时，`Runtime` 不会创建 `SeekDBBridge`，`EpisodeRecorder` 行为与之前完全一致。

---

## RuntimeConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `seekdb_url` | `str \| None` | `ROSCLAW_SEEKDB_URL` 环境变量 | 旧 HTTP adapter 地址，不是原生 2881 SQL 端口 |
| `seekdb_fallback_dir` | `str` | `ROSCLAW_SEEKDB_FALLBACK_DIR` 或 `/data/rosclaw/fallback` | 提交失败时的本地 JSON 落盘目录 |

---

## 数据映射

`SeekDBBridge._convert()` 把 ROSClaw 的 `PraxisEvent`（frozen dataclass）映射为 `rosclaw_practice` 的 Pydantic `PraxisEvent`：

| ROSClaw `PraxisEvent` | `rosclaw_practice` `PraxisEvent` | 说明 |
|-----------------------|----------------------------------|------|
| `event_id` | `practice_id` | episode 唯一标识 |
| `timestamp`（float） | `timestamp`（ISO 8601 UTC） | 转换时格式化为 UTC 时间字符串 |
| `robot_id` | `robot_id` | 机器人标识 |
| `agent_instruction` | `cognitive_context.semantic_intent` | 自然语言指令 |
| `cot_trace`（list） | `cognitive_context.llm_cot` | 用 `\n` 连接 |
| `event_type` | `physical_feedback.status` | 转大写，如 `SUCCESS` / `FAILURE` |
| `metadata["reward"]` | `physical_feedback.reward` | 缺失时默认 `0.0` |
| `error_details` | `physical_feedback.error_log` | 缺失时为空字符串 |
| `mcap_path` | `data_pointers.mcap_path` | 当前 `EpisodeRecorder` 中一般为 `None`，映射为空字符串 |

---

## 失败降级

- `SeekDBBridge.commit()` 内部使用 `requests.post` 在 `2.0` 秒超时内向 `POST {seekdb_url}/api/v1/insert` 提交事件。
- 该路径要求额外部署兼容的 HTTP adapter；原生 SeekDB server 不提供此 HTTP API。
- 任何网络或服务器错误都会被捕获，并将事件以 JSON 形式写入 `seekdb_fallback_dir`。
- `EpisodeRecorder` 对 `SeekDBBridge.commit()` 再做一层 `try/except`，确保 SeekDB 提交失败不会中断本地 artifact 写入，也不会阻止 `praxis.recorded` 事件发布。

---

## 验证

运行相关测试：

```bash
python3 -m pytest tests/test_seekdb.py tests/practice/test_cli_ingest_seekdb.py tests/practice/test_seekdb_ingestor.py -v
```

离线验证 fallback：关闭 SeekDB 后执行一次 skill，检查 `ROSCLAW_SEEKDB_FALLBACK_DIR` 目录下是否生成新的 `.json` 文件。

---

## 相关链接

- `rosclaw_practice` 独立仓库文档与示例：`https://github.com/ros-claw/rosclaw-practice`
- 主仓库 `SeekDBBridge` 实现：`src/rosclaw/practice/seekdb_bridge.py`
- `EpisodeRecorder` 集成点：`src/rosclaw/practice/episode_recorder.py`
