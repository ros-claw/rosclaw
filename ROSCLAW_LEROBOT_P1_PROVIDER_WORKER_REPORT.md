# ROSClaw × LeRobot Bridge — P1 Provider Worker 实施报告

**报告日期：** 2026-07-10  
**目标：** 将 LeRobot policy 升级为真正的 ROSClaw Provider，支持 `inspect`、`load-test`、`infer`，但所有 `infer` 结果仍只是 `action_proposal`，标记 `not_executed=true`、`requires_sandbox=true`、`executable=false`。  
**核心机制：** 一次性子进程 worker（`worker_main.py`）+ ROSClaw 侧 runner（`worker_runner.py`）+ JSON 文件协议。  

---

## 1. 交付内容

### 1.1 新增模块

| 文件 | 职责 |
|------|------|
| `src/rosclaw/integrations/lerobot/worker_main.py` | LeRobot 运行时侧 worker，支持 `inspect/load_test/infer`，不 import `rosclaw` |
| `src/rosclaw/integrations/lerobot/worker_runner.py` | ROSClaw 侧 runner：写 request JSON、起子进程、读 response JSON、翻译错误 |
| `src/rosclaw/integrations/lerobot/worker_schema.py` | worker JSON 协议 dataclasses（已在 Task 12 完成） |
| `src/rosclaw/integrations/lerobot/observation_adapter.py` | ROSClaw 输入 → worker observation dict |
| `src/rosclaw/integrations/lerobot/action_adapter.py` | worker action → ROSClaw action_proposal |
| `src/rosclaw/integrations/lerobot/policy_manifest.py` | 本地 `config.json`/`config.yaml` 轻量解析 |
| `tests/integrations/test_lerobot_worker_main.py` | worker_main inspect 单元测试 |
| `tests/integrations/test_lerobot_worker_runner.py` | runner 错误处理与请求构造测试 |
| `tests/integrations/test_lerobot_provider_p1_contract.py` | provider `inspect/load_test/infer` 与安全契约测试 |
| `tests/integrations/test_lerobot_adapters.py` | observation/action adapter 与 manifest 测试 |
| `tests/fixtures/lerobot_policy_minimal/` | 最小 LeRobot policy fixture |
| `examples/lerobot/sample_worker_request_inspect.json` | inspect 请求示例 |
| `examples/lerobot/sample_worker_request_infer.json` | infer 请求示例 |
| `examples/lerobot/sample_policy_manifest_p1.yaml` | P1 provider manifest 示例 |

### 1.2 修改模块

| 文件 | 主要变更 |
|------|----------|
| `src/rosclaw/integrations/lerobot/provider.py` | 支持 `lerobot.policy.inspect/load_test/infer`，dispatch 到 worker runner，强制安全契约 |
| `src/rosclaw/integrations/lerobot/cli.py` | 新增 `cmd_provider_inspect_lerobot`、`cmd_provider_load_test_lerobot`，扩展 `cmd_provider_infer_lerobot` |
| `src/rosclaw/cli.py` | 新增 `provider inspect`、`provider load-test` subparser；扩展 `provider infer` 参数 |
| `src/rosclaw/integrations/lerobot/capabilities.py` | 新增 `real_policy_inspect/load_test/infer`，worker 可用性动态判断 |
| `src/rosclaw/integrations/lerobot/doctor.py` | 使用动态 capability 列表，报告 P1 真实能力就绪状态 |
| `src/rosclaw/integrations/lerobot/__init__.py` | 导出新增符号 |
| `tests/integrations/conftest.py` | 新增 `minimal_policy_dir`、`fake_worker_script*` fixtures |
| `tests/integrations/test_lerobot_provider_dry_run.py` | 适配 P1 语义（`action_proposal`、无 runtime/policy.path 失败） |
| `docs/integrations/lerobot_bridge.md` | 新增 P1 worker 协议、inspect/load-test/infer 用法、安全边界 |

---

## 2. 关键设计决策

- **运行时隔离不变**：ROSClaw core 仍不 import `torch`/`lerobot`；真实 LeRobot 代码只在 `worker_main.py` 中、由 LeRobot runtime Python 执行。
- **一次性 worker**：每次 `inspect/load_test/infer` 起一个新子进程，写完 request、读完 response 后清理临时目录（可通过 `ROSCLAW_LEROBOT_DEBUG_WORKER=1` 保留）。
- **默认 `allow_network=false`、`device=cpu`**：避免意外下载或 GPU 依赖。
- **安全契约**：所有 `infer` 返回 `action_proposal`，`executable=false`，`requires_sandbox=true`，`not_executed=true`；`--execute` 被明确拒绝。
- **错误码结构化**：`LeRobotWorkerErrorCode` 覆盖 `runtime_not_configured`、`worker_timeout`、`worker_invalid_json`、`worker_process_failed`、`policy_config_not_found`、`policy_load_failed`、`policy_infer_failed`、`network_disabled` 等。
- **无 LeRobot 环境也可测试**：所有核心路径使用 fake worker / fixture 跑通，真实 LeRobot runtime 只用于可选的手工冒烟。

---

## 3. Worker 协议

Request（`worker_schema.py`）：

```json
{
  "schema_version": "rosclaw.lerobot.worker.v1",
  "op": "inspect|load_test|infer",
  "policy_path": "<local dir or hf repo id>",
  "revision": "main",
  "device": "cpu",
  "dtype": "auto",
  "allow_network": false,
  "timeout_sec": 120,
  "observation": {
    "task": "...",
    "observation.state": [...],
    "observation.images.front": "<image file path>"
  }
}
```

Response：

```json
{
  "schema_version": "rosclaw.lerobot.worker.v1",
  "status": "ok|error",
  "op": "inspect|load_test|infer",
  "policy_path": "...",
  "real_model_loaded": true|false,
  "real_inference": true|false,
  "policy_metadata": {...},
  "action": {"type": "raw_lerobot_action", "values": [...], "shape": [...], "dtype": "float32"},
  "timing": {"load_time_sec": ..., "infer_time_sec": ...},
  "error": {"code": "...", "message": "...", "details": "..."}
}
```

---

## 4. CLI 用法示例

### Inspect（本地 fixture）

```bash
rosclaw provider inspect \
  --type lerobot_policy \
  --manifest examples/lerobot/sample_policy_manifest_p1.yaml \
  --policy.path tests/fixtures/lerobot_policy_minimal
```

### Load-test（真实 policy 目录）

```bash
rosclaw provider load-test \
  --type lerobot_policy \
  --manifest examples/lerobot/sample_policy_manifest_p1.yaml \
  --policy.path /data/policies/my_policy \
  --device cpu
```

### Infer（真实 action proposal）

```bash
rosclaw provider infer \
  --type lerobot_policy \
  --manifest examples/lerobot/sample_policy_manifest_p1.yaml \
  --policy.path /data/policies/my_policy \
  --input examples/lerobot/sample_observation.json \
  --device cpu
```

输出中的 `action_proposal` 带有 `executable: false` 和 `requires_sandbox: true`。

### Dry-run

```bash
rosclaw provider infer \
  --type lerobot_policy \
  --manifest examples/lerobot/sample_policy_manifest_p1.yaml \
  --input examples/lerobot/sample_observation.json \
  --dry-run
```

---

## 5. 验证结果

### 5.1 集成测试

```bash
env -u PYTHONPATH .venv/bin/python -m pytest tests/integrations -q --tb=short
```

结果：

```text
52 passed in 8.40s
```

> 测试覆盖：worker_main inspect、runner 错误翻译、provider 安全契约、adapter、manifest、原有 P0.1 runtime/doctor 测试。

### 5.2 手工冒烟

| 命令 | 结果 |
|------|------|
| `provider inspect --type lerobot_policy --policy.path tests/fixtures/lerobot_policy_minimal` | `policy_inspect` 成功，metadata 正确 |
| `provider infer --dry-run` | `dry_run` 模式，sample action，安全标记正确 |
| `provider infer --help` | `--policy.path/--device/--allow-network/--timeout-sec/--worker` 参数已出现 |
| `provider inspect --help` / `provider load-test --help` | 新 subparser 正常 |

---

## 6. 当前限制与后续工作

| 限制 | 说明 |
|------|------|
| 真实 policy load_test/infer | 需要已配置 LeRobot runtime 和真实 policy 目录；当前环境已验证 inspect，未验证真实权重加载 |
| 一次性 worker | 每次 inference 都重新加载模型，未做 GPU 常驻或 batch 优化 |
| worker 不支持 persistent daemon | P2/P3 按需引入 |
| `--worker in-process` | CLI 保留选项，但当前 provider 统一使用 subprocess runner |
| dataset 真实写入 | 仍是 P2 目标 |
| eval/rollout/reward backend | 仍是未来能力 |

---

## 7. 安全提示

- P1 明确拒绝 `--execute`；任何 `infer` 结果都是提案，不会直接驱动机器人。
- `worker_main.py` 不 import `rosclaw`，避免跨运行时污染。
- 子进程环境默认 `HF_HUB_OFFLINE=1`（`allow_network=false` 时），并移除 `PYTHONPATH`，防止 ROS pytest 插件等泄露。

---

**报告生成位置：** `ROSCLAW_LEROBOT_P1_PROVIDER_WORKER_REPORT.md`  
**生成人：** Claude Code
