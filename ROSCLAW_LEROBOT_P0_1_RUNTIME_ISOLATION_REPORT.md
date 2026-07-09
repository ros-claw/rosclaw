# ROSClaw × LeRobot Bridge — P0.1 Runtime Isolation Report

**报告日期：** 2026-07-09  
**分支：** `rosclaw_lerobot_bridge_round1`  
**对应提交：** `95ac6d1`  
**目标仓库：** https://github.com/ros-claw/rosclaw

---

## 1. 为什么需要 P0.1

第一轮（P0/P1）已经搭好了 LeRobot 桥接骨架，但存在关键产品化缺口：

> 支持的 LeRobot 0.6.x 需要 Python >= 3.12，而 ROSClaw core 不应该因此被强制升级到 Python 3.12。

真实用户环境通常是：

```text
ROSClaw 主环境：Python 3.11+
ROS2 / RealSense / 机器人 SDK：Python 3.10 / 3.11
LeRobot：Python 3.12+
```

P0.1 的目标不是继续堆功能，而是把 **LeRobot runtime 隔离机制** 做稳，使 ROSClaw 可以在 Python 3.11+ 下运行，同时通过独立的 Python 3.12 环境使用 LeRobot。

---

## 2. 修改了哪些文件

### 新增

| 文件 | 职责 |
|------|------|
| `src/rosclaw/integrations/lerobot/runtime.py` | Python / LeRobot runtime 发现、检查、`find_python312` |
| `src/rosclaw/integrations/lerobot/env_manager.py` | 创建/管理 isolated venv，安装 LeRobot |
| `src/rosclaw/integrations/lerobot/config.py` | `~/.rosclaw/integrations/lerobot.yaml` 读写与 v0→v1 迁移 |
| `tests/integrations/test_lerobot_runtime_isolation.py` | mode、config、runtime 测试 |

### 修改

| 文件 | 主要变更 |
|------|----------|
| `src/rosclaw/integrations/lerobot/installer.py` | 支持 `auto/current-env/isolated/external` 四种模式 |
| `src/rosclaw/integrations/lerobot/doctor.py` | 区分 ROSClaw Runtime 与 LeRobot Runtime |
| `src/rosclaw/integrations/lerobot/provider.py` | `provider infer` 非 dry-run 返回 `import_smoke`，`action: null` |
| `src/rosclaw/integrations/lerobot/cli.py` | 新 setup 参数、新 doctor 输出、info 使用配置 runtime |
| `src/rosclaw/integrations/lerobot/schemas.py` | 新增 `LeRobotSetupErrorCode`，扩展 `InstallReport` / `LeRobotDoctorReport` |
| `src/rosclaw/integrations/lerobot/profiles.py` | 解析 `requires_python` 与 `capabilities` 元数据 |
| `src/rosclaw/integrations/lerobot/profiles.yaml` | 增加 `requires_python` 和 capability 元数据 |
| `src/rosclaw/integrations/lerobot/capabilities.py` | 静态列表增加 `worker_subprocess` / `worker_in_process` |
| `src/rosclaw/integrations/lerobot/__init__.py` | 导出新增模块 |
| `src/rosclaw/cli.py` | `setup lerobot` 增加 `--mode/--python/--runtime-path/--force/--index-url/--extra-index-url` |
| `tests/integrations/conftest.py` | session 级 `ROSCLAW_HOME` 隔离，避免测试互相污染 |
| `tests/integrations/test_lerobot_doctor.py` | 增加双 runtime 断言；fake-info 测试在 LeRobot 已安装时 skip |
| `tests/integrations/test_lerobot_provider_dry_run.py` | 增加 import_smoke 语义测试 |
| `docs/integrations/lerobot_bridge.md` | 新增 runtime isolation、modes、doctor/provider 语义说明 |

---

## 3. 新增了哪些 CLI 参数

```bash
rosclaw setup lerobot --profile core \
  --mode [auto|current-env|isolated|external] \
  --python /path/to/python3.12 \
  --runtime-path /path/to/runtime \
  --upgrade \
  --force \
  --dry-run \
  --index-url URL \
  --extra-index-url URL
```

| 参数 | 说明 |
|------|------|
| `--mode` | `auto`（默认）/ `current-env` / `isolated` / `external` |
| `--python` | external 模式必须；isolated 模式可指定 Python 3.12 |
| `--runtime-path` | isolated 模式目标 venv 路径，默认 `~/.rosclaw/envs/lerobot` |
| `--upgrade` | 即使已安装也重新 pip install --upgrade |
| `--force` | 重建 isolated runtime 或覆盖已有 config |
| `--dry-run` | 只输出计划，不执行 |
| `--index-url` / `--extra-index-url` | pip 源配置 |

---

## 4. 三种 setup 模式如何工作

### 4.1 auto（默认）

```text
if current_python >= 3.12:
    mode = current-env
else:
    mode = isolated
```

### 4.2 current-env

- 要求当前 Python >= 3.12，否则返回 `error_code=python_too_old`，exit code 2。
- 已安装且未 `--upgrade` 时跳过 pip install。
- 运行 post-install checks，写入 `install_mode=current-env` 的配置。

### 4.3 isolated

- 查找 `python3.12`（或 `--python` 指定）。
- 找不到时返回 `error_code=python312_not_found`，提示 `--mode external --python ...`。
- 在 `--runtime-path` 创建 venv，升级 pip，安装 LeRobot，执行 `lerobot-info` smoke。
- 写入 `install_mode=isolated` 的配置。

### 4.4 external

- 必须传 `--python`。
- 检查 Python >= 3.12、可 import `lerobot`、可调用 `lerobot-info`。
- 仅注册已有环境，不创建 venv，不执行 pip install。

---

## 5. Python 3.11 ROSClaw + Python 3.12 LeRobot 验证

环境：

- ROSClaw: `.venv` (Python 3.11.15)
- LeRobot: `.venv-lerobot` (Python 3.12.13, LeRobot 0.6.0, torch 2.11.0+cu128)

### 5.1 auto 模式 dry-run

```bash
PYTHONPATH=src .venv/bin/python -m rosclaw.cli setup lerobot --profile core --mode auto --dry-run
```

输出：

```text
[rosclaw-lerobot] Mode: isolated
[rosclaw-lerobot] OK: True
Dry-run: resolved mode='isolated'. Planned steps:
  - Find Python 3.12 executable (preferred: auto)
  - Create isolated venv at /home/dell/.rosclaw/envs/lerobot
  - Upgrade pip in isolated runtime
  - pip install lerobot in isolated runtime
  - Run lerobot-info smoke test
  - Write config with install_mode=isolated
```

### 5.2 external 模式注册已有环境

```bash
PYTHONPATH=src .venv/bin/python -m rosclaw.cli setup lerobot \
  --profile core --mode external \
  --python /code/rosclaw/rosclaw_lerobot/rosclaw_repo/.venv-lerobot/bin/python
```

输出：

```text
[rosclaw-lerobot] Mode: external
[rosclaw-lerobot] OK: True
[rosclaw-lerobot] LeRobot external runtime registered: .../.venv-lerobot/bin/python
[rosclaw-lerobot] LeRobot version: 0.6.0
```

### 5.3 doctor

```bash
PYTHONPATH=src .venv/bin/python -m rosclaw.cli lerobot doctor
```

输出：

```text
ROSClaw × LeRobot Bridge Doctor

ROSClaw Runtime
  Python executable: .../.venv/bin/python
  Python version:    3.11.15
  In-process LeRobot import: no

LeRobot Runtime
  Mode:              external
  Python executable: .../.venv-lerobot/bin/python
  Python version:    3.12.13
  LeRobot version:   0.6.0
  lerobot-info:      ok
  Torch:             2.11.0+cu128
  CUDA:              available

Bridge Capabilities
  provider_type_lerobot_policy:     enabled
  dataset_export_lerobot:           enabled
  worker_subprocess:                enabled
  worker_in_process:                disabled

Status: INSTALLED
```

### 5.4 current-env 在 Python 3.11 下被拒绝

```bash
PYTHONPATH=src .venv/bin/python -m rosclaw.cli setup lerobot --mode current-env
# exit code 2
# Error code: python_too_old
```

---

## 6. Python 3.12 ROSClaw 环境验证

```bash
PYTHONPATH=src .venv-lerobot/bin/python -m rosclaw.cli setup lerobot --mode auto --dry-run
# Mode: current-env

PYTHONPATH=src .venv-lerobot/bin/python -m rosclaw.cli lerobot doctor
# worker_in_process: enabled
# worker_subprocess: enabled
```

---

## 7. provider infer 语义修正

### 7.1 dry-run

```bash
rosclaw provider infer --type lerobot_policy ... --dry-run
```

返回 sample action，但明确标记：

```json
{
  "mode": "dry_run",
  "dry_run": true,
  "real_inference": false,
  "not_executed": true,
  "action": [0.0, ...],
  "safety": { "executable": false, "sandbox_required": true }
}
```

### 7.2 非 dry-run（import smoke）

```bash
rosclaw provider infer --type lerobot_policy ...
```

P0.1 不做真实推理，只验证 LeRobot runtime 可导入：

```json
{
  "mode": "import_smoke",
  "real_inference": false,
  "action": null,
  "lerobot_smoke": {
    "runtime_mode": "external",
    "python_executable": ".../.venv-lerobot/bin/python",
    "import_ok": true,
    "version": "0.6.0"
  }
}
```

---

## 8. 新增测试列表

| 测试文件 | 覆盖点 |
|----------|--------|
| `tests/integrations/test_lerobot_runtime_isolation.py` | auto mode 选择、current-env 拒绝、isolated 缺 python3.12、isolated dry-run plan、external 缺失/旧 Python、v0→v1 config 迁移、`inspect_python` |
| `tests/integrations/test_lerobot_provider_dry_run.py` | dry-run sample action、非 dry-run import_smoke、未知 capability 拒绝 |
| `tests/integrations/test_lerobot_doctor.py` | ROSClaw / LeRobot runtime 字段、worker_subprocess/in_process |

测试结果：

```text
# LeRobot integration 回归
38 passed

# 仓库完整测试
3750 passed, 30 skipped, 15 deselected

# Python 3.12 外部 LeRobot 环境
LeRobot 0.6.0 setup / doctor / info / provider import smoke 均通过
```

---

## 9. 手工测试命令与结果

| 命令 | 结果 |
|------|------|
| `setup lerobot --mode auto --dry-run` (Py3.11) | 自动选择 isolated，plan 正确 |
| `setup lerobot --mode current-env` (Py3.11) | 拒绝，error_code=python_too_old，exit 2 |
| `setup lerobot --mode external --python .venv-lerobot/bin/python` | 注册成功，version 0.6.0 |
| `lerobot doctor` (Py3.11 + external runtime) | 显示双 runtime，Status=INSTALLED |
| `lerobot info` | 调用配置 runtime 的 lerobot-info，输出正常 |
| `provider infer ... --dry-run` | 返回 sample action，real_inference=false |
| `provider infer ...` | 返回 import_smoke，action=null |
| `setup lerobot --mode auto --dry-run` (Py3.12) | 自动选择 current-env |

---

## 10. 当前限制

- **isolated 模式真实 pip install 受网络环境影响**：本次实测因 PyPI 超时未能完成完整 isolated 安装，但 dry-run、venv 创建、错误处理路径已验证。
- **P0.1 不执行真实 policy inference**：`provider infer` 只到 import smoke。
- **显式 episode 目录只输出骨架**：`--episode <directory>` 不写入真实帧；位置参数 practice ID 仍使用既有真实 Parquet 导出器。
- **worker 协议未实现**：仅预留 `worker_subprocess` / `worker_in_process` 能力标记。
- **conda/uv 未支持**：仅使用标准库 `venv`。

---

## 11. P1 建议

P0.1 已经把 runtime 边界打稳，P1 可以进入真实 LeRobot worker 协议：

```bash
rosclaw provider inspect lerobot/<policy>
rosclaw provider load-test lerobot/<policy>
rosclaw provider infer lerobot/<policy> --worker isolated --input sample_observation.json
```

P1 目标：真实 policy loading smoke，但仍不下发真机，所有 action 标记 `executable=false` + `requires_sandbox=true`。

---

**报告人：** Claude Code  
**生成位置：** `ROSCLAW_LEROBOT_P0_1_RUNTIME_ISOLATION_REPORT.md`
