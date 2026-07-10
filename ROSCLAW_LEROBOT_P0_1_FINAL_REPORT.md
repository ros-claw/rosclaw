# ROSClaw × LeRobot Bridge — P0.1 实施报告（最终版）

**报告日期：** 2026-07-09  
**对应分支：** `rosclaw_lerobot_bridge_round1`  
**对应提交：** `3db889b`  
**Pull Request：** https://github.com/ros-claw/rosclaw/pull/56  
**目标仓库：** https://github.com/ros-claw/rosclaw  

---

## 1. 任务目标

完成 ROSClaw × LeRobot Bridge 的 P0.1 修复：把 LeRobot runtime 与 ROSClaw core 解耦，使 ROSClaw 可以继续运行在 Python 3.10/3.11，而 LeRobot 0.6.x 运行在独立的 Python 3.12+ 环境中。

---

## 2. 交付内容

### 2.1 新增模块

| 文件 | 职责 |
|------|------|
| `src/rosclaw/integrations/lerobot/runtime.py` | Python / LeRobot runtime 发现、检查、`find_python312` |
| `src/rosclaw/integrations/lerobot/env_manager.py` | 创建/管理 isolated venv，安装 LeRobot |
| `src/rosclaw/integrations/lerobot/config.py` | `~/.rosclaw/integrations/lerobot.yaml` 读写与 v0→v1 迁移 |
| `tests/integrations/test_lerobot_runtime_isolation.py` | mode、config、runtime 单元/集成测试 |

### 2.2 重写/更新模块

| 文件 | 主要变更 |
|------|----------|
| `src/rosclaw/integrations/lerobot/installer.py` | 支持 `auto/current-env/isolated/external` 四种安装模式 |
| `src/rosclaw/integrations/lerobot/doctor.py` | 区分 ROSClaw Runtime 与 LeRobot Runtime，输出 worker 能力 |
| `src/rosclaw/integrations/lerobot/provider.py` | `--dry-run` 返回 sample action；非 dry-run 返回 `import_smoke`，`action: null` |
| `src/rosclaw/integrations/lerobot/cli.py` | 新 setup 参数、双 runtime doctor、info 使用配置 runtime |
| `src/rosclaw/cli.py` | `setup lerobot` 增加 `--mode/--python/--runtime-path/--force/--index-url/--extra-index-url` |
| `src/rosclaw/integrations/lerobot/schemas.py` | 新增 `LeRobotSetupErrorCode`，扩展 `InstallReport` / `LeRobotDoctorReport` |
| `src/rosclaw/integrations/lerobot/profiles.py` | 解析 `requires_python` 与 `capabilities` |
| `src/rosclaw/integrations/lerobot/profiles.yaml` | 增加 `requires_python` 与 capability 元数据 |
| `src/rosclaw/integrations/lerobot/capabilities.py` | 新增 `worker_subprocess` / `worker_in_process` |
| `src/rosclaw/integrations/lerobot/__init__.py` | 导出新增符号 |
| `tests/integrations/conftest.py` | session 级 `ROSCLAW_HOME` 隔离 |
| `tests/integrations/test_lerobot_doctor.py` | 双 runtime 断言；fake-info 测试在 LeRobot 已安装时 skip |
| `tests/integrations/test_lerobot_provider_dry_run.py` | import_smoke 语义测试 |
| `docs/integrations/lerobot_bridge.md` | 新增 runtime isolation、modes、doctor/provider 语义说明 |

---

## 3. 关键设计决策

- **不在 rosclaw-core 中 import torch/lerobot**：所有 LeRobot 检查都通过子进程 probe 完成。
- **标准库 `venv` 隔离**：P0.1 不使用 conda/uv，降低依赖。
- **`RuntimeMode` 语义**：
  - `auto`：当前 Python >= 3.12 时选 `current-env`，否则选 `isolated`。
  - `current-env`：在当前解释器直接安装/使用 LeRobot。
  - `isolated`：创建独立 venv 并 pip install。
  - `external`：只注册用户已准备好的 Python 3.12+ 环境。
- **退出码约定**：用户/环境错误 → 2，安装失败 → 1，doctor not-installed/degraded → 0。
- **`lerobot-info` 发现优先级**：sibling binary → PATH → `python -m lerobot_info`。

---

## 4. 验证结果

### 4.1 集成测试

```bash
# Python 3.11 ROSClaw 环境
env -u PYTHONPATH .venv/bin/python -m pytest tests/integrations -q --tb=short
# 28 passed in 5.59s

# Python 3.12 LeRobot 环境
env -u PYTHONPATH .venv-lerobot/bin/python -m pytest tests/integrations -q --tb=short
# 26 passed, 2 skipped in 35.85s
```

> 由于 ROS `PYTHONPATH` 会自动加载 `launch_testing` pytest 插件并触发 `ModuleNotFoundError: No module named 'lark'`，测试时必须用 `env -u PYTHONPATH` 运行。

### 4.2 手工冒烟测试

| 命令 | 结果 |
|------|------|
| `setup lerobot --mode auto --dry-run` (Py3.11) | 自动选择 `isolated`，plan 正确 |
| `setup lerobot --mode current-env` (Py3.11) | 拒绝，error_code=`python_too_old`，exit 2 |
| `setup lerobot --mode external --python .venv-lerobot/bin/python` | 注册成功，version 0.6.x |
| `lerobot doctor` (Py3.11 + external runtime) | 显示双 runtime，Status=`INSTALLED` |
| `lerobot info` | 调用配置 runtime 的 lerobot-info，输出正常 |
| `provider infer ... --dry-run` | 返回 sample action，`real_inference=false` |
| `provider infer ...` | 返回 `import_smoke`，`action=null` |
| `setup lerobot --mode auto --dry-run` (Py3.12) | 自动选择 `current-env` |

---

## 5. Pull Request 信息

- **PR 编号：** #56
- **PR 链接：** https://github.com/ros-claw/rosclaw/pull/56
- **标题：** feat(lerobot): P0.1 runtime isolation (auto/current-env/isolated/external)
- **状态：** open
- **变更文件：** 36 个
- **新增行数：** +4391 / -7

---

## 6. 已知限制与后续工作

| 限制 | 说明 |
|------|------|
| isolated 真实 pip install | 受网络环境影响，本环境未跑完全程；dry-run、venv 创建、错误处理路径已验证 |
| 真实 policy inference | P0.1 只到 import smoke，未加载真实 policy |
| 真实 dataset 写入 | `practice export --format lerobot` 仍输出骨架 |
| worker 协议 | 仅预留 `worker_subprocess` / `worker_in_process` 能力标记 |
| conda/uv | P0.1 仅支持标准库 `venv` |

### P1 建议方向

- 实现 LeRobot worker 协议（subprocess / in-process）。
- `provider inspect/load-test/infer` 真实 policy loading smoke。
- 真实 LeRobotDataset 写入（parquet/mp4）。
- 支持 conda/uv runtime。

---

## 7. 安全提示

本次操作用户提供的 GitHub Personal Access Token 已用于推送分支和创建 PR。建议立即到 GitHub Settings → Developer settings → Personal access tokens 中 rotate/revoke 该 Token。

---

**报告生成位置：** `ROSCLAW_LEROBOT_P0_1_FINAL_REPORT.md`  
**生成人：** Claude Code
