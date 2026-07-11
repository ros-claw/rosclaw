# ROSClaw × LeRobot Bridge — 第一轮开发报告

**报告日期：** 2026-07-09  
**分支：** `rosclaw_lerobot_bridge_round1`  
**对应提交：** `dff5375`（报告生成时最新）  
**目标仓库：** https://github.com/ros-claw/rosclaw

---

## 1. 背景与目标

在第一轮（P0/P1）中，为 ROSClaw 引入一个**可选的** LeRobot 桥接层，使得：

- `rosclaw-core` 不强制依赖 LeRobot / torch / HuggingFace；
- 用户在安装 LeRobot 后可以通过 ROSClaw CLI 完成安装诊断、能力发现、策略推理 dry-run、以及 practice episode 的 LeRobotDataset v3 骨架导出；
- 当 LeRobot 未安装时，所有命令都能优雅降级，不崩溃。

## 2. 已完成交付物

| 命令 | 状态 | 说明 |
|------|------|------|
| `rosclaw setup lerobot --profile core` | ✅ 已实现 | 安装 LeRobot，dry-run，跳过已安装，Python 3.12+ 校验 |
| `rosclaw setup lerobot --profile core --dry-run` | ✅ 已实现 | 仅展示将要执行的安装步骤 |
| `rosclaw lerobot doctor` | ✅ 已实现 | 报告 LeRobot 导入状态、版本、torch/CUDA、HF 镜像、配置状态 |
| `rosclaw lerobot info` | ✅ 已实现 | 调用 `lerobot-info` 并透传输出 |
| `rosclaw capability list` | ✅ 已实现 | 列出 LeRobot 提供的能力（已安装/未安装） |
| `rosclaw provider infer --type lerobot_policy ... --dry-run` | ✅ 已实现 | 返回 sample action + 安全元数据 |
| `rosclaw provider infer --type lerobot_policy ...` | ✅ 已实现 | 真实 LeRobot import smoke（P0 不执行真实推理） |
| `rosclaw practice export --format lerobot --episode <dir> --output <dir>` | ✅ 已实现 | 导出 LeRobotDataset v3 骨架 |
| 单元/集成测试 | ✅ 16 个 | `tests/integrations/` 全覆盖 |
| 示例与文档 | ✅ 已提供 | `examples/lerobot/`, `examples/practice/minimal_episode/`, `docs/integrations/lerobot_bridge.md` |

## 3. 关键实现

### 3.1 集成注册表（无依赖）

- `src/rosclaw/integrations/registry.py`
  - `IntegrationRegistry` / `IntegrationCapability` / `IntegrationReport`
  - `GLOBAL_INTEGRATION_REGISTRY` 单例
  - 注册 provider type、practice exporter、integration report

### 3.2 LeRobot 集成包

`src/rosclaw/integrations/lerobot/` 下包含：

| 文件 | 职责 |
|------|------|
| `installer.py` | `rosclaw setup lerobot` 的安装逻辑 |
| `doctor.py` | 环境诊断 |
| `capabilities.py` | 能力注册与静态元数据 |
| `provider.py` | `LeRobotPolicyProvider`（P0 dry-run + import smoke） |
| `dataset_exporter.py` | LeRobotDataset v3 骨架导出 |
| `feature_mapping.py` | ROSClaw ↔ LeRobot 字段映射（预留） |
| `profiles.py` / `profiles.yaml` | 安装配置文件 |
| `cli.py` | LeRobot 相关 CLI 分发函数 |
| `schemas.py` | `InstallReport`、`LeRobotDoctorReport` 等数据类 |
| `subprocess_runner.py` | 子进程命令封装 |

### 3.3 CLI 接入

`src/rosclaw/cli.py`：

- 在 `main()` 初始化时调用 `register_lerobot_capabilities(GLOBAL_INTEGRATION_REGISTRY)`；
- 新增 `setup lerobot`、`lerobot {doctor,info,capabilities}`、`capability list`、`provider infer --type lerobot_policy` 等子命令；
- `practice export --format lerobot` 在 `--episode` 为目录时走骨架导出器。

### 3.4 骨架导出包装器

`src/rosclaw/practice/exporters/lerobot_skeleton_exporter.py`：

- 包装 `LeRobotDatasetExporter.export_from_episode_dir()`；
- 支持相对路径（在 `data_root` 下解析）和绝对路径（直接作为 episode 目录）。

## 4. 第一轮迭代中解决的问题

| 问题 | 解决方案 |
|------|----------|
| `pip install lerobot` 在已安装环境下耗时 10 分钟超时 | 安装器检测 LeRobot 已可导入且未传 `--upgrade` 时跳过 pip install |
| `lerobot-info` 作为 console script，不能用 `python -m lerobot-info` 调用 | 安装器先通过 `which` 找二进制，找不到再回退 `python -m` |
| `practice export --episode <dir>` 被错误路由到 parquet 导出器 | CLI 现在优先判断 `--episode` 是否指向已存在目录，是则直接骨架导出 |
| 无 LeRobot 环境的测试在已安装环境失败 | `test_doctor_report_when_not_installed` 在检测到 LeRobot 时自动 skip |
| 在 Python 3.11 下执行 setup 会导致长时间无效安装 | 安装器在 pip install 前校验 Python ≥ 3.12，否则直接返回错误提示 |

## 5. 本地联调验证结果

### 5.1 环境

- **LeRobot 环境：** `.venv-lerobot`（Python 3.12.13，LeRobot 0.6.x，PyTorch 2.11.0+cu128，CUDA 12.8）
- **无 LeRobot 环境：** `.venv`（Python 3.11.15，仅 rosclaw-core 依赖）

### 5.2 命令实测

```bash
# setup（已安装 LeRobot，秒级完成）
HF_ENDPOINT=https://hf-mirror.com \
  PATH=.venv-lerobot/bin:$PATH \
  PYTHONPATH=src \
  .venv-lerobot/bin/python -m rosclaw.cli setup lerobot --profile core
# 输出：OK: True, LeRobot version: 0.6.x
```

```bash
# doctor
rosclaw lerobot doctor
# Status: installed, LeRobot importable: True, Torch available: True, CUDA available: True
```

```bash
# info
rosclaw lerobot info
# 正常输出 LeRobot 版本、PyTorch 版本、GPU 等信息
```

```bash
# provider infer dry-run
rosclaw provider infer --type lerobot_policy \
  --manifest examples/lerobot/sample_policy_manifest.yaml \
  --input examples/lerobot/sample_observation.json \
  --dry-run
# status: ok, action: [0.0, ...], safety.executable: false
```

```bash
# provider infer 真实 smoke
rosclaw provider infer --type lerobot_policy \
  --manifest examples/lerobot/sample_policy_manifest.yaml \
  --input examples/lerobot/sample_observation.json
# lerobot_smoke.import_ok: true, version: 0.6.x
```

```bash
# practice export skeleton
rosclaw practice export --format lerobot \
  --episode examples/practice/minimal_episode \
  --output /tmp/rosclaw_lerobot_export
# 生成 README.md, meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl,
# meta/rosclaw_mapping.json, data/placeholder.jsonl, videos/README.md
```

### 5.3 测试结果

**无 LeRobot 环境：**

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/integrations -q -p asyncio
16 passed in 5.38s
```

**有 LeRobot 环境：**

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PATH=.venv-lerobot/bin:$PATH \
  .venv-lerobot/bin/python -m pytest tests/integrations -q -p asyncio
15 passed, 1 skipped in 17.06s
```

`1 skipped` 是预期的：未安装场景下的 doctor 断言在 LeRobot 已安装时自动跳过。

## 6. 已知限制与后续计划

- **P0 不实现真实策略推理**：`provider infer` 仅做 import smoke 并返回 sample action，真实加载 LeRobot policy 模型留到下一轮。
- **P0 不实现真实数据集写入**：`practice export --format lerobot` 输出骨架（metadata + mapping），真实 frame/parquet/video 写入留到下一轮。
- **train / eval / rollout / reward 后端**：已在 `capability list` 中列出，标记为 disabled / future。
- **PR 待提交**：本地分支已就绪，需用户完成 GitHub 授权后才能 push 并创建 PR。

## 7. 关键提交记录

```text
dff5375 docs(lerobot): note Python 3.12+ requirement and setup fast-path
77fc983 lerobot bridge: enforce Python 3.12+ before pip install
d563c69 lerobot bridge: fast-path setup, absolute episode paths, skip no-lerobot test when installed
aa71fd4 feat(integrations): add ROSClaw × LeRobot bridge (P0/P1)
```

## 8. 如何继续提交 PR

请在对话中执行以下任意一种授权方式，我即可继续 push 并创建 PR：

1. `! gh auth login` 完成 GitHub CLI 登录；
2. `export GITHUB_TOKEN=你的token` 并告诉我；
3. 手动把 remote 改为带 token 的 URL：`git remote set-url origin https://<TOKEN>@github.com/ros-claw/rosclaw.git`。

---

**报告人：** Claude Code  
**生成位置：** `ROSCLAW_LEROBOT_ROUND1_REPORT.md`
