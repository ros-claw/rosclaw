# ROSClaw × LeRobot Bridge — P1.1 Real Policy Smoke Gate 实施报告

**报告日期：** 2026-07-10  
**目标：** 为 P1 Provider Worker 做一次真实验收：真实 LeRobot policy 权重加载 → 真实推理 → ROSClaw action_proposal，并始终标记 `not_executed=true`、`requires_sandbox=true`、`executable=false`。  
**验收命令：** `rosclaw lerobot smoke-policy`

---

## 1. 交付内容

### 1.1 新增模块

| 文件 | 职责 |
|------|------|
| `src/rosclaw/integrations/lerobot/smoke_policy.py` | P1.1 多阶段 smoke 编排器：runtime/materialize/inspect/load-test/infer/report |
| `src/rosclaw/integrations/lerobot/policy_cache.py` | Policy 物化：本地目录校验、HF cache 查找、离线/联网下载 |
| `src/rosclaw/integrations/lerobot/smoke_report.py` | Smoke report 读写与 `latest.json` 管理，供 doctor 读取 validation 状态 |
| `src/rosclaw/integrations/lerobot/action_adapter.py` | 升级：保留 action chunk 原始 shape（如 `[100, 14]`），不再强制压平 |
| `src/rosclaw/integrations/lerobot/worker_main.py` | 升级：更健壮的 LeRobot factory 加载、safetensors checkpoint 加载、chunk 输出保留 |
| `examples/lerobot/sample_observation_aloha_act.json` | ACT ALOHA smoke 用 observation 示例 |
| `examples/lerobot/aloha_top_480x640.png` | 640×480 RGB smoke 图像 |
| `tests/integrations/test_lerobot_smoke_policy.py` | smoke-policy / report / action chunk adapter 测试 |
| `tests/integrations/test_lerobot_real_policy_smoke.py` | 可选真实 policy smoke（依赖环境变量） |

### 1.2 修改模块

| 文件 | 主要变更 |
|------|----------|
| `src/rosclaw/integrations/lerobot/cli.py` | 新增 `cmd_smoke_policy_lerobot`；doctor 输出增加 validation 区块 |
| `src/rosclaw/cli.py` | 新增 `rosclaw lerobot smoke-policy` 子命令与参数解析 |
| `src/rosclaw/integrations/lerobot/__init__.py` | 导出新增符号 |
| `docs/integrations/lerobot_bridge.md` | 新增 P1.1 章节：推荐 policy、allow-network/本地路径用法、validation 含义、非 rollout 说明 |

---

## 2. 推荐 smoke policy

```text
lerobot/act_aloha_sim_transfer_cube_human
```

理由：

- LeRobot 官方 ACT policy；
- 仓库约 213 MB；
- 标准 artifact：`config.json`、`model.safetensors`、`train_config.json`；
- 输入输出形状简单：
  - `observation.images.top`: `[3, 480, 640]`
  - `observation.state`: `[14]`
  - `action`: `[14]`（ACT 可能输出 chunk `[100, 14]`）

备用 policy：`lerobot/act_aloha_sim_insertion_human`。

---

## 3. CLI 用法

### 联网下载并 smoke

```bash
rosclaw lerobot smoke-policy \
  --policy.path lerobot/act_aloha_sim_transfer_cube_human \
  --device cpu \
  --allow-network
```

### 本地已下载 policy

```bash
rosclaw lerobot smoke-policy \
  --policy.path /data/rosclaw/policies/act_aloha_sim_transfer_cube_human \
  --device cpu
```

### 查看 doctor validation 状态

```bash
rosclaw lerobot doctor
```

---

## 4. Smoke 执行流程

```text
Stage 0: runtime check
Stage 1: policy materialization
Stage 2: inspect
Stage 3: load-test
Stage 4: infer
Stage 5: smoke report + doctor state更新
```

- 默认 `allow_network=false`、`device=cpu`；
- 对未缓存 HF repo 且未传 `--allow-network` 返回 `network_disabled`；
- 真实 policy load/infer 失败时返回结构化错误，不伪造 success；
- 输出 action_proposal 始终 `not_executed=true`、`requires_sandbox=true`、`executable=false`；
- 拒绝 `--execute`（已在 P1 provider 层实现）。

---

## 5. 验证结果

### 5.1 集成测试

```bash
env -u PYTHONPATH .venv/bin/python -m pytest tests/integrations -q --tb=short
```

结果：

```text
59 passed, 1 skipped in 11.46s
```

> 新增的 smoke policy/report/action chunk 测试全部通过；原有 P1 worker/provider/adapter/doctor 测试保持通过。

### 5.2 真实 policy 端到端 smoke

在已配置的 LeRobot Python 3.12 runtime 上执行：

```bash
export HF_ENDPOINT=https://hf-mirror.com
rosclaw lerobot smoke-policy \
  --policy.path lerobot/act_aloha_sim_transfer_cube_human \
  --device cpu \
  --allow-network
```

结果：

```text
Status: ok
Stages:
  runtime_check: ok
  materialize:   ok
  inspect:       ok
  load_test:     ok
  infer:         ok
Policy: lerobot/act_aloha_sim_transfer_cube_human (act)
Input features:
  observation.images.top: [3, 480, 640]
  observation.state:      [14]
Output features:
  action: [14]
Action proposal:
  type:  raw_lerobot_action
  shape: [14]
  executable: false
  requires_sandbox: true
  not_executed: true
Timing:
  inspect: 6.784 s
  load:    17.962 s
  infer:   17.182 s
```

`rosclaw lerobot doctor` 随后报告：

```text
Real Policy Smoke Validation
  Status:            validated
  Last policy:       lerobot/act_aloha_sim_transfer_cube_human
  Action shape:      [14]
  Time:              2026-07-09T19:23:16.156854Z
```

> 本次运行使用的是真实 LeRobot 0.6.x runtime、真实 ACT policy 权重与真实单步推理，action_proposal 已按 P1 安全契约标记为 `not_executed=true`、`requires_sandbox=true`、`executable=false`。

### 5.3 手工冒烟

| 命令 | 结果 |
|------|------|
| `rosclaw lerobot smoke-policy --help` | 子命令与参数已正确注册 |
| `rosclaw provider inspect --type lerobot_policy --policy.path tests/fixtures/lerobot_policy_minimal` | 仍返回正确 metadata |

---

## 6. 当前限制

| 限制 | 说明 |
|------|------|
| 真实 policy smoke | 已在本地 LeRobot Python 3.12+ runtime 通过；CI 仍使用 fake worker 覆盖 |
| 默认不联网 | HF repo id 必须显式 `--allow-network` 才会下载 |
| action chunk 解释 | P1.1 只负责把 chunk 原样交给 ROSClaw；后续 sandbox/body mapping 决定如何执行 |
| 本体映射 | ALOHA 14 维动作不会自动映射到 RH56/G1；留待 body/action mapping 阶段 |
| dtype 参数 | worker 已接收，当前主要依赖 policy 默认 dtype；可后续细化 |

---

## 7. 安全提示

- `rosclaw lerobot smoke-policy` 只是验证 policy 加载与单步推理；
- 它不会控制机器人；
- 它不会调用 MCP；
- 它不会绕过 sandbox；
- 所有 infer 结果都是 `action_proposal`，不可直接执行。

---

## 8. P2 建议

P1.1 通过后，建议进入 **P2：Practice → 真实 LeRobotDataset writer**，将 ROSClaw practice episode 真正写成 LeRobotDataset v3 parquet/mp4。

---

**报告生成位置：** `ROSCLAW_LEROBOT_P1_1_REAL_POLICY_SMOKE_REPORT.md`  
**生成人：** Claude Code
