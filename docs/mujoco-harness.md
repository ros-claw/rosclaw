# MuJoCo Simulation Harness — Agent 使用指南（MH6，experimental）

> 面向 Native Agent 的原生物理实验面。原则：**理解物理世界 → 修改物理
> 世界 → 做物理实验 → 用物理证据判断结果。** 成功由 Verifier/审计决定，
> 不是 Agent 自由文本。

## 工具面（全部 ≤ S1，永不生成 REAL permit）

| 工具 | 级别 | 用途 |
|---|---|---|
| `sim_get_capabilities` | S0 | 探测本机 MuJoCo 能力面（真探测，不靠记忆猜） |
| `sim_load_model` | S1 | 加载 task-local / e-URDF zoo MJCF → `model_ref` |
| `sim_inspect_model` | S0 | 编译态检查：关节/执行器/传感器/相机/site/物理选项 + 中文摘要 |
| `sim_patch_model` | S1 | MjSpec 结构化补丁（set 白名单）→ 新 `model_ref`（母模型不变） |
| `sim_snapshot` | S1 | 完整可续仿真状态快照 → `state_ref` |
| `sim_observe` | S0 | 语义化有界观测通道（joint/body/site/sensor/contact/energy） |
| `sim_rollout` | S1 | 有界 rollout → SimulationReceipt（含指标+审计） |
| `sim_audit` | S1 | 物理诚实审计 A01-A08 + A15-A20 |
| `sim_compare` | S0 | 实验对比：指标表 + best + Pareto 候选 |
| `sim_render` | S1 | trace → GIF 证据 artifact（render 不是验证真相） |

## 标准实验链

```text
sim_load_model → sim_inspect_model          # 先理解身体，不要猜
  → sim_audit                               # baseline 物理诚实
  → sim_snapshot                            # 固定起点
  → sim_patch_model（参数分支 A/B/N）
  → sim_rollout（每个分支 → SimulationReceipt）
  → sim_compare（指标表 + best + Pareto）
  → sim_audit（候选复核）→ sim_render（证据 artifact）
```

## 硬性约束

- ref（`simmdl_/simsta_/simtrc_/simadt_/simrnd_/simexp_`）是内容寻址令牌，
  不可变；patch 必产新 ref，亲缘（parent/patches）全记录。
- 跨模型 state/trace → `CROSS_MODEL_REF`；维度/NaN/越界全部 fail closed。
- `trust_level=SIMULATED`、`usable_for_real_execution=false` 恒成立。
- patch P0 只开放 set 白名单（damping/range/friction/mass/density/rgba/
  kp/kv/ctrlrange/pos/quat/timestep/integrator）；add/remove/attach 显式拒绝。
- rollout 强制预算（steps/duration/wall_time/trace_bytes），超限
  `SIM_BUDGET_EXCEEDED`。
