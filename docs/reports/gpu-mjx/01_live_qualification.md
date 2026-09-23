# GPU/MJX Live Qualification 报告（2026-09-23）

## 结论

**G41 GPU Live Qualification：✅ PASS**（此前 NOT_RUN——2026-09-23 起
jax 0.10.2 cuda12 plugin 提供官方 aarch64 wheel，GB10 实测可用）。

**G42 GPU/CPU Semantic Agreement：✅ PASS（live 证据）**——真实 MJX GPU
轨迹与 CPU authoritative 全字段一致（SEMANTIC_AGREEMENT），候选 promoted。

机器可读证据：`live_qualification_2026-09-23.json`（本目录）。

## 环境指纹

| 项 | 值 |
|---|---|
| 硬件 | NVIDIA GB10（aarch64，DGX Spark 级） |
| 驱动 | 580.159.03，CUDA driver 13.0 |
| 栈 | jax 0.10.2 + jax-cuda12-plugin 0.10.2 + mujoco 3.13.0 + mujoco-mjx 3.13.0 |
| 精度 | jax_enable_x64=True（与 CPU float64 权威面对齐） |

## 用例与结果

- 模型：tiny_arm（2 hinge + 2 单输入 position 伺服，无接触）
- 控制：`position_targets [0.4, 0.2]`，100 步（dt=0.002）
- 谓词：shoulder ∈ [0.3, 0.5]
- MJX final qpos = [0.44191151, 0.21470198]；task_success=True
- 逐字段比对（qpos/qvel 1e-6、checkpoint 1e-5）：**零分歧字段**，
  SEMANTIC_AGREEMENT → promoted。

## 诚实边界（不粉饰）

1. **CPU strict replay 永远是权威验证面**（ADR-0014 #8）——GPU 只做
   探索/候选产出；本证据证明的是"GB10 上 MJX 与 CPU 对该用例语义
   一致"，不是 GPU 结果可独立采信。
2. **MJWarp 未装**（warp 模块缺失）——MJWarp 路径仍 NOT_RUN。
3. **peak_contact_force 通道未接**（MJX 接触力提取未实现）——候选
   记 None，门跳过该字段；接触密集场景的力一致性尚无 live 证据。
4. 谓词通道映射限关节量（joint_positions/joint_velocities）——
   接触/位姿通道的 GPU 侧求值待一般化。
5. CI 无 GPU——live 测试在 CI 走 skip 路径（诚实留档），本机
   （GB10）走真实路径；`gpu_execution_status()` 探测为唯一事实源，
   旧"本机永远 NOT_RUN"假设已被环境证伪并修正为探测一致性断言。
