# W03 真实状态记录与完整回放（规格 §7）

**分支**：a-w03-state-replay（基于 a-w02-contracts，§17 依赖：W03←W02/W00）
**日期**：2026-09-09

## 改动

### §7.1 记录侧：完整状态 + 显式维度声明

`src/rosclaw/sandbox/backends/mujoco_cpu.py`：

- `record_sample` 记录**完整** `qpos`（nq）/`qvel`（nv）/`ctrl`（nu）——
  替换旧的 `data.qpos[:nu]` / `data.qvel[:nu]` 截断。
- `trajectory_states.json` 升级 `rosclaw.trajectory_states.v2`：
  显式携带 `dims: {nq, nv, nu}` 与 `model_digest`——维度成为
  声明契约，不再由消费者猜。
- 观测指标改用完整状态数组；跟踪误差保持「实际 vs 同时刻指令」
  语义（受控子集 `qpos[:nu] - command`，§7.3 定义锚点）。
- receipt 的 `final_qpos` 保持 nu 维契约不变（与 trajectory
  目标同维度比较）。
- 回放校验接受 v1（旧记录）与 v2。

### §7.2 回放侧：同模型完整恢复，旧记录诚实标注

`src/rosclaw/agentd/sim_render.py` 新增 `restore_frame_state()`：

- 文件声明 dims 与模型不符 → `MODEL_DIMS_MISMATCH`（跨模型引用拒绝）；
- qpos 长度 == nq → 完整恢复（视频回放 = 状态恢复 + 相机 +
  `mj_forward` 仅前向，不重仿真）；
- qpos 长度 == nu < nq（旧截断记录）→ `LEGACY_PARTIAL_STATE`
  诚实拒绝——**绝不补零冒充完整状态**；
- 其他长度 → `STATE_DIMENSION`。

`_render_impl` 帧循环改走该函数；render receipt 新增
`state_contract`（`declared_dims_v2` / `legacy_undeclared_dims`）
标注兼容回放。`sim_trajectory.py` 的实际 eef FK 回放复用同一
恢复函数（记录侧 v2 后旧截断赋值在 nq≠nu 模型上必然错位）。

### §7.2 续仿真检查点

`src/rosclaw/sim/api.py` 新增 `_restore_initial_state()`：
`observe` / `submit_simulation` 的 initial_state_ref 恢复统一
校验——状态记录声明 `model_digest` 且不符 → `CROSS_MODEL_REF`；
qpos/qvel 维度不符 → `STATE_DIMENSION`（不截断、不补零）。

### W00 解标

`TestR2FullStateReplay::test_replay_restores_full_nq_not_nu`
xfail(strict) 解除——源码锚点 + W03 行为测试双保险。

## 验证（红→绿）

| 测试 | 结果 |
|---|---|
| test_w03_state_replay.py（10：dims/模型摘要声明、完整 qpos/qvel/ctrl、源码无截断守卫、恢复接受完整 nq、错误长度拒绝不补零、声明 dims 跨模型拒绝、LEGACY_PARTIAL_STATE、FK 回放锚点、续仿真维度/跨模型拒绝） | 红（首跑 1 failed 确认）→ 绿 10/10 |
| W00 R2 解标 + 全量 baseline | 绿（2 xfail 剩 R1/R3 = W05/W04 目标；R5 NOT_RUN skip） |
| tests/sandbox + tests/agentd 全量（除 PTY journey） | 783 passed, 1 flaky（test_discover_limo_fixture 单独重跑通过，与本轮无关——MCP fixture 发现时序） |
| ruff check src tests | 全过 |

## 边界说明

- 当前 `_model_layout_error` 仍拒绝 nq≠nv≠nu 的 rollout 模型
  （joint-position 后端的指令语义是 nu 维航点）——这是**诚实拒绝**
  而非数据丢失；本轮把记录/回放契约改成显式全维度，后续放开
  布局时不会再静默丢状态。
- 真实模型（K3）具身回放验收：**NOT_RUN**（无 ROSCLAW_KIMI_API_KEY
  ——不合成冒充；operator 带 key 重跑）。
