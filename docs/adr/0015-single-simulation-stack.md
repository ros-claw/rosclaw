# ADR-0015：单一仿真栈——SimulationRuntime 为唯一仿真内核

- 状态：Accepted（2026-09-16，G-6/三审 B-4）
- 基线：main=`cf9aeb3d`（MH15 合入后）
- 依据：0916 三审稿「正在形成新的'双仿真栈'风险」；ADR-0014（仿真
  不是 Harness Backend——本 ADR 与其正交：它冻结语义边界，本 ADR
  冻结**实现栈数量**）

## 背景

MuJoCo Harness 轨道（MH0–MH15）落地了 `src/rosclaw/sim/`
（SimulationRuntime + MujocoBackend：load_model/observe/rollout/
render/batch/audit，状态真权归 MuJoCo），而 2025 年以来的
`src/rosclaw/agentd/sim_render.py`（1304 行：RenderSpec/overlay/
receipt/相机取景/幂等渲染）仍在生产面服务 `simulation_render_scene`
等模型工具。两个栈各自带渲染/探测/状态回放实现，已开始分叉
（各自修各自的 bug——0914 mjv 米单位只修了旧栈；MH 的后端探测
只修了新栈）。

## 决策

1. **SimulationRuntime 是唯一仿真内核。** 一切新仿真能力
   （动力学/渲染/观测/审计）只能进 `src/rosclaw/sim/`——禁止在
   `agentd/` 或别处新增第三个实现点。
2. **`agentd/sim_render.py` 降级为 deprecated 适配层**：
   - 只允许维护性修改（bug 修复、契约保持）；
   - 公开函数集合冻结（机器门禁：`tests/agentd/test_g6_single_sim_stack.py`
     比 baseline 多一个公开函数即红）；
   - 模块 docstring 标注 deprecated 与迁移指向；
   - 其承载的产品契约（RenderSpec/overlays_applied/unfulfilled/
     per-render receipt/相机取景/渲染幂等 key）在迁移完成前继续
     生效——这些契约的归宿是新栈 render 能力（MH6 已起步），
     迁移必须连同 U05/U06/G01/G02 验收 oracle 一起过，不允许
     先删后补。
3. **收敛路线**（后续轮次，非本 PR）：
   - R1：新栈 render 吸收 overlay/receipt 契约（RenderSpec →
     MujocoBackend.render 的 spec 面）；
   - R2：`simulation_render_scene` 工具 dispatch 改走
     SimulationRuntime（旧路径影子模式跑一个周期比对 receipt）；
   - R3：旧适配层退役（物理删除 sim_render.py——公开面已全迁走）。

## 后果

- 双栈分叉被冻结在现状（不再恶化）；收敛有明确验收口径
  （oracle 绑定收据不变绿不过）。
- 短期内 agentd/sim_render.py 继续存在是**有意的过渡形态**，
  不是双栈辩护——冻结清单 + 防分叉门禁保证它只缩不胀。
