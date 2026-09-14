# ADR-0014：Simulation Harness 架构冻结——MuJoCo 不是 Harness Backend

- 状态：Accepted（2026-09-14，PR-MH0）
- 基线：main=`6776715c`
- 依据：《ROSClaw MuJoCo Harness 原生物理仿真能力升级实施总纲》§1/§4/§5/§10/§64
- Amends：无；与 ADR-0012/0013 正交（不推翻、不细化）

## 背景

ROSClaw 已经具备相当多 MuJoCo 底层能力：`sim/api.py` 的最小编程接口
（load_model/observe/submit_simulation/read_trace/render/export）、
`model_inspect.py` 的编译态身体推导、`sandbox/backends/mujoco_cpu.py`
的 strict replay 与物理审计原语、SimForge 的搜索/演化/晋升门。

但这些是**散落的仿真基础设施**，边界存在两类漂移风险：

1. 仿真被误当 Harness Backend——与 ADR-0012/0013 冻结的
   `NativeHarnessBackend`（Pi，future Codex）词表混淆；
2. 仿真结果被误当物理执行证据——冲击 Gate H 三层证据语言
   （COMMAND_REPLAY / SIMULATED / PHYSICALLY_OBSERVED）与 REAL 门禁。

本 ADR 在写任何 Agent 面功能之前先冻结边界。

## 决策

1. **MuJoCo 是物理仿真引擎，不是 NativeHarnessBackend。** 它永不进入
   Harness Backend 词表；CLI `--engine/--backend` choices 只保留
   ADR-0013 #1 的仿真/运行时语义（mujoco/isaac/mock/fixture/ros2），
   不得混入 pi/codex 等 Harness 名称。不存在也不许新增
   `--engine mujoco` 之外的"harness 语义" engine 面。
2. **SimulationRuntime 属于 ROSClaw Native Runtime**（进程内能力，
   非独立进程、非 daemon 子系统）。代码命名使用
   `rosclaw.sim` / `SimulationRuntime` / `SimulationBackend` /
   `MujocoBackend` / `SimulationReceipt`；产品层可称 "MuJoCo Harness /
   Physical AI Harness"，但不得污染 `NativeHarnessBackend` SPI。
3. **Agent 只能经 ToolGateway 触达仿真。** 不存在 Agent 直连
   MuJoCo 的通道；仿真工具不暴露 raw `mj_step`/geom array/MjData
   可变操作作为主要接口。
4. **仿真永远拿不到 REAL permit。** 仿真证据 ≠ 物理执行证据；
   证据三层语言与 `usable_for_real_execution` 语义不变
   （`kernel/contracts.py` ExecutionReceipt 语义不动）；simulation
   evidence 永远不能 promote 为 REAL evidence。
5. **Model / State / Trace 不可变。** 修改模型 = 新 ModelPatch →
   编译 → 新 ModelReference（digest 派生，记录
   parent_model_ref/patch/backend_version/created_at）；不原地改。
   已生成的 ref 其内容不可变更。
6. **Verifier 拥有任务终态**（ADR-0012 #5 延伸至仿真域）：
   仿真"跑完"不等于任务成功；Agent 自由文本不能宣布成功，终态由
   success predicate / audit / independent replay / Verifier 决定。
7. **CPU MuJoCo strict replay 是 MuJoCo 权威验证面**（沿用
   `sandbox/backends/mujoco_cpu.py` 的 ReplayReport 语义）。
8. **MJX / MJWarp 未来只能作为探索/训练加速面**（"GPU 找候选，
   CPU MuJoCo 判结果"），不进权威验证面，本阶段不引入依赖。
9. **sim.api / sandbox / SimForge 不是三套系统**，而是共享原语的
   三个消费者：sandbox = 安全验证消费者；SimForge =
   大规模搜索/演化消费者；Native Agent = 交互实验消费者。
   后续 MH 系列把它们收束到共享的 SimulationRuntime 原语上，
   渐进迁移，不另建平行框架。

## 为什么

换引擎焦虑与品牌泄漏之外，仿真域最大的风险是**证据等级漂移**与
**模型原地可改**——两者都会让 Verifier 无法重放、让"假成功"穿过
防线。先冻结边界，再长能力。

## 后果

- 新增：`src/rosclaw/sim/contracts.py`（仿真域 v1 契约骨架，
  maturity=experimental；首个跨进程消费者落地时才晋升
  `rosclaw.contracts` 包）、`src/rosclaw/sim/capabilities.py`
  （MujocoRuntimeCapabilities 真探测；必需能力缺失 →
  `SIM_CAPABILITY_UNAVAILABLE` fail-fast，绝不静默降级）。
- 新增：`tests/sim/`（红→绿）与
  `tests/architecture/test_adr0014_simulation_harness.py`
  （静态不变量，随 CI）。
- MuJoCo 版本策略：legacy minimum 保持 `mujoco>=3.0.0`；
  Qualified Harness Runtime 目标为 MuJoCo 3.13.x；能力以运行时
  探测为准，不做版本字符串比较。
- 不变：ADR-0012/0013 全部条款；无 `--engine pi/codex`；无新
  Harness backend 名称；`sim/api.py`、`model_inspect.py`、
  sandbox、SimForge 现有行为本阶段不改写。
