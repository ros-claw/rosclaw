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

## 补充：MH1 不可变 Store 决策（2026-09-14，PR-MH1）

1. **布局共存**：新对象落 `task_root/sim/{models,states,traces,audits,
   renders,experiments}/`；legacy `sim/api.py` 布局
   （`task_root/models/`）不动，由 `SimStore` 只读桥解析
   `model_`/`obs_`/`op_` ref——legacy 分区的写入一律拒绝
   （`STORE_LEGACY_READONLY`）。api.py 写路径的迁移按能力逐个搬，
   不在本 PR。
2. **ref 是内容寻址令牌，不是路径**：
   `sim<kind>_<sha256[:16]>`，词法层拒绝 `../`、绝对路径与 URI
   （`REF_INVALID`），路径逃逸在构造上不可能；`SimStore` 另做
   resolve 前缀 + symlink 双重校验兜底。
3. **不可变语义 = 幂等写 + 绝不覆写**：同 ref 同内容幂等返回
   （Agent 重试/网络重放场景的唯一安全语义）；同 ref 异内容
   （构造上不可达，防御盘外篡改）→ `STORE_IMMUTABLE_VIOLATION`。
   `get` 读回重算 digest，防盘外篡改。

## 补充：MH2 模型服务决策（2026-09-14，PR-MH2）

1. **存储 = manifest + 资产双对象**：`models/<simmdl_*>.json` 存
   model manifest（mjcf_xml + assets ref 表 + source +
   parent_model_ref + patches + compile_warnings），不含 created_at
   → 内容寻址天然幂等（同 patch 重放同 ref）；mesh/texture 资产以
   原始字节内容寻址落盘（母子模型自动去重，ur5e 约 31MB 不重复
   存储）。created_at 由 manifest 落盘 mtime 派生。
2. **编译警告字段常驻但允许为空**：MuJoCo 3.11 Python 无可靠编译
   警告捕获通道（`set_mju_user_warning` 实测对 from_string/compile
   路径不触发）。`compile_warnings` best-effort 捕获、默认 `[]`，
   测试只断言类型；未来版本通道可用时自动开始记录。
3. **geom mass/density 取自来源 MjSpec**：编译后 MjModel 将 geom
   质量并入 body 惯性，不再保留逐 geom 值；`inspect_model_full`
   在传入 spec 时按名字补齐，未显式声明的记 `None`（诚实标注，
   不拿编译推导值冒充声明值）。
4. **P0 patch 只开放 `set` 白名单**；`add`/`remove`/`attach` 显式
   `MODEL_FIELD_UNSUPPORTED`（拓扑/命名空间语义后续里程碑开放）。
   actuator kp/kv 映射 `gainprm[0]`/`biasprm[1,2]`，仅限
   gaintype=fixed + biastype none/affine 的 position 类执行器，
   其余拒绝而非静默误改。
5. **api.py 历史缺口记录**：`sim/api.py` 仍接受 task_root 外绝对
   路径；新 `sim/resolve.py` 已堵死（外部 URI → MODEL_NOT_FOUND、
   越界 → MODEL_PATH_ESCAPE），api.py 迁移 PR 时收编。

## 补充：MH3 状态与实验决策（2026-09-14，PR-MH3）

1. **快照即完整可续仿真状态**：time/qpos/qvel/act/ctrl/mocap_pos/
   mocap_quat，绑定 model_digest；跨模型 `CROSS_MODEL_REF`、维度
   不符 `STATE_DIMENSION`/`CTRL_DIMENSION`、非有限 `STATE_INVALID`，
   绝不 silent truncate 或 pad（沿用 api.py 强约束语义）。
2. **fork 是记录不是复制**：内容寻址下 N 个 branch 初始 ref 必然
   相同（digest 一致由构造保证）；分支身份为 fork 记录中的
   branch_id（b0..bN-1）；`max_branch_count=64` 预算 fail closed。
3. **rollout 强制预算**：max_steps=200k / max_duration=600s /
   max_wall_time=60s / max_record_points=480 / max_trace_bytes=32MB，
   超限 `SIM_BUDGET_EXCEEDED`；逐步 NaN/Inf 哨兵 → `SIM_DIVERGED`；
   controller 白名单 hold / ctrl_series / position_targets，Python
   controller 不开放（ADR-0014 #3：无 Agent 直连通道）。
4. **observe 语义化有界**：contact 最多 50 对；能量读取用
   `mj_energyPos/Vel` 写回 `data.energy` 的 3.11 实际签名；图像类
   通道留给渲染 PR（返回 artifact ref 而非 RGB 数组）。

## 补充：MH4 物理诚实审计决策（2026-09-14，PR-MH4）

1. **A01-A08 吸收 Text2Mujoco（MIT）思想、按 ROSClaw 后端重写**：
   碰撞覆盖/显式质量/伺服保持/连杆连续/初始穿透/序列穿透/隐藏
   自重叠/marker 接地；阈值继承其经验值（0.1mm/1mm/2mm·1°/5mm/
   3mm·1mm）并全部收拢为 `AuditPolicy` named policy——不为测试
   通过调松阈值，先修模型。
2. **红绿 fixture 纪律**：`tests/sim/fixtures/broken_models/` 每个
   audit 至少 1 red（必须被抓）+ 1 green（必须通过）；fixture
   设计本身经过物理实证修正——A07 需要 contype/conaffinity 真正
   互斥才算"隐藏"；A03/A16 需要质量水平偏置打破不稳定平衡；
   kp=5000 伺服必须配 damping=50 才稳定。
3. **A15（NaN/Inf）语义实证**：float64 CPU MuJoCo 下动力学极限
   环几乎不产生真 NaN（3.11 实测 kp=2e9 仍有限）——A15 核心
   检测面是 **trace/状态数据完整性**（非有限值 fail），live 扫描
   发散为副面；红测试用伪造 NaN trace 验证检测逻辑。
4. **审计器故障 ≠ 模型通过**：单项 check 抛异常记 ERROR，总状态
   FAIL（fail closed）；结果（含逐项 detail/violations/warnings/
   evidence）机器可读落 audits 分区，`audit_ref` 可寻址。
5. **A09-A14/A21-A24 暂缓**（actuator saturation/force·velocity·
   acceleration 限值/接触冲量/sensor 有效性/坐标系/solver·timestep
   敏感性）：注册表结构已预留，后续里程碑开放。

## 补充：MH5 回执与重放决策（2026-09-14，PR-MH5）

1. **SimulationReceipt 是实验的唯一证据形态**（规格 §31）：
   model/initial_state/action/trace/audit 全 ref 化 + 双层 digest
   （raw `states_digest` 逐状态 + `semantic_digest` 指标容差层，
   规格 §56）；payload 不含 created_at → 内容寻址幂等。
   `trust_level=SIMULATED`、`usable_for_real_execution=false` 恒成立。
2. **strict replay 判定层级**（§57）：backend/backend_version/
   model_digest 不符即 `REPLAY_DIVERGED`；raw digest 一致 →
   verified(raw)；raw 不符但语义指标 + success 容差内一致 →
   verified(semantic)；双不符 → `REPLAY_DIVERGED`，不得 promotion。
3. **指标在 rollout 中逐步采集**（不经采样 trace——采样漏峰值）；
   采集目标必须在采集前应用（position_targets 预应用 +
   ctrl_series 逐行查表，实证修复过目标错位 bug）。
4. **sim_compare 机器比较**：指标表 + Pareto 非支配集（
   tracking_rmse/energy_end/peak_qvel 三轴，success=False 不参与
   支配）+ best（Pareto 内 rmse 最小）；不让 LLM 肉眼比 JSON。
5. **ExecutionReceipt 集成点**：SimulationReceipt 字段与
   `kernel/contracts.py` 的 simulation_result 槽位对齐，嵌入
   接线属 MH6/MH8（本 PR 不改 kernel 冻结语义）。

## 补充：MH6 Agent 工具面决策（2026-09-14，PR-MH6）

1. **P0_SIM_TOOLS 独立成组**（规格 §29）：10 个 v1 工具不塞进
   P0_CORE_TOOLS；S0 = get_capabilities/inspect/observe/compare，
   S1 = load/patch/snapshot/rollout/audit/render；全部 ≤ S1，
   `usable_for_real_execution=false`，零 REAL permit。
2. **SimulationRuntime 是 Agent 面编排门面**（`sim/runtime.py`）：
   RuntimeClient.sim_* 只委派、不自己实现 MuJoCo（规格 §28）；
   默认任务根 `ROSCLAW_SIM_TASK_ROOT > $ROSCLAW_HOME/sim_tasks/agent`。
3. **sim_rollout 直接产 SimulationReceipt**（rollout+指标+审计一体），
   sim_compare 消费 receipt_refs——Agent 的实验链是
   inspect→patch→snapshot→rollout(receipt)→observe→audit→compare
   几次结构化调用，不需要写 Python glue。
4. **render 是证据 artifact 不是真相**：GIF 落 renders 分区，
   实际渲染后端（EGL/OSMesa/glfw）诚实记录；GL 不可用时
   `SIM_RENDER_UNAVAILABLE` 显式失败，不静默降级。
5. **sandbox_run 保留**（规格 §27 legacy facade）；P0_TOOLS 注册
   顺序与 P0_AGENT_MCP_TOOLS 目录一致（test_server 锁定）。

## 补充：MH7 WorldSpec 决策（2026-09-14，PR-MH7）

1. **WorldSpec 是语义，MJCF 是物理实现**：Body 由 e-URDF 管，
   WorldSpec 管世界与任务（ground/objects/markers/cameras/task
   谓词）；不复制 robot schema。`rosclaw.sim.worldspec.v1`。
2. **typed Interaction Contract**（吸收 Text2Mujoco）：typed
   target + affordance 词表（press/pull/grasp/place/push/inspect/
   move）+ action_schema JSON-Schema 子集 + depends_on 只许前向
   引用（DAG 由构造保证）；success/failure 只开放 inside/near
   两种机器谓词——未知形式 fail closed。
3. **无假 Affordance 硬约束**（§24/H05）：grasp 要求被挂 body
   有真实夹爪执行器（编译后推导，不从机器人名字猜）；没有机器人
   的世界也不能声明 grasp——`CAPABILITY_UNAVAILABLE`。
4. **MjSpec.attach 实证**（3.11）：`spec.attach(robot_spec,
   frame=frame, prefix="<id>_")` 挂机器人；attach 后 mesh 文件
   引用**带 prefix**——剥 prefix 按 basename 复用内容寻址 blob
   （不重复存储 31MB mesh）。
5. **marker 贴面**：交互 marker（site group=2）贴目标物体顶面，
   放原点会被 A08 判 buried（审计自证）。
6. **sim_build_world / sim_interact 工具面暂缓**（规格 §25 后续）；
   本 PR 交付编译器与契约层，T1 世界端到端（compile→audit PASS→
   谓词求值）由 tests/sim/test_worldspec.py 锁定。

## 补充：MH8 验收与 Release Gate 决策（2026-09-14，PR-MH8）

1. **H01-H08 分两层**：harness 能力层（tests/sim/test_agent_scenarios.py，
   Verifier 直接对 MuJoCo 真相，确定性）随本栈合入；真实 LLM 变体
   按 W09 纪律（PTY 串行、repo .venv PATH）列为 pending-live，不合成
   冒充。
2. **显式状态移植**：`transplant_state`（维度校验 + provenance）是
   参数实验的标准动作；fork 状态**不能**静默跨模型
   （CROSS_MODEL_REF 语义不变）。
3. **A/B 指标 B 侧机器生成**（test_ab_harness.py）：tool_calls≤12、
   bash/Python/XML glue=0、evidence 完整、**false_success=0**（每个
   claim success 的 receipt 必须 strict replay 复核一致）。
4. **Release Gate**：docs/validation/MUJOCO_HARNESS_V1.md G1-G12
   绑定 test+report+commit；G10 A 侧与 G11 为 PARTIAL/pending-live，
   留痕不阻塞。
5. **残留风险记录**：mujoco 3.11 `to_xml` 对 `mass=1.0` 特殊省略
   （patch geom.mass=1.0 会丢 round-trip——后续 patch 层加
   round-trip 校验或值域提示）；A09-A14/A21-A24 audit 扩展、
   sim_build_world/sim_interact 工具面、Menagerie source、
   MJX/MJWarp 加速面均按总纲留待后续里程碑。

## 补充：MH9 证据语义硬化（2026-09-15，0915 优化文档）

合入 main 前的语义修正（不作新功能扩展）：

1. **成功语义三分**：`SimulationReceipt` 拆分为
   `simulation_valid`（rollout 跑完）/ `physical_audit_pass`（审计
   通过）/ `task_success`（任务谓词机器判定，None=未评估）+
   `verification_status`（PASS/FAIL/NOT_EVALUATED）；兼容字段
   `success` ≡ `task_success`——audit PASS 不再冒充任务成功。
   `false_success` 重定义为：claim PASS 但 strict replay（含任务
   判定复算）不一致。
2. **model_digest 纳入资产**：`sha256(canonical{xml_sha, sorted
   资产 blob sha})`——相同 XML 不同 mesh 字节 = 不同物理模型，
   state/trace/replay 绑定全部 fail closed（红测试锁定）。
3. **gripper 能力 = 声明→证明绑定**：e-URDF capabilities.yaml
   声明（required_hardware/constraints.requires_gripper）+
   semantic.yaml affordance link 子树 actuated joint 证明；task
   模型经 `<model>.capabilities.yaml` sidecar 声明并由模型证明。
   三态 AVAILABLE/UNDECLARED/UNPROVEN，后两者拒绝 grasp——
   `"gripper" in name` 的名字猜测被移除。
4. **transplant 结构签名**：joint 名/类型/qpos 地址/dof 地址/
   actuator→joint 映射/mocap 布局全等才允许移植——维度相同但
   语义不同的模型拒绝 STATE_INCOMPATIBLE。
5. **replay 错误分类**：REPLAY_ENV_MISMATCH（backend/版本）/
   REPLAY_MODEL_MISMATCH / REPLAY_STATE_MISMATCH /
   REPLAY_PHYSICS_DIVERGED——版本升级不再误报"物理发散"。
6. **高层原语**：`sim_branch_experiment`（fork+移植+rollout 一次
   调用，Agent 不碰 transplant 底层）与 `sim_compile_world`
   （WorldSpec→validation→能力绑定→compile）加入 P0_SIM_TOOLS
   （S1，全组 ≤S1 不变）。
