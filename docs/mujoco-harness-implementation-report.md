# MuJoCo Simulation Harness 实施报告（MH0–MH8，2026-09-14；MH9 硬化，2026-09-15；已全量合入 main）

> 依据：《ROSClaw MuJoCo Harness 原生物理仿真能力升级实施总纲》+《实施优化0915》；
> 架构冻结：ADR-0014；发布门禁：docs/validation/MUJOCO_HARNESS_V1.md。
>
> **十个栈叠 PR 全部 CI 绿后 squash 合入 main（01ac050b）**：
> #547(MH0) → #548(MH1) → #549(MH2) → #552(MH3) → #553(MH4) →
> #554(MH5) → #555(MH6) → #557(MH7) → #558(MH8) → #567(MH9)。

## 0.2 合并与终验记录（2026-09-15）

- **合并瀑布**：每 PR 先 rebase --onto origin/main 再 force-push
  （先改 base=main 再推，否则 CI 不触发），13 项必检 + gate 全回归
  全绿后 admin squash。期间实证修复：#547 CI 暴露探测 GL 副作用、
  #555 Product Acceptance 暴露 init 模板 purposes 缺失、#558 gate
  暴露渲染进程内创建 native abort、#552 scene_b trace_id 偶发
  （本地 10 passed，重跑绿）。
- **终验（main 本地）**：tests/sim + architecture + w02 + mcp
  catalog + W09 = **309 passed**；tests/mcp（含 stdio/http e2e）=
  **216 passed**。
- **G13 全仓库终验（main，xdist）**：7602 passed / 16 failed /
  11 errors——失败全部落在既有 flaky/环境集（u_matrix acceptance、
  hf5_4 fixture、how/mysql、interaction_perf 墙钟、tmux/soak/pty、
  mcp health 陈旧机器状态等，0911 以来已记录同类零回归）；
  **tests/sim、tests/mcp、install/init/security、architecture
  零失败**。
- **卫生终检**：栈内新代码 ruff check / format / mypy 全净
  （ur5e_mcp.py 为既有 legacy，不在本栈范围）。

## 0.1 MH9 证据语义硬化（0915 优化文档，2026-09-15）

合入 main 前的语义修正（不加新功能）：

1. **成功语义三分**：`simulation_valid` / `physical_audit_pass` /
   `task_success` / `verification_status`；`success ≡ task_success`——
   audit PASS 不再冒充任务成功；`false_success` 含任务判定复算。
2. **model_digest 纳入资产**：相同 XML 不同 mesh = 不同物理模型，
   state/trace/replay 绑定 fail closed（test_model_identity 红测试）。
3. **gripper 能力 = 声明→证明绑定**：e-URDF capabilities/semantic
   或 task sidecar 声明，模型证明；三态 AVAILABLE/UNDECLARED/
   UNPROVEN，移除 `"gripper" in name` 名字猜测。
4. **transplant 结构签名**：joint 名/类型/地址/actuator 映射/mocap
   全等才允许（STATE_INCOMPATIBLE）。
5. **replay 错误分类**：ENV/MODEL/STATE/PHYSICS 四类，版本升级不
   误报物理发散。
6. **高层原语**：`sim_branch_experiment`（fork+移植+rollout 一次
   调用）与 `sim_compile_world` 入 P0_SIM_TOOLS（S1）。
7. **MuJoCo 3.13 资格认证（方案 A）**：pin `>=3.13.0,<3.14`；
   surfacevel 是 geom 属性（探测通道修正）；pid_actuator 实测存在。
8. **GL 实证根治**（#547 CI + 本机全量跑暴露）：能力探测改
   find_spec+ctypes 零副作用（GL 模块导入会翻转渲染后端选择）；
   渲染全程隔离子进程（Jetson 上 GL 上下文创建可 native abort）；
   记录实际使用的后端而非环境声明。
9. **G13 全仓库回归**：发现 install/init/test_command/security 31
   例本栈引入失败（init 模板 purposes 未注册 sim 工具）→ MH9c
   修复；其余失败基线对比确证为既有环境/时序腐烂。

## 0. 目标与结果一句话

把 MuJoCo 从"ROSClaw 能调用的一个 simulator"升级为 Native Agent
天生熟悉、可直接操作、可实验、可验证的物理工作环境：
**理解物理世界 → 修改物理世界 → 做物理实验 → 用物理证据判断结果。**

终验：**295 passed / 9 skipped**（GL×2 + W09 真实模型层×6 按纪律
NOT_RUN + 1 既有 skip）；tests/mcp 214 passed（含 stdio/http e2e）；
tests/architecture、tests/agentd/test_w02_contracts.py、
tests/eval/test_w09_cases.py 全绿；栈内新代码 ruff/format/mypy 全净。

## 1. 架构（ADR-0014 冻结）

```text
User → ROSClaw Native Agent（Pi，NativeHarnessBackend）
  → ToolGateway（P0_SIM_TOOLS，全部 ≤ S1）
  → RuntimeClient.sim_*（只委派）
  → SimulationRuntime（Native Runtime 进程内能力，sim/runtime.py）
  → MujocoBackend（sim/backends/mujoco/）
  → 不可变 SimStore（task_root/sim/{models,states,traces,audits,renders,experiments}）
```

九条冻结：①MuJoCo 不是 Harness Backend；②SimulationRuntime 属
Native Runtime；③Agent 只经 ToolGateway；④仿真永不获 REAL permit；
⑤Model/State/Trace 不可变；⑥Verifier 拥有终态；⑦CPU strict replay
为权威验证面；⑧MJX/MJWarp 仅探索加速面；⑨sim.api/sandbox/SimForge
是共享原语的三个消费者而非三套系统。

**存量零破坏**：api.py/model_inspect.py/sandbox/SimForge/kernel
contracts 全部未改；w02 精确相等断言全程作回归护栏。

## 2. 逐 PR 实施要点

### MH0（#547）架构冻结与契约/探针骨架
- `docs/adr/0014-simulation-harness.md`（冻结 9 条）+ README 索引。
- `sim/contracts.py`：16 个 v1 契约（继承 ContractModel：canonical
  JSON + sha256、未知字段前向兼容、未知主版本 fail closed），公共
  信封 backend/backend_version/created_at/digest，`with_digest()` 幂等。
- `sim/capabilities.py`：12 项能力**真探测**（属性/子模块/import，
  不做版本字符串比较）；必需能力缺失 `SIM_CAPABILITY_UNAVAILABLE`
  fail-fast，可选缺失只记录不静默降级。
- 测试锁定"非硬编码"（探测结果与解释器符号存在性交叉断言）。

### MH1（#548）不可变引用体系
- `sim/refs.py`：`sim<kind>_<sha256[:16]>` 内容寻址 ref——**ref 是
  令牌不是路径**，`../`/绝对路径/URI 词法层即拒（逃逸构造上不可能）。
- `sim/store.py`：六分区 SimStore；幂等写（同 ref 同内容）+
  `STORE_IMMUTABLE_VIOLATION`（异内容绝不覆写）+ tempfile 原子写 +
  symlink/超限/盘外篡改 fail closed；legacy `model_/obs_/op_` 只读桥。
- 决策：api.py 布局不动（w02 护栏），新对象落 `task_root/sim/`。

### MH2（#549）模型服务与 MjSpec patch
- load/inspect/compile/patch 四方法；**manifest+资产双对象**：
  manifest 不含 created_at → 内容寻址幂等（同 patch 重放同 ref）；
  31MB mesh 资产母子模型自动去重；created_at 由落盘 mtime 派生。
- §12.2 全量 inspect（body 树/joint 详情/actuator gear·forcerange·
  kp·kv/geom/传感器/相机/site/equality/tendon/contact exclude/
  keyframe/物理选项 + 中文 summary——summary 仅供理解，结构化权威）。
- patch P0 set 白名单 14 字段；add/remove/attach 显式拒绝；
  NaN/Inf/倒挂 range/越界 rgba/非法四元数 fail closed。

### MH3（#552）状态与实验原语
- 完整可续仿真状态快照（time/qpos/qvel/act/ctrl/mocap），跨模型
  CROSS_MODEL_REF、维度 STATE_DIMENSION/CTRL_DIMENSION、非有限
  STATE_INVALID——绝不 silent truncate/pad。
- fork 是记录不是复制（内容寻址 branch digest 天然一致）；
  max_branch_count=64。
- rollout 三型 controller（hold/ctrl_series/position_targets）+
  强制预算（SIM_BUDGET_EXCEEDED）+ 逐步 NaN 哨兵（SIM_DIVERGED）。
- observe 语义化有界通道（contact ≤50 对）。

### MH4（#553）物理诚实审计（核心里程碑底座）
- A01-A08（吸收 Text2Mujoco/MIT）+ A15-A20 ROSClaw 扩展；
  阈值全收 `AuditPolicy` named policy——不为测试通过调松阈值。
- 9 组红绿 fixtures（每 audit ≥1 red+1 green）。
- 审计器崩溃记 ERROR、总状态 FAIL——故障不伪装 PASS。

### MH5（#554）回执、重放与对比
- `SimulationReceipt`（rosclaw.sim.receipt.v1）：全 ref 化 + **双层
  digest**（raw states_digest + semantic_digest 指标容差层）；
  `trust_level=SIMULATED`/`usable_for_real_execution=false` 恒成立。
- strict_replay：backend/version/model digest 不符即 REPLAY_DIVERGED；
  raw 一致 verified(raw)；raw 不符语义层容差 verified(semantic)。
- 指标 rollout 中**逐步采集**（采样 trace 会漏峰值）。
- compare：指标表 + Pareto 非支配集 + best——不让 LLM 肉眼比 JSON。

### MH6（#555）Agent 工具面
- `P0_SIM_TOOLS` 十工具独立成组：S0=get_capabilities/inspect/observe/
  compare，S1=load/patch/snapshot/rollout/audit/render。
- `SimulationRuntime` 编排门面；RuntimeClient.sim_* 只委派；
  fixture 模式无物理；sim_rollout 直接产 SimulationReceipt。
- render：trace→GIF artifact；实际渲染后端诚实记录；GL 不可用
  `SIM_RENDER_UNAVAILABLE` 显式失败。

### MH7（#557）WorldSpec 与交互契约
- `rosclaw.sim.worldspec.v1`：world/body_refs/objects/sensors/
  interaction_points/task；typed target + action_schema 子集 +
  depends_on 只许前向引用 + inside/near 机器谓词。
- **无假 Affordance**：grasp 要求被挂 body 有真实夹爪（编译后推导），
  否则 CAPABILITY_UNAVAILABLE。
- MjSpec.attach 挂机器人；marker 贴物体顶面（放原点被自家 A08 抓）。

### MH8（#558）验收门禁
- H01-H08 harness 能力层场景测试（Verifier 直接对 MuJoCo 真相）。
- `transplant_state`：参数实验的显式状态移植（维度校验+provenance）。
- A/B B 侧度量管线：tool_calls≤12、glue=0、**false_success=0**。
- `docs/validation/MUJOCO_HARNESS_V1.md` G1-G12 门禁。

## 3. MuJoCo 3.11 实证发现（全部实测，非文档猜测）

| 发现 | 影响/对策 |
|---|---|
| `spec.find()` 不存在 | 用 s.joints/s.geoms 等集合遍历按名查找 |
| `joint.damping` 是 shape-(3,) ndarray | 必须 `el.damping[0]=v`，标量赋值 TypeError |
| actuator 无 kp/kv 属性 | kp→`gainprm[0]`+`biasprm[1]`、kv→`biasprm[2]`，仅 FIXED+NONE/AFFINE 伺服开放 |
| `MjModel` 无 `geom_mass` | 编译时并入 body 惯性；geom mass/density 从来源 MjSpec 按名补，未声明记 None |
| `spec.assets` 恒为空 dict | mesh 手动扫 spec.meshes 读文件建 assets dict 供 from_string 重编译 |
| prefix attach 后 mesh 引用带 prefix | 剥 prefix 按 basename 映射内容寻址 blob |
| `to_xml` 对 `mass=1.0` 特殊省略 | patch geom.mass=1.0 会丢 round-trip（残留风险，见 §6） |
| `mj_energyPos/Vel(m,d)→None` | 结果写回 `data.energy`（3.11 实际签名） |
| `set_mju_user_warning` 对 compile 不触发 | compile_warnings 字段常驻允许空 list |
| kp=5000 伺服 damping=1.0 实测发散 | dt·ω≪2 的理论判据不可靠，以实测为准（damping=50 才稳） |
| float64 极限环几乎不产生真 NaN | A15 核心检测面定为 trace/状态数据完整性 |
| contype/conaffinity 互斥语义 | `contype1&conaff2==0 AND contype2&conaff1==0` 才算被 mask |

## 4. 实施中抓出并修复的实质问题

1. **指标采集目标错位**：position_targets 在 run_rollout 内才应用，
   采集器构造时 ctrl 仍为 0——rmse 一度测的是 |qpos| 而非
   |qpos-target|。修复：预应用 + ctrl_series 逐行查表。
2. **H03 场景模型自由铰链重力自折**：未伺服 jz2 在重力下折叠
   -1 rad 与祖父级 geom 真穿透——场景模型伺服全部铰链。
3. **H02 修复面几何冲突**：A04 要求贴地而 A06 禁止挖掘——base
   重排为焊接体（贴地不动）+ 臂关节独立。
4. **fork 状态跨模型被正确拦截**：参数实验需要显式
   `transplant_state`（维度校验+provenance），不是放松
   CROSS_MODEL_REF。
5. **compare 阈值诚实化**：kp=10 vs 400 在 1s 窗口实测 rmse 比
   ~0.69，阈值取 0.8 留物理余量，不编造 0.5。
6. **load_model 重构回归**：broken MJCF 的 inspect 错误统一归
   MODEL_COMPILE_FAILED（回归测试锁定）。
7. **P0_TOOLS 注册顺序**：必须与 P0_AGENT_MCP_TOOLS 目录一致
   （test_server 锁定），sim 工具注册在尾部。

## 5. 验证证据（可复跑）

```bash
cd rosclaw
.venv/bin/python -m compileall -q src tests
.venv/bin/ruff check <栈内文件> && .venv/bin/ruff format --check <栈内文件>
.venv/bin/python -m mypy --config-file .github/mypy-ci.ini src/rosclaw/sim   # 新代码 0 error
.venv/bin/python -m pytest tests/sim -q            # 247 passed, 2 skipped(GL)
.venv/bin/python -m pytest tests/mcp -q            # 214 passed（含 e2e）
.venv/bin/python -m pytest tests/architecture tests/agentd/test_w02_contracts.py tests/eval/test_w09_cases.py -q
```

## 6. 残留风险与后续（Release Gate 留痕）

1. **G10 A 侧真实 LLM 对跑**：pending-live（需真实模型 key，W09
   PTY 串行纪律）——B 侧度量管线已锁定 false_success=0。
2. **G11 EGL 渲染复跑**：本机 headless GL 不可用，render 测试诚实
   skip；EGL 环境复跑后更新 Gate。
3. **to_xml mass=1.0 quirk**：patch geom.mass=1.0 丢 round-trip；
   后续 patch 层加 round-trip 校验或值域提示。
4. **A09-A14/A21-A24**（saturation/限值/sensor/坐标系/solver·
   timestep 敏感性）暂缓，audit 注册表已预留。
5. **sim_build_world / sim_interact 工具面**、Menagerie source、
   MJX/MJWarp 加速面、System Identification 均按总纲留待后续。
6. **api.py 历史缺口**：仍接受 task_root 外绝对路径（本栈不动其
   行为；新 resolve.py 已堵死，api.py 迁移 PR 时收编）。
7. **大规模实验**（1000 branches/10min rollout）仍同步执行——
   Operation/SimForge job 化属后续（总纲 §62）。

## 7. 合入顺序

#547 → #548 → #549 → #552 → #553 → #554 → #555 → #557 → #558。
每个 PR 合入后，将下一个 PR 的 base 改成 main（GitHub 会自动
重定向，或 `gh pr edit <N> --base main`）。
