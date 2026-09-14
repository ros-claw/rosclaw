# MUJOCO_HARNESS_V1 Release Gate（MH8，2026-09-14）

总纲 §69：每个 Gate 必须绑定 **test + report + commit**，不写"理论支持"。
本文件是 MuJoCo Simulation Harness v1 的发布门禁。

栈叠 PR 序列（按合入顺序）：
**#547(MH0) → #548(MH1) → #549(MH2) → #552(MH3) → #553(MH4) → #554(MH5) → #555(MH6) → #557(MH7) → 本 PR(MH8)**

统一验证命令（每个 PR 均跑过）：

```bash
.venv/bin/python -m compileall -q src tests
.venv/bin/ruff check <PR 文件>
.venv/bin/ruff format --check <PR 文件>
.venv/bin/python -m pytest tests/sim tests/architecture \
  tests/agentd/test_w02_contracts.py tests/mcp -q
```

| Gate | 内容 | 绑定测试（路径） | 状态 |
|---|---|---|---|
| G1 Architecture boundary | MuJoCo 不是 NativeHarnessBackend；无 `--engine pi/codex`；无 Harness 名称泄漏 | `tests/architecture/test_adr0014_simulation_harness.py`、`tests/agentd/test_adr0012a_no_engine_surface.py`、ADR-0014 | ✅ PASS（MH0，#547） |
| G2 Model identity | model_ref/model_digest/亲缘 immutable，patch 必产新 ref | `tests/sim/test_model_load.py`、`test_model_patch.py`（lineage/idempotent/母模型不动） | ✅ PASS（MH2，#549） |
| G3 Immutable refs | 内容寻址 ref；路径/symlink/URI 逃逸 fail closed；绝不覆写 | `tests/sim/test_refs.py`、`test_store.py` | ✅ PASS（MH1，#548） |
| G4 State restore | 完整状态快照/恢复往返一致 | `tests/sim/test_state_snapshot.py`、`test_state_restore.py` | ✅ PASS（MH3，#552） |
| G5 Cross-model fail closed | 跨模型 state/trace → CROSS_MODEL_REF；维度/NaN 拒绝 | `tests/sim/test_state_restore.py`、`test_rollout.py`、`test_observe.py`、`test_audit_determinism.py`（A20） | ✅ PASS（MH3/MH4） |
| G6 Audit red fixtures | 每个 audit 至少 1 red + 1 green fixture 且全部被正确抓住 | `tests/sim/fixtures/{broken,fixed}_models/`、`test_audit_geometry/contact/control/determinism.py` | ✅ PASS（MH4，#553） |
| G7 Strict replay | REPLAY_DIVERGED 判定；双层 digest（raw+semantic） | `tests/sim/test_strict_replay.py` | ✅ PASS（MH5，#554） |
| G8 MCP safety | 全部 sim tool ≤ S1；usable_for_real_execution=false；零 REAL permit | `tests/mcp/test_sim_tool_catalog.py`、`tests/mcp/test_e2e.py`（stdio+http 冒烟） | ✅ PASS（MH6，#555） |
| G9 Missing-capability honesty | 无夹爪 grasp → CAPABILITY_UNAVAILABLE，无 fake weld | `tests/sim/test_worldspec.py`（grasp 拒绝/放行）、`test_agent_scenarios.py::test_h05` | ✅ PASS（MH7，#557） |
| G10 Agent A/B | Harness 路径少 glue code、false_success=0 | B 侧度量管线：`tests/sim/test_ab_harness.py`（false_success=0 锁定） | 🟡 PARTIAL：B 侧锁定；**A 侧真实 LLM 对跑 pending-live**（需真实模型 key，按 W09 PTY 串行纪律单独跑） |
| G11 Rendering evidence | trace→GIF artifact；渲染后端诚实记录 | `tests/sim/test_runtime.py::test_render_produces_artifact`、`test_agent_scenarios.py::test_h04` | 🟡 PARTIAL：实现完成；本机 headless GL 不可用 → 诚实 skip（`SIM_RENDER_UNAVAILABLE` 不静默降级），EGL 环境复跑 |
| G12 Full regression | 全套件无回归 | `pytest tests/sim tests/mcp tests/architecture tests/agentd/test_w02_contracts.py tests/eval/test_w09_cases.py` | ✅ PASS：**281+ passed**（W09 物理层 6 passed/6 skip，agent 层按纪律标 NOT_RUN） |

## H01-H08 场景验收（规格 §47-§54）

绑定 `tests/sim/test_agent_scenarios.py`（harness 能力层，Verifier 直接对
MuJoCo 模型/物理真相；真实 LLM 变体按 W09 纪律 pending-live）：

| 场景 | 验证点 | 状态 |
|---|---|---|
| H01 Unknown Body | inspect 推导 = 模型真相（不从名字猜） | ✅ |
| H02 Broken Model Doctor | audit 定位三注入缺陷 → patch 修复 → audit PASS，母模型仍 FAIL | ✅ |
| H03 Parameter Scientist | baseline→snapshot→fork(3)→kp 扫描→compare→best 优于 baseline 且 ∈ Pareto | ✅ |
| H04 World Generation | WorldSpec→compile→interaction→audit PASS→task predicate 诚实判 False→render 证据 | ✅（render 本机 skip） |
| H05 Honest Missing Capability | 无夹爪 grasp → CAPABILITY_UNAVAILABLE | ✅ |
| H06 Mid-path Collision | endpoint PASS 但 A06 sequence FAIL | ✅ |
| H07 Numerical Fragility | dt 0.002 vs 0.001 指标差异被量化报告 | ✅ |
| H08 Revision Chain | patch 亲缘链 parent 正确，无重复世界重建 | ✅ |

## 关键实证修正记录（诚实性审计线索）

1. H03 的 x7 模型未驱动关节在重力下自折导致 A06 真穿透——场景模型
   必须伺服所有铰链（物理教训：自由铰链不是"零成本"）。
2. MetricCollector 采集目标必须在采集前应用（position_targets 预应用 +
   ctrl_series 逐行查表），否则 rmse 测的是 |qpos| 而非 |qpos-target|。
3. MuJoCo 3.11 `to_xml` 对 `mass=1.0` 特殊省略——patch geom.mass=1.0
   会丢失 round-trip（已记 ADR-0014 残留风险；H02 用 mass=8.0 规避）。
4. 参数实验需要**显式状态移植**（`transplant_state`：维度校验 +
   provenance），不能让 fork 状态静默跨模型（CROSS_MODEL_REF 是对的）。
5. compare 测试阈值诚实化：kp=10 vs 400 在 1s 窗口实测 rmse 比 ~0.69，
   阈值取 0.8 留物理余量，不编造 0.5。

## Definition of Done 对照（总纲 §68）

- Architecture/Model/State/Experiment/Physical honesty/Evidence/Agent/Safety
  各项：见 G1-G9、G12 全 PASS。
- Tests：broken fixtures 全被抓（G6）；W09 无回归（G12）；
  Harness A/B false_success=0（G10 B 侧）。
- 未完成项：G10 A 侧真实 LLM 对跑、G11 EGL 环境渲染复跑——
  两项均为 pending-live，不阻塞代码面 v1 合入，但必须在本文件
  留痕并在补跑后更新状态。
