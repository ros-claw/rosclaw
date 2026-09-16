# PHYSICAL_HARNESS_V2 Release Gate（MH10+，2026-09-16）

0916 优化文档 §三十八：任何没有实证的标 **NOT_RUN**，不要 PASS。
前置门禁：docs/validation/MUJOCO_HARNESS_V1.md（G1-G13，Core v1 已闭环）。

| Gate | 内容 | 绑定测试 | 状态 |
|---|---|---|---|
| G14 Full Integration State Replay | mjSTATE_INTEGRATION 快照/恢复精确往返（含 time/history） | `tests/sim/test_state_v2.py::test_v2_roundtrip_exact` | ✅ PASS（MH10） |
| G15 Delayed Sensor/Actuator Replay | 延迟传感器：v1 partial 故意 FAIL，v2 FULL 精确 | `tests/sim/test_state_v2.py::test_s10_01_delay_replay_red` | ✅ PASS（MH10） |
| G16 Equality State Preservation | eq_active off/on 快照恢复分别保持 | `tests/sim/test_state_v2.py::test_s10_02_equality_state` | ✅ PASS（MH10） |
| G17 Replay Fidelity Rule | FULL_INTEGRATION→RAW_EXACT；LEGACY_PARTIAL→SEMANTIC 封顶 | `tests/sim/test_replay_fidelity.py` | ✅ PASS（MH10） |
| G18 Control Schema | PID 多输入 control_channels；position_targets 拒绝多输入；setpoints 按名寻址 | `tests/sim/test_control_schema.py` | ✅ PASS（MH10） |
| G19 MjVfs Portable Model | .mjz 自包含导出/重建（资产嵌入） | `tests/sim/test_model_mjz.py` | ✅ PASS（MH10b，含绑定不一致实证记录） |
| G20 True Agent H01-H08（HarnessBench live） | 无答案泄漏的独立 workspace + 外部 oracle + 真实 LLM | `benchmarks/harnessbench/`（框架待建） | ⏸ NOT_RUN（需真实模型 key） |
| G21 Real A/B Baseline vs Harness | A=coding agent，B=harness，同模型/prompt/seed | 同上 | ⏸ NOT_RUN（需真实模型 key） |
| G22 Executable Interaction Honesty | sim_interact + executor registry + GRASP_HONESTY | `tests/sim/test_interact.py`（grasp 诚实流/precondition/未声明 weld/未知 executor）、`tests/sim/test_predicates_v2.py` | ✅ PASS（MH12） |
| G23 Multimodal Observation Evidence | camera_rgb/depth/segmentation → artifact_ref | `tests/sim/test_observe_camera.py`（PNG magic/dtype/intrinsics/实际后端） | ✅ PASS（MH13） |
| G24 Parallel CPU Agreement | mujoco.rollout native batch 与串行一致性 | `tests/sim/test_batch_parallel.py`（轨迹逐步一致 abs 1e-9/异构拒绝/branch_experiment 并行+串行回退） | ✅ PASS（MH14） |
| G25 Numerical Robustness | A23/A24（solver/timestep/discrete diagnostic） | `tests/sim/test_audit_limits.py`（A09-A14/A21-A24 红绿 + NOT_EVALUATED 中性） | ✅ PASS（MH15） |
| G26 SysID Synthetic Recovery + Holdout | 合成数据参数恢复 + holdout 改进 | MH17 待做 | ⏸ NOT_RUN |

## MH10 实证记录（2026-09-16）

1. **S10-01 延迟重放实证**：v1 partial（time/qpos/qvel/ctrl）恢复后
   延迟传感器读数 = 0.0（history buffer 丢失），原始轨迹 = 1.133322，
   v2 INTEGRATION 恢复 = 1.133322——v1 快照对 delay/plugin/weld 类
   模型确实打穿 strict replay，MH10 是必修。
2. **PID ctrl 组成实证**：`<pid kp kv>` 占 2 个 ctrl 槽（pos, vel），
   `nu` 是 ctrl 维而非执行器个数；`actuator_actnum/actadr` 是激活
   状态地址而非 ctrl 映射（不可误用）。
3. **MjVfs 绑定实证**：`MjSpec.from_file/from_string(vfs=)` 对
   meshdir 资产不解析 VFS；`MjModel.from_xml_path(vfs=)` 正常；
   `spec.assets` 填充后 `to_zip` 嵌入资产、`from_zip` 自包含编译。
   MjSpec 资产面暂留 `assets=`（3.13 零弃用警告）。
