"""并行分支状态语义测试（MH20-A，讨论总纲 §3-§6，红→绿）。

P0 实证（main f340ed34 复核）：branch_experiment(parallel=True) 的
rollout_batch 从未收到 caller 的 base_state——从各模型默认初态
起跑；serial 路径却 transplant_state + run_experiment(state_ref)。
caller 传非初始 state_ref 时两条路径不是同一个实验。

本文件四个硬测试 + 签名测试（B01-B06）全部要求 batch == serial
或诚实回退，绝不为了并行降低 state fidelity。
"""

from __future__ import annotations

import pytest

TINY_ARM = """<mujoco model="tiny_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="0.5"/>
      <geom name="base_geom" type="capsule" size="0.05 0.2" mass="1.0"/>
      <body name="forearm" pos="0 0 0.4">
        <joint name="elbow" type="hinge" axis="0 1 0" damping="0.1"/>
        <geom name="forearm_geom" type="capsule" size="0.03 0.2" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_servo" joint="shoulder" kp="10" ctrlrange="-1.57 1.57"/>
    <position name="elbow_servo" joint="elbow" kp="5"/>
  </actuator>
</mujoco>
"""

DELAY_ARM = TINY_ARM.replace(
    "</mujoco>",
    '  <sensor><jointpos name="sp" joint="shoulder" nsample="20" delay="0.02"/></sensor>\n</mujoco>',
)

WELD_WORLD = """<mujoco model="weld_world">
  <compiler autolimits="true"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="gripper_base" pos="0 0 0.3">
      <joint name="lift" type="slide" axis="0 0 1"/>
      <geom name="palm" type="box" size="0.05 0.05 0.02" mass="1.0"/>
      <body name="finger" pos="0.05 0 0">
        <joint name="grip" type="slide" axis="1 0 0" range="-0.03 0.03"/>
        <geom name="fg" type="box" size="0.01 0.02 0.02" mass="0.05"/>
      </body>
    </body>
    <body name="cube" pos="0.05 0 0.25">
      <freejoint name="cube_free"/>
      <geom name="cube_g" type="box" size="0.02 0.02 0.02" mass="0.1"/>
    </body>
  </worldbody>
  <equality>
    <weld name="grip_weld" body1="finger" body2="cube"/>
  </equality>
  <actuator>
    <position name="grip_servo" joint="grip" kp="50"/>
  </actuator>
</mujoco>
"""

PID_ARM = """<mujoco model="pid_arm">
  <compiler autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="link" pos="0 0 0.3">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.2"/>
      <geom name="g" type="capsule" size="0.04 0.2" mass="1.0"/>
    </body>
  </worldbody>
  <actuator>
    <pid name="p1" joint="j1" kp="10" kv="2"/>
  </actuator>
</mujoco>
"""


def _kp_patches(kp: float) -> list[dict]:
    return [
        {"op": "set", "target": {"type": "actuator", "name": "shoulder_servo"}, "field": "kp", "value": kp},
        {"op": "set", "target": {"type": "actuator", "name": "elbow_servo"}, "field": "kp", "value": kp},
    ]


def _branches() -> list[dict]:
    return [
        {"name": "baseline", "patches": []},
        {"name": "kp50", "patches": _kp_patches(50.0)},
        {"name": "kp200", "patches": _kp_patches(200.0)},
    ]


@pytest.fixture
def runtime(tmp_path):
    from rosclaw.sim.runtime import SimulationRuntime

    (tmp_path / "arm.xml").write_text(TINY_ARM, encoding="utf-8")
    (tmp_path / "delay_arm.xml").write_text(DELAY_ARM, encoding="utf-8")
    (tmp_path / "weld.xml").write_text(WELD_WORLD, encoding="utf-8")
    (tmp_path / "pid_arm.xml").write_text(PID_ARM, encoding="utf-8")
    return SimulationRuntime(tmp_path)


def _mid_swing_state(runtime, model_ref: str, steps: int = 60) -> str:
    """构造非初始状态：跑一段后快照（qpos≠0、qvel≠0、time>0）。"""
    receipt = runtime.rollout(model_ref, controller={"position_targets": [0.5, 0.2]}, steps=steps)
    return receipt["final_state_ref"]


def _final_qpos_time(runtime, receipt: dict) -> tuple[list[float], float]:
    state = runtime.backend.store.get(receipt["final_state_ref"])
    return [float(v) for v in state["qpos"]], float(state["time"])


def test_b01_nonzero_state_batch_equals_serial(runtime) -> None:
    """B01：qpos≠0/qvel≠0 的 caller state——parallel 必须与 serial
    同一实验（修复前 parallel 从默认初态起跑，必红）。"""
    loaded = runtime.load_model("arm.xml")
    base_state = _mid_swing_state(runtime, loaded["model_ref"])
    serial = runtime.branch_experiment(
        loaded["model_ref"],
        branches=_branches(),
        controller={"position_targets": [0.4, 0.2]},
        state_ref=base_state,
        steps=80,
        parallel=False,
    )
    parallel = runtime.branch_experiment(
        loaded["model_ref"],
        branches=_branches(),
        controller={"position_targets": [0.4, 0.2]},
        state_ref=base_state,
        steps=80,
        parallel=True,
    )
    assert parallel["execution"] == "batch_parallel"
    for s_receipt, p_receipt in zip(serial["receipts"], parallel["receipts"], strict=True):
        s_qpos, s_time = _final_qpos_time(runtime, s_receipt)
        p_qpos, p_time = _final_qpos_time(runtime, p_receipt)
        assert p_qpos == pytest.approx(s_qpos, abs=1e-9), (
            f"branch {s_receipt['model_ref'][:20]}: parallel {p_qpos} != serial {s_qpos}"
        )
        assert p_time == pytest.approx(s_time, abs=1e-9)


def test_b02_nonzero_time_continuity(runtime) -> None:
    """B02：time>0 的快照分叉后 time 连续性正确（不从 0 重计）。"""
    loaded = runtime.load_model("arm.xml")
    base_state = _mid_swing_state(runtime, loaded["model_ref"], steps=50)
    base_meta = runtime.backend.store.get(base_state)
    base_time = float(base_meta["time"])
    assert base_time > 0
    parallel = runtime.branch_experiment(
        loaded["model_ref"],
        branches=_branches(),
        controller={"position_targets": [0.4, 0.2]},
        state_ref=base_state,
        steps=80,
        parallel=True,
    )
    for receipt in parallel["receipts"]:
        _qpos, final_time = _final_qpos_time(runtime, receipt)
        assert final_time == pytest.approx(base_time + 80 * 0.002, abs=1e-9)


def test_b03_eq_active_state_requires_serial(runtime) -> None:
    """B03：eq_active=1（weld 激活）状态 ∈ INTEGRATION-only——
    native batch 无法承载 → 必须诚实 serial 回退（不得假装兼容），
    且 serial 结果与 parallel=False 完全一致。"""
    loaded = runtime.load_model("weld.xml")
    model_ref = loaded["model_ref"]
    snap = runtime.snapshot(model_ref)
    attached = runtime.interact(
        model_ref,
        snap["state_ref"],
        {"executor": "constraint_attach", "target": {"type": "equality", "name": "grip_weld"}},
        {"weld": "grip_weld", "attach_threshold_m": 0.05, "evidence_level": "PROXIMITY_ASSISTED_ATTACH"},
    )
    eq_state = attached["state_ref"]

    def weld_branches() -> list[dict]:
        return [
            {"name": "baseline", "patches": []},
            {"name": "kp100", "patches": [
                {"op": "set", "target": {"type": "actuator", "name": "grip_servo"}, "field": "kp", "value": 100.0},
            ]},
        ]

    serial = runtime.branch_experiment(
        model_ref,
        branches=weld_branches(),
        controller={"position_targets": [0.02]},
        state_ref=eq_state,
        steps=40,
        parallel=False,
    )
    parallel = runtime.branch_experiment(
        model_ref,
        branches=weld_branches(),
        controller={"position_targets": [0.02]},
        state_ref=eq_state,
        steps=40,
        parallel=True,
    )
    # eq_active 承载不了 → 不允许 batch_parallel（诚实回退）。
    assert parallel["execution"] != "batch_parallel"
    for s_receipt, p_receipt in zip(serial["receipts"], parallel["receipts"], strict=True):
        s_qpos, _ = _final_qpos_time(runtime, s_receipt)
        p_qpos, _ = _final_qpos_time(runtime, p_receipt)
        assert p_qpos == pytest.approx(s_qpos, abs=1e-9)


def test_b04_delay_history_batch_equals_serial(runtime) -> None:
    """B04：delay sensor history ∈ FULLPHYSICS（实证）——batch 可以
    承载但必须真一致（不得丢 history）。"""
    loaded = runtime.load_model("delay_arm.xml")
    base_state = _mid_swing_state(runtime, loaded["model_ref"])
    serial = runtime.branch_experiment(
        loaded["model_ref"],
        branches=_branches(),
        controller={"position_targets": [0.4, 0.2]},
        state_ref=base_state,
        steps=60,
        parallel=False,
    )
    parallel = runtime.branch_experiment(
        loaded["model_ref"],
        branches=_branches(),
        controller={"position_targets": [0.4, 0.2]},
        state_ref=base_state,
        steps=60,
        parallel=True,
    )
    assert parallel["execution"] == "batch_parallel"
    for s_receipt, p_receipt in zip(serial["receipts"], parallel["receipts"], strict=True):
        s_trace = runtime.backend.store.get(s_receipt["trace_ref"])
        p_trace = runtime.backend.store.get(p_receipt["trace_ref"])
        # 延迟读数一致 = history 被 batch 正确承载。
        s_sensor = runtime.backend.observe(
            s_receipt["model_ref"], s_receipt["final_state_ref"], ["sensor:sp"]
        ).values
        p_sensor = runtime.backend.observe(
            p_receipt["model_ref"], p_receipt["final_state_ref"], ["sensor:sp"]
        ).values
        assert p_sensor == pytest.approx(s_sensor, abs=1e-9)
        assert p_trace["states_digest"] == s_trace["states_digest"]


def test_b05_act_state_batch_equals_serial(runtime) -> None:
    """B05：PID 执行器非零 act（激活态 ∈ FULLPHYSICS）——batch 一致。"""
    loaded = runtime.load_model("pid_arm.xml")
    # PID 多输入：mid-swing 也必须用 setpoints（position_targets 拒绝）。
    receipt = runtime.rollout(loaded["model_ref"], controller={"setpoints": {"p1": {"pos": 0.5}}}, steps=40)
    base_state = receipt["final_state_ref"]
    # 分支 patch joint damping（kp 对 gaintype=PID 属白名单拒绝——
    # patch 守卫实证，不改守卫）。
    branches = [
        {"name": "baseline", "patches": []},
        {"name": "damp2", "patches": [
            {"op": "set", "target": {"type": "joint", "name": "j1"}, "field": "damping", "value": 2.0},
        ]},
    ]
    serial = runtime.branch_experiment(
        loaded["model_ref"],
        branches=branches,
        controller={"setpoints": {"p1": {"pos": 0.3}}},
        state_ref=base_state,
        steps=60,
        parallel=False,
    )
    parallel = runtime.branch_experiment(
        loaded["model_ref"],
        branches=branches,
        controller={"setpoints": {"p1": {"pos": 0.3}}},
        state_ref=base_state,
        steps=60,
        parallel=True,
    )
    assert parallel["execution"] == "batch_parallel"
    for s_receipt, p_receipt in zip(serial["receipts"], parallel["receipts"], strict=True):
        s_qpos, _ = _final_qpos_time(runtime, s_receipt)
        p_qpos, _ = _final_qpos_time(runtime, p_receipt)
        assert p_qpos == pytest.approx(s_qpos, abs=1e-9)


def test_b06_timestep_mismatch_semantics_incompatible(runtime) -> None:
    """B06：分支 patch 了 option.timestep → 500 steps 不再同时长——
    BATCH_SEMANTICS_INCOMPATIBLE → 诚实 serial 回退（不混比较）。"""
    loaded = runtime.load_model("arm.xml")
    branches = [
        {"name": "dt2", "patches": []},
        {"name": "dt1", "patches": [
            {"op": "set", "target": {"type": "option"}, "field": "timestep", "value": 0.001},
        ]},
    ]
    result = runtime.branch_experiment(
        loaded["model_ref"],
        branches=branches,
        controller={"position_targets": [0.4, 0.2]},
        steps=60,
        parallel=True,
    )
    assert result["execution"] == "serial"
    assert result.get("serial_fallback_reason", "").startswith("BATCH_")
