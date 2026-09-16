"""并行批次实验测试（MH14，0916 优化 §十七-§十九，红→绿）。

CPU batch 与权威 truth 同域：batch 轨迹必须与串行轨迹逐步一致
（G24 Parallel CPU Agreement）。
"""

from __future__ import annotations

import pytest

TINY_MJCF = """<mujoco model="tiny_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="0.5"/>
      <geom name="base_geom" type="capsule" size="0.05 0.2" mass="1.0"/>
      <body name="forearm" pos="0 0 0.4">
        <joint name="elbow" type="hinge" axis="0 1 0" damping="0.1"/>
        <geom name="forearm_geom" type="capsule" size="0.03 0.2" mass="0.5"/>
        <site name="tool0" pos="0 0 0.3"/>
      </body>
    </body>
    <camera name="top" pos="0 0 2"/>
  </worldbody>
  <actuator>
    <position name="shoulder_servo" joint="shoulder" kp="10" ctrlrange="-1.57 1.57" forcerange="-50 50"/>
    <position name="elbow_servo" joint="elbow" kp="5"/>
  </actuator>
  <sensor>
    <jointpos name="shoulder_pos" joint="shoulder"/>
  </sensor>
</mujoco>
"""

SIBLING_ARM = """<mujoco model="sibling_arm">
  <compiler autolimits="true"/>
  <worldbody>
    <body name="hip" pos="0 0 0">
      <joint name="wrist_roll" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="0.5"/>
      <geom name="s0" type="capsule" size="0.05 0.2" mass="1.0"/>
      <body name="segment" pos="0 0 0.4">
        <joint name="wrist_pitch" type="hinge" axis="0 1 0" damping="0.1"/>
        <geom name="s1" type="capsule" size="0.03 0.2" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="srv_a" joint="wrist_roll" kp="10"/>
    <position name="srv_b" joint="wrist_pitch" kp="5"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def backend(tmp_path):
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    (tmp_path / "arm.xml").write_text(TINY_MJCF, encoding="utf-8")
    b = MujocoBackend(tmp_path)
    return b, b.load_model("arm.xml")


def _kp_patches(kp: float):
    return [
        {
            "op": "set",
            "target": {"type": "actuator", "name": "shoulder_servo"},
            "field": "kp",
            "value": kp,
        },
        {
            "op": "set",
            "target": {"type": "actuator", "name": "elbow_servo"},
            "field": "kp",
            "value": kp,
        },
    ]


def test_batch_trajectory_matches_serial(backend) -> None:
    """G24：同 kp sweep，batch 与串行 qpos 轨迹逐步一致。"""
    b, ref = backend
    refs = [ref.model_ref]
    for kp in (50.0, 400.0):
        refs.append(b.patch_model(ref.model_ref, _kp_patches(kp)).new_model_ref)

    batch_results = b.rollout_batch(refs, controller={"position_targets": [0.4, 0.1]}, steps=100)
    assert len(batch_results) == 3
    assert all(r["execution"] == "batch_parallel" for r in batch_results)

    for ref_i, batch_result in zip(refs, batch_results, strict=True):
        serial = b.rollout(ref_i, controller={"position_targets": [0.4, 0.1]}, steps=100)
        serial_states = b.store.get(serial.trace_ref)["states"]
        batch_states = b.store.get(batch_result["trace_ref"])["states"]
        # 采样点一一对应，qpos 逐步一致。
        assert len(serial_states) == len(batch_states)
        for s_serial, s_batch in zip(serial_states, batch_states, strict=True):
            assert s_serial["qpos"] == pytest.approx(s_batch["qpos"], abs=1e-9)
            assert s_serial["ctrl"] == pytest.approx(s_batch["ctrl"], abs=1e-9)


def test_batch_final_state_matches_serial(backend) -> None:
    b, ref = backend
    refs = [ref.model_ref, b.patch_model(ref.model_ref, _kp_patches(100.0)).new_model_ref]
    batch_results = b.rollout_batch(refs, controller={"position_targets": [0.3, 0.0]}, steps=80)
    for ref_i, batch_result in zip(refs, batch_results, strict=True):
        serial = b.rollout(ref_i, controller={"position_targets": [0.3, 0.0]}, steps=80)
        serial_final = b.store.get(serial.final_state_ref)
        batch_final = b.store.get(batch_result["final_state_ref"])
        assert batch_final["qpos"] == pytest.approx(serial_final["qpos"], abs=1e-9)


def test_batch_rejects_heterogeneous(backend, tmp_path) -> None:
    (tmp_path / "sibling.xml").write_text(SIBLING_ARM, encoding="utf-8")
    b, ref = backend
    sibling = b.load_model("sibling.xml")
    with pytest.raises(ValueError, match="BATCH_NOT_HOMOGENEOUS"):
        b.rollout_batch(
            [ref.model_ref, sibling.model_ref],
            controller={"position_targets": [0.4, 0.1]},
            steps=10,
        )


def test_branch_experiment_parallel(tmp_path) -> None:
    from rosclaw.sim.runtime import SimulationRuntime

    (tmp_path / "arm.xml").write_text(TINY_MJCF, encoding="utf-8")
    runtime = SimulationRuntime(tmp_path)
    loaded = runtime.load_model("arm.xml")
    result = runtime.branch_experiment(
        loaded["model_ref"],
        branches=[
            {"name": "baseline", "patches": []},
            {"name": "kp100", "patches": _kp_patches(100.0)},
            {"name": "kp400", "patches": _kp_patches(400.0)},
        ],
        controller={"position_targets": [0.4, 0.1]},
        steps=80,
    )
    assert result["execution"] == "batch_parallel"
    assert result["count"] == 3
    for receipt in result["receipts"]:
        assert receipt["execution"] == "batch_parallel"
        assert receipt["trace_ref"].startswith("simtrc_")
        assert receipt["audit_ref"].startswith("simadt_")
        assert receipt["metrics_mode"] == "batch_trajectory"
        assert receipt["usable_for_real_execution"] is False


def test_branch_experiment_serial_still_works(tmp_path) -> None:
    from rosclaw.sim.runtime import SimulationRuntime

    (tmp_path / "arm.xml").write_text(TINY_MJCF, encoding="utf-8")
    runtime = SimulationRuntime(tmp_path)
    loaded = runtime.load_model("arm.xml")
    result = runtime.branch_experiment(
        loaded["model_ref"],
        branches=[{"name": "b0", "patches": []}, {"name": "b1", "patches": _kp_patches(60.0)}],
        controller={"position_targets": [0.4, 0.1]},
        steps=40,
        parallel=False,
    )
    assert result["execution"] == "serial"
    assert result["count"] == 2
    assert all(r["trace_ref"] for r in result["receipts"])
