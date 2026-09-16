"""Control Schema 测试（MH10，0916 优化 §三，红→绿）。

一个 actuator ≠ 一个 ctrl scalar（3.12 PID 多输入）。position_targets
只允许全部单输入 position actuator 的模型；setpoints 按名寻址。
"""

from __future__ import annotations

import pytest

PID_MODEL = """<mujoco model="pid_arm">
  <compiler autolimits="true"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <joint name="shoulder" type="hinge" axis="0 1 0"/>
      <geom name="g0" type="capsule" size="0.05 0.2" mass="1.0"/>
    </body>
  </worldbody>
  <actuator>
    <pid name="shoulder_pid" joint="shoulder" kp="10" kv="0.5"/>
    <position name="extra_pos" joint="shoulder" kp="5"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def pid_backend(tmp_path):
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    (tmp_path / "pid.xml").write_text(PID_MODEL, encoding="utf-8")
    backend = MujocoBackend(tmp_path)
    return backend, backend.load_model("pid.xml")


def test_control_channels_in_inspection(pid_backend) -> None:
    backend, ref = pid_backend
    detail = backend.inspect_model(ref.model_ref).detail
    channels = detail["control_channels"]
    # pid(kp+kv → pos+vel 两路) + position(1 路) = nu 3。
    assert len(channels) == 3
    assert channels[0]["actuator"] == "shoulder_pid"
    assert channels[0]["role"] == "pos"
    assert channels[1]["actuator"] == "shoulder_pid"
    assert channels[1]["role"] == "vel"
    assert channels[2]["actuator"] == "extra_pos"
    assert channels[2]["role"] == "ctrl"
    assert [c["index"] for c in channels] == [0, 1, 2]


def test_single_input_models_keep_position_targets(loaded_backend) -> None:
    backend, ref = loaded_backend
    trace = backend.rollout(ref.model_ref, controller={"position_targets": [0.3, 0.1]}, steps=50)
    final = backend.store.get(trace.final_state_ref)
    assert final["ctrl"] == [0.3, 0.1]


def test_position_targets_rejected_on_multi_input(pid_backend) -> None:
    backend, ref = pid_backend
    with pytest.raises(ValueError, match="CONTROLLER_SCHEMA_MISMATCH"):
        backend.rollout(ref.model_ref, controller={"position_targets": [0.4, 0.0, 0.0]}, steps=10)


def test_setpoints_by_name(pid_backend) -> None:
    backend, ref = pid_backend
    trace = backend.rollout(
        ref.model_ref,
        controller={"setpoints": {"shoulder_pid": {"pos": 0.4, "vel": 0.0}, "extra_pos": {"ctrl": 0.4}}},
        steps=100,
    )
    final = backend.store.get(trace.final_state_ref)
    assert final["ctrl"] == pytest.approx([0.4, 0.0, 0.4])
    assert final["qpos"][0] > 0.05  # pos setpoint 生效


def test_setpoints_unknown_target_rejected(pid_backend) -> None:
    backend, ref = pid_backend
    with pytest.raises(ValueError, match="CONTROLLER_SCHEMA_MISMATCH"):
        backend.rollout(ref.model_ref, controller={"setpoints": {"ghost": {"pos": 1.0}}}, steps=10)
    with pytest.raises(ValueError, match="CONTROLLER_SCHEMA_MISMATCH"):
        backend.rollout(
            ref.model_ref, controller={"setpoints": {"shoulder_pid": {"torque": 1.0}}}, steps=10
        )
