"""Observation API 测试（PR-MH3，ADR-0014，规格 §16，红→绿）。"""

from __future__ import annotations

import pytest

CONTACT_MJCF = """<mujoco model="ball_drop">
  <compiler autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="ball" pos="0 0 0.05">
      <freejoint name="ball_free"/>
      <geom name="ball_geom" type="sphere" size="0.05" mass="0.1"/>
    </body>
    <camera name="side" pos="1 0 0.2"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def contact_backend(tmp_path):
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    (tmp_path / "ball.xml").write_text(CONTACT_MJCF, encoding="utf-8")
    backend = MujocoBackend(tmp_path)
    ref = backend.load_model("ball.xml")
    return backend, ref


def test_observe_joint_channels(loaded_backend) -> None:
    backend, ref = loaded_backend
    trace = backend.rollout(
        ref.model_ref, controller={"position_targets": [0.3, 0.1]}, duration_s=0.1
    )
    result = backend.observe(
        ref.model_ref,
        trace.final_state_ref,
        ["joint_positions", "joint_velocities", "actuator_force"],
    )
    assert result.model_ref == ref.model_ref
    assert result.time == pytest.approx(0.1)
    assert result.values["joint_positions"][0] > 0.05
    assert len(result.values["joint_velocities"]) == 2
    assert len(result.values["actuator_force"]) == 2


def test_observe_body_and_site_pose(loaded_backend) -> None:
    backend, ref = loaded_backend
    state_ref = backend.initial_state(ref.model_ref)
    result = backend.observe(ref.model_ref, state_ref, ["body_pose:forearm", "site_pose:tool0"])
    pose = result.values["body_pose:forearm"]
    assert pose["pos"] == pytest.approx([0.0, 0.0, 0.4])
    assert len(pose["quat"]) == 4
    site = result.values["site_pose:tool0"]
    assert site["pos"][2] == pytest.approx(0.7)


def test_observe_sensor_channel(loaded_backend) -> None:
    backend, ref = loaded_backend
    state_ref = backend.initial_state(ref.model_ref)
    result = backend.observe(ref.model_ref, state_ref, ["sensor:shoulder_pos"])
    assert result.values["sensor:shoulder_pos"] == pytest.approx([0.0])


def test_observe_energy_and_com(loaded_backend) -> None:
    backend, ref = loaded_backend
    state_ref = backend.initial_state(ref.model_ref)
    result = backend.observe(ref.model_ref, state_ref, ["energy", "com"])
    energy = result.values["energy"]
    assert set(energy) == {"potential", "kinetic", "total"}
    assert energy["potential"] != 0.0
    assert len(result.values["com"]) == 3


def test_observe_contact_channels_bounded(contact_backend) -> None:
    backend, ref = contact_backend
    # 球放在地板上（球心 0.05 = 半径）→ 有接触。
    state_ref = backend.initial_state(ref.model_ref)
    result = backend.observe(
        ref.model_ref, state_ref, ["contact_summary", "contact_pairs", "contact_force"]
    )
    summary = result.values["contact_summary"]
    assert summary["count"] >= 1
    assert len(summary["pairs"]) <= 50
    assert "max_penetration" in summary
    pairs = result.values["contact_pairs"]
    assert pairs[0]["geom1"] == "floor" or pairs[0]["geom2"] == "floor"


def test_observe_unknown_channel_fails(loaded_backend) -> None:
    backend, ref = loaded_backend
    state_ref = backend.initial_state(ref.model_ref)
    with pytest.raises(ValueError, match="OBSERVE_CHANNEL_UNKNOWN"):
        backend.observe(ref.model_ref, state_ref, ["raw_geom_arrays"])
    with pytest.raises(ValueError, match="OBSERVE_CHANNEL_UNKNOWN"):
        backend.observe(ref.model_ref, state_ref, ["body_pose:ghost"])
    with pytest.raises(ValueError, match="OBSERVE_CHANNEL_UNKNOWN"):
        backend.observe(ref.model_ref, state_ref, ["sensor:ghost"])


def test_observe_cross_model_rejected(loaded_backend) -> None:
    backend, ref = loaded_backend
    other = backend.patch_model(
        ref.model_ref,
        [
            {
                "op": "set",
                "target": {"type": "joint", "name": "shoulder"},
                "field": "damping",
                "value": 2.0,
            }
        ],
    )
    state_ref = backend.initial_state(other.new_model_ref)
    with pytest.raises(ValueError, match="CROSS_MODEL_REF"):
        backend.observe(ref.model_ref, state_ref, ["joint_positions"])
