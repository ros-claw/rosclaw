"""An elevated body origin is not an elevated collision surface."""

import numpy as np
import pytest

from rosclaw.sim.backends.mujoco.surface import surface_snapshot

mujoco = pytest.importorskip("mujoco")


def scene():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .1"/>'
        '<body pos="0 0 .1"><freejoint/><geom name="foot" type="box" size=".2 .1 .1"/>'
        "</body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_current_surface_not_stale_body_origin_and_no_live_cache_refresh():
    model, data = scene()
    data.qpos[2] = 0.3  # Derived xpos intentionally still describes the old pose.
    names = ("qpos", "qvel", "qacc", "qacc_warmstart", "xpos", "geom_xpos", "energy")
    before = {name: getattr(data, name).copy() for name in names}
    result = surface_snapshot(model, data, (("foot", "floor"),))
    assert result["pairs"][0]["signed_distance_m"] == pytest.approx(0.2)
    assert result["continuous_clearance_verified"] is False
    assert result["contact_forces_measured"] is False
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(data, name), value)


def test_tilted_foot_origin_can_be_high_while_surface_penetrates():
    model, data = scene()
    data.qpos[2] = 0.15
    data.qpos[3:7] = (np.cos(np.pi / 8), 0, np.sin(np.pi / 8), 0)
    row = surface_snapshot(model, data, (("foot", "floor"),))["pairs"][0]
    assert row["status"] == "MEASURED"
    assert row["signed_distance_m"] < 0


def test_cutoff_is_unknown_not_free_space():
    model, data = scene()
    data.qpos[2] = 3
    row = surface_snapshot(model, data, (("foot", "floor"),))["pairs"][0]
    assert row["status"] == "UNKNOWN"
    assert row["signed_distance_m"] is None
    assert row["nearest_segment_world_m"] is None


@pytest.mark.parametrize(
    "pairs",
    [
        (),
        (("foot", "foot"),),
        (("missing", "floor"),),
        (("foot", "floor"), ("floor", "foot")),
        [("foot", "floor")],
    ],
)
def test_invalid_pairs_fail_closed(pairs):
    model, data = scene()
    with pytest.raises(ValueError):
        surface_snapshot(model, data, pairs)


@pytest.mark.parametrize("limit", [True, 0, -1, float("nan"), float("inf"), 11])
def test_invalid_cutoff(limit):
    model, data = scene()
    with pytest.raises(ValueError):
        surface_snapshot(model, data, (("foot", "floor"),), maximum_distance_m=limit)


def test_wrong_model_or_nonfinite_pose_or_nonunit_quaternion():
    model, data = scene()
    other, _ = scene()
    with pytest.raises(ValueError):
        surface_snapshot(other, data, (("foot", "floor"),))
    for q in ((float("nan"), 0, 0, 0), (2.0, 0, 0, 0)):
        data.qpos[3:7] = q
        with pytest.raises(ValueError):
            surface_snapshot(model, data, (("foot", "floor"),))


def test_detects_interleaved_step_without_overwriting_live_state(monkeypatch):
    model, data = scene()
    original = mujoco.mj_kinematics

    def interleave(m, private):
        assert private is not data
        original(m, private)
        data.time += 0.002

    monkeypatch.setattr(mujoco, "mj_kinematics", interleave)
    with pytest.raises(RuntimeError, match="pose changed"):
        surface_snapshot(model, data, (("foot", "floor"),))
    assert data.time == 0.002
