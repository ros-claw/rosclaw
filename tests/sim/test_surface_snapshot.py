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


@pytest.mark.parametrize("separation", [0.4, 0.12])
def test_signed_surface_derivative_matches_tangent_finite_difference(separation):
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><freejoint/>"
        '<geom name="first" type="sphere" size=".1"/></body>'
        f'<body pos="{separation} .03 .01"><freejoint/>'
        '<geom name="second" type="sphere" size=".05"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    pair = (("first", "second"),)
    _assert_jacobian_finite_difference(model, data, pair)
    forward = surface_snapshot(model, data, pair, include_distance_jacobian=True)
    reverse = surface_snapshot(model, data, (("second", "first"),), include_distance_jacobian=True)
    np.testing.assert_allclose(
        forward["pairs"][0]["distance_jacobian_qvel"],
        reverse["pairs"][0]["distance_jacobian_qvel"],
        atol=1e-12,
    )


def _assert_jacobian_finite_difference(model, data, pair):
    names = ("qpos", "qvel", "qacc", "qacc_warmstart", "xpos", "geom_xpos", "cdof")
    before = {name: getattr(data, name).copy() for name in names}
    report = surface_snapshot(model, data, pair, include_distance_jacobian=True)
    assert report["schema"] == "rosclaw.sim.surface_snapshot.v2"
    row = report["pairs"][0]
    assert row["jacobian_status"] == "LOCAL_LINEARIZATION"
    numeric = []
    for dof in range(model.nv):
        direction = np.zeros(model.nv)
        direction[dof] = 1
        distances = []
        for step in (-1e-6, 1e-6):
            private = mujoco.MjData(model)
            private.qpos[:] = data.qpos
            mujoco.mj_integratePos(model, private.qpos, direction, step)
            distances.append(
                surface_snapshot(model, private, pair)["pairs"][0]["signed_distance_m"]
            )
        numeric.append((distances[1] - distances[0]) / 2e-6)
    np.testing.assert_allclose(row["distance_jacobian_qvel"], numeric, atol=2e-6, rtol=1e-5)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(data, name), value)


def test_articulated_contact_surface_not_body_origin_derivative():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".01"/>'
        '<body pos=".1 0 0"><joint axis="0 1 0"/>'
        '<geom name="foot" type="capsule" fromto="0 0 0 .15 0 0" size=".02"/>'
        '</body></body><body pos=".30 .02 .08"><freejoint/>'
        '<geom name="ball" type="sphere" size=".05"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    data.qpos[7] = 0.2
    _assert_jacobian_finite_difference(model, data, (("foot", "ball"),))
    row = surface_snapshot(model, data, (("foot", "ball"),), include_distance_jacobian=True)[
        "pairs"
    ][0]
    assert abs(row["distance_jacobian_qvel"][6]) > 0.05


@pytest.mark.parametrize(
    "height,reason", [(0.1, "near_zero_distance"), (3, "cutoff_or_unsupported_pair")]
)
def test_unknown_derivative_is_not_zero(height, reason):
    model, data = scene()
    data.qpos[2] = height
    row = surface_snapshot(model, data, (("foot", "floor"),), include_distance_jacobian=True)[
        "pairs"
    ][0]
    assert row["distance_jacobian_qvel"] is None
    assert row["jacobian_status"] == "UNKNOWN" and row["jacobian_reason"] == reason


def test_derivative_request_must_be_explicit_boolean():
    model, data = scene()
    with pytest.raises(ValueError):
        surface_snapshot(model, data, (("foot", "floor"),), include_distance_jacobian=1)
    row = surface_snapshot(model, data, (("foot", "floor"),))["pairs"][0]
    assert "distance_jacobian_qvel" not in row


def test_degenerate_native_segment_is_not_a_zero_gradient(monkeypatch):
    model, data = scene()

    def degenerate(model, private, first, second, cutoff, segment):
        segment[:] = 0
        return 0.1

    monkeypatch.setattr(mujoco, "mj_geomDistance", degenerate)
    row = surface_snapshot(model, data, (("foot", "floor"),), include_distance_jacobian=True)[
        "pairs"
    ][0]
    assert row["status"] == "MEASURED"
    assert row["distance_jacobian_qvel"] is None
    assert row["jacobian_reason"] == "degenerate_native_segment"
