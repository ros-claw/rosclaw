"""Native force sign and integration-state semantics, without robot assets."""

import numpy as np
import pytest

from rosclaw.sim.backends.mujoco.contact import contact_snapshot

mujoco = pytest.importorskip("mujoco")


def scene():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep=".002"/><worldbody>'
        '<geom name="floor" type="plane" size="2 2 .1"/>'
        '<body pos="0 0 .095"><freejoint/>'
        '<geom name="object" type="sphere" size=".1" mass="1"/>'
        "</body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    data.qvel[:3] = (0.3, 0, -0.2)
    mujoco.mj_forward(model, data)
    return model, data


def query(model, data, **changes):
    kwargs = {
        "evaluated_qpos": tuple(data.qpos.tolist()),
        "evaluated_qvel": tuple(data.qvel.tolist()),
        "evaluated_time_sec": float(data.time),
        "geometry_names": ("object",),
    }
    kwargs.update(changes)
    return contact_snapshot(model, data, **kwargs)


def test_force_sign_velocity_and_no_mutation():
    model, data = scene()
    before = {
        n: getattr(data, n).copy()
        for n in ("qpos", "qvel", "ctrl", "qacc", "qacc_warmstart", "efc_force", "xpos")
    }
    report = query(model, data)
    (row,) = report["contacts"]
    assert row["geometry_names"] == ["floor", "object"]
    assert row["force_on_second_world_n"][2] > 0
    np.testing.assert_allclose(row["force_on_second_world_n"], data.qfrc_constraint[:3])
    assert row["relative_normal_velocity_mps"] == pytest.approx(-0.2)
    assert row["relative_tangential_speed_mps"] == pytest.approx(0.3)
    assert report["impulse_measured"] is False
    assert report["task_success_verified"] is False
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(data, name), value)


@pytest.mark.parametrize("integrator", [0, 2, 3])
def test_post_step_uses_preintegration_pose_and_velocity(integrator):
    model, data = scene()
    model.opt.integrator = integrator
    q, v, time = tuple(data.qpos.tolist()), tuple(data.qvel.tolist()), float(data.time)
    mujoco.mj_step(model, data)
    report = query(model, data, evaluated_qpos=q, evaluated_qvel=v, evaluated_time_sec=time)
    assert report["contacts"][0]["relative_normal_velocity_mps"] == pytest.approx(-0.2)
    with pytest.raises(ValueError, match="kinematics"):
        query(model, data)


@pytest.mark.parametrize(
    "changes",
    [
        {"geometry_names": ()},
        {"geometry_names": ["object"]},
        {"geometry_names": ("absent",)},
        {"geometry_names": ("object", "object")},
        {"evaluated_time_sec": True},
        {"evaluated_time_sec": float("nan")},
        {"evaluated_time_sec": -1},
        {"evaluated_time_sec": 1},
        {"evaluated_qpos": ()},
        {"evaluated_qvel": (float("inf"),) * 6},
        {"evaluated_qpos": (0.0, 0.0, 0.095, 2.0, 0.0, 0.0, 0.0)},
    ],
)
def test_bad_query_rejected(changes):
    model, data = scene()
    with pytest.raises(ValueError):
        query(model, data, **changes)


def test_wrong_model_rk4_and_stale_evaluation_rejected():
    model, data = scene()
    other, _ = scene()
    with pytest.raises(ValueError):
        query(other, data)
    model.opt.integrator = 1
    with pytest.raises(ValueError):
        query(model, data)
    model.opt.integrator = 0
    with pytest.raises(ValueError, match="kinematics"):
        query(model, data, evaluated_qvel=(0.0,) * 6)


def test_empty_contacts_do_not_prove_success():
    model, data = scene()
    data.qpos[2] = 1
    mujoco.mj_forward(model, data)
    result = query(model, data)
    assert result["contacts"] == [] and not result["task_success_verified"]


def test_interleaved_mutation_is_detected_without_restore(monkeypatch):
    model, data = scene()
    original = mujoco.mj_contactForce

    def mutate(m, d, index, result):
        original(m, d, index, result)
        d.time += 0.002

    monkeypatch.setattr(mujoco, "mj_contactForce", mutate)
    with pytest.raises(RuntimeError, match="changed"):
        query(model, data)
    assert data.time == 0.002


def test_nonfinite_native_force_rejected(monkeypatch):
    model, data = scene()

    def corrupt(m, d, index, result):
        result[:] = float("nan")

    monkeypatch.setattr(mujoco, "mj_contactForce", corrupt)
    with pytest.raises(ValueError, match="orthonormal"):
        query(model, data)


def test_two_dynamic_bodies_opposite_forces_and_rotating_contact_velocity():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option gravity="0 0 0"/><worldbody>'
        '<body><freejoint/><geom name="first" type="sphere" size=".1" mass="1"/></body>'
        '<body pos=".18 0 0"><freejoint/>'
        '<geom name="second" type="sphere" size=".1" mass="1"/></body>'
        "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    data.qvel[5] = 2
    mujoco.mj_forward(model, data)
    report = query(model, data, geometry_names=("first", "second"))
    (row,) = report["contacts"]
    np.testing.assert_allclose(row["force_on_second_world_n"], data.qfrc_constraint[6:9])
    np.testing.assert_allclose(-np.array(row["force_on_second_world_n"]), data.qfrc_constraint[:3])
    expected = np.cross((0.0, 0.0, 2.0), row["point_world_m"])
    np.testing.assert_allclose(row["first_point_velocity_world_mps"], expected)
    assert row["relative_tangential_speed_mps"] > 0.1


def test_observing_entire_trajectory_is_exactly_noninterfering():
    model, original = scene()
    observed = mujoco.MjData(model)
    observed.qpos[:] = original.qpos
    observed.qvel[:] = original.qvel
    mujoco.mj_forward(model, observed)
    for _ in range(200):
        q, v, time = (
            tuple(observed.qpos.tolist()),
            tuple(observed.qvel.tolist()),
            float(observed.time),
        )
        mujoco.mj_step(model, original)
        mujoco.mj_step(model, observed)
        query(model, observed, evaluated_qpos=q, evaluated_qvel=v, evaluated_time_sec=time)
        for name in ("qpos", "qvel", "qacc_warmstart", "efc_force"):
            np.testing.assert_array_equal(getattr(original, name), getattr(observed, name))


@pytest.mark.parametrize("field", ["geom_xpos", "cvel", "qvel", "efc_force"])
def test_nonfinite_live_cache_rejected(field):
    model, data = scene()
    getattr(data, field).flat[0] = float("nan")
    with pytest.raises(ValueError):
        query(model, data)


def test_changed_contact_cache_is_detected(monkeypatch):
    model, data = scene()
    original = mujoco.mj_contactForce

    def mutate(m, d, index, result):
        original(m, d, index, result)
        d.contact[index].dist += 0.001

    monkeypatch.setattr(mujoco, "mj_contactForce", mutate)
    with pytest.raises(RuntimeError, match="changed"):
        query(model, data)
