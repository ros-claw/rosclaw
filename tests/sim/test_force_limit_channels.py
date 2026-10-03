"""Compiled counterexamples: joint clamps, distinct bounds and safety profiles."""

import pytest

from rosclaw.sim.audit.context import AuditContext
from rosclaw.sim.audit.limits import a10_force_limit


def context(joints, actuators, ctrl, safety=None):
    import mujoco

    bodies = "".join(
        f'<body pos="{i} 0 0"><joint name="j{i}" type="hinge" {attrs}/>'
        '<geom type="sphere" size=".1" mass="1"/></body>'
        for i, attrs in enumerate(joints)
    )
    xml = f'<mujoco><option gravity="0 0 0"/><worldbody>{bodies}</worldbody><actuator>{actuators}</actuator></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    return AuditContext(
        model=model,
        spec=None,
        xml_text=xml,
        trace_record={"steps": 20, "controller": {"ctrl_series": [ctrl] * 20}},
        extra={"safety_limits": safety} if safety is not None else {},
    )


def test_joint_layer_clamp_is_evaluated():
    ctx = context(['actuatorfrcrange="-2 2"'], '<motor joint="j0"/>', [50])
    result = a10_force_limit(ctx)
    assert result["status"] == "FAIL"
    assert result["saturated_ratio"] == 1
    assert result["max_joint_actuator_force"] == pytest.approx(2)


def test_distinct_actuator_bounds_do_not_share_global_minimum():
    ctx = context(
        ["", ""],
        '<motor joint="j0" forcerange="-1 1"/><motor joint="j1" forcerange="-10 10"/>',
        [0, 3],
    )
    assert a10_force_limit(ctx)["status"] == "PASS"


def test_asymmetric_negative_force_boundary():
    ctx = context([""], '<motor joint="j0" forcerange="-2 10"/>', [-20])
    assert a10_force_limit(ctx)["status"] == "FAIL"


def test_safety_only_joint_effort_checks_transmitted_force():
    ctx = context(
        [""], '<motor joint="j0" gear="2"/>', [1.5], {"force_limits": {"max_joint_effort": 2}}
    )
    result = a10_force_limit(ctx)
    assert result["status"] == "FAIL"
    assert any(v["reason"] == "joint_force_limit_exceeded" for v in result["violations"])


def test_pid_multiple_controls_are_not_actuator_count():
    ctx = context([""], '<pid joint="j0" kp="10" kv="2" forcerange="-100 100"/>', [0, 0])
    assert ctx.model.nu > ctx.model.actuator_trnid.shape[0]
    assert a10_force_limit(ctx)["status"] == "PASS"


def test_zero_lower_force_bound_is_not_idle_saturation():
    ctx = context([""], '<motor joint="j0" forcerange="0 10"/>', [0])
    assert a10_force_limit(ctx)["status"] == "PASS"


def test_joint_limit_applies_to_combined_motor_forces():
    ctx = context(['actuatorfrcrange="-2 2"'], '<motor joint="j0"/><motor joint="j0"/>', [1.5, 1.5])
    result = a10_force_limit(ctx)
    assert result["status"] == "FAIL"
    assert result["max_actuator_force"] == pytest.approx(1.5)
    assert result["max_joint_actuator_force"] == pytest.approx(2)


@pytest.mark.parametrize("effort", [-1, "invalid", float("nan"), float("inf")])
def test_invalid_safety_effort_fails_closed(effort):
    ctx = context([""], '<motor joint="j0"/>', [0], {"force_limits": {"max_joint_effort": effort}})
    assert a10_force_limit(ctx)["status"] == "FAIL"


def test_zero_safety_effort_is_enforced():
    ctx = context([""], '<motor joint="j0"/>', [1], {"force_limits": {"max_joint_effort": 0}})
    assert a10_force_limit(ctx)["status"] == "FAIL"


def test_no_declared_limit_remains_not_evaluated():
    ctx = context([""], '<motor joint="j0"/>', [1])
    assert a10_force_limit(ctx)["status"] == "NOT_EVALUATED"


def test_full_inspection_exposes_joint_force_bounds():
    from rosclaw.sim.backends.mujoco.inspect import inspect_model_full

    ctx = context(['actuatorfrcrange="-2 2"'], '<motor joint="j0"/>', [0])
    joint = inspect_model_full(ctx.model)["joints_detail"][0]
    assert joint["actuatorfrclimited"] is True
    assert joint["actuatorfrcrange"] == [-2, 2]
