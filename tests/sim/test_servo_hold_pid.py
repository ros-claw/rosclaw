"""Compiled servo audit: multi-input PID and transmission-space hold targets."""

import mujoco
import pytest

from rosclaw.sim.audit.context import AuditContext
from rosclaw.sim.audit.dynamics import a03_servo_hold


def ctx(actuators):
    xml = f"""<mujoco><compiler angle="radian"/><option gravity="0 0 0"/>
    <worldbody><body><joint name="j" ref=".4"/><geom type="sphere" size=".1" mass="1"/></body>
    <body pos="1 0 0"><joint name="k" ref="-.3"/><geom type="sphere" size=".1" mass="1"/></body></worldbody>
    <actuator>{actuators}</actuator></mujoco>"""
    spec = mujoco.MjSpec.from_string(xml)
    return AuditContext(model=spec.compile(), spec=spec, xml_text=xml)


@pytest.mark.parametrize("gear", [1, 2, -2])
def test_pid_and_following_position_servo_hold_actual_length(gear):
    context = ctx(f'<pid joint="j" kp="10" kv="2" gear="{gear}"/><position joint="k" kp="10"/>')
    assert context.model.actuator_ctrladr.tolist() == [0, 2]
    result = a03_servo_hold(context)
    assert result["status"] == "PASS"
    assert result["max_angular_drift_rad"] == pytest.approx(0, abs=1e-12)


def test_position_servo_hold_respects_gear():
    result = a03_servo_hold(ctx('<position joint="j" kp="10" gear="2"/>'))
    assert result["status"] == "PASS"
    assert result["max_angular_drift_rad"] == pytest.approx(0, abs=1e-12)


def test_motor_is_not_classified_as_position_servo():
    result = a03_servo_hold(ctx('<motor joint="j"/>'))
    assert result["detail"]["note"] == "no_position_servos"


def test_general_position_feedback_normalizes_control_gain():
    result = a03_servo_hold(
        ctx('<general joint="j" gainprm="5" biastype="affine" biasprm="0 -10 -2"/>')
    )
    assert result["status"] == "PASS"
    assert result["max_angular_drift_rad"] == pytest.approx(0, abs=1e-12)
