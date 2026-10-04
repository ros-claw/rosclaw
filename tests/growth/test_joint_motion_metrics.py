import numpy as np
import pytest

from rosclaw.growth.joint_motion_metrics import joint_motion_metrics


def inputs():
    return [
        np.zeros((8, 2)),
        np.zeros((8, 2)),
        np.zeros((8, 2)),
        np.zeros((8, 10, 2)),
        np.array([5.0, 10.0]),
    ]


def measure(values):
    return joint_motion_metrics(*values, sample_interval_s=0.02)


def test_stationary_all_joints_are_kept_and_inputs_not_mutated():
    values = inputs()
    copies = [v.copy() for v in values]
    result = measure(values)
    assert all(v == [0.0, 0.0] for v in result.values())
    assert all(np.array_equal(a, b) for a, b in zip(values, copies, strict=True))


def test_measured_velocity_and_command_tracking_are_distinct():
    values = inputs()
    values[0][:, 0] = np.arange(8) * 0.02
    values[1][:, 0] = 1
    values[2][:, 0] = values[0][:, 0] + 0.1
    result = measure(values)
    assert result["measured_velocity_rms_rad_s"] == [1.0, 0.0]
    assert result["target_velocity_rms_rad_s"] == pytest.approx([1.0, 0.0])
    assert result["target_tracking_error_rms_rad"] == pytest.approx([0.1, 0.0])
    assert result["measured_acceleration_rms_rad_s2"] == [0.0, 0.0]


def test_force_slew_keeps_actual_substep_resolution_and_frame_boundaries():
    values = inputs()
    values[3][:, :, 0] = np.arange(80).reshape(8, 10) * 0.002
    result = measure(values)
    assert result["actuator_force_slew_rms_nm_s"] == pytest.approx([1.0, 0.0])


def test_saturation_uses_joint_specific_limits_and_keeps_exceedance():
    values = inputs()
    values[3][:, :, 0] = -5
    values[3][:, :, 1] = 11
    result = measure(values)
    assert result["actuator_limit_99pct_fraction"] == [1.0, 1.0]
    assert result["actuator_limit_maximum_fraction"] == pytest.approx([1.0, 1.1])


@pytest.mark.parametrize("field", range(5))
def test_nonfinite_rejected(field):
    values = inputs()
    values[field].flat[0] = np.nan
    with pytest.raises(ValueError):
        measure(values)


def test_alignment_boolean_zero_limit_and_overflow_rejected():
    values = inputs()
    values[1] = values[1][:-1]
    with pytest.raises(ValueError):
        measure(values)
    values = inputs()
    values[2] = values[2].astype(bool)
    with pytest.raises(ValueError):
        measure(values)
    values = inputs()
    values[4][0] = 0
    with pytest.raises(ValueError):
        measure(values)
    values = inputs()
    values[1][:, 0] = np.where(np.arange(8) % 2, 1e308, -1e308)
    with pytest.raises(ValueError, match="finite"):
        measure(values)
