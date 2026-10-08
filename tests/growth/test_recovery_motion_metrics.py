import numpy as np
import pytest

from rosclaw.growth.recovery_motion_metrics import recovery_motion_metrics


def data():
    return [np.zeros((10, 3)), np.tile([0.0, 0.0, 0.0, 1.0], (10, 1)), np.zeros((10, 6))]


def measure(values, **options):
    return recovery_motion_metrics(
        *values, timestep_s=0.02, event_frame=2, forward_direction=[1, 0, 0], **options
    )


def test_stationary_motion_and_full_window():
    result = measure(data())
    assert result["post_event_duration_s"] == pytest.approx(0.14)
    assert all(v == 0 for k, v in result.items() if k != "post_event_duration_s")


def test_backward_then_forward_is_not_hidden_by_final_progress():
    values = data()
    values[0][2:, 0] = [0, -0.1, -0.2, -0.3, -0.2, 0, 0.1, 0.2]
    result = measure(values)
    assert result["maximum_retreat_m"] == pytest.approx(0.3)
    assert result["final_forward_displacement_m"] == pytest.approx(0.2)
    assert result["root_path_length_m"] == pytest.approx(0.8)


def test_sign_flips_do_not_create_false_angular_motion():
    values = data()
    first = measure(values)
    values[1][::2] *= -1
    assert measure(values) == first


def test_actual_rotation_and_linear_joint_motion_units():
    values = data()
    theta = np.arange(10) * 0.02
    values[1][:, 2] = np.sin(theta / 2)
    values[1][:, 3] = np.cos(theta / 2)
    values[2] = np.repeat(theta[:, None], 6, axis=1)
    result = measure(values)
    assert result["maximum_orientation_excursion_rad"] == pytest.approx(0.14)
    assert result["root_angular_speed_rms_rad_s"] == pytest.approx(1.0)
    assert result["joint_speed_rms_rad_s"] == pytest.approx(1.0)
    assert result["joint_jerk_rms_rad_s3"] < 1e-9


@pytest.mark.parametrize("field", range(3))
def test_nonfinite_measured_states_rejected(field):
    values = data()
    values[field][3, 0] = np.nan
    with pytest.raises(ValueError):
        measure(values)


def test_short_window_zero_direction_and_bad_quaternion_rejected():
    values = data()
    with pytest.raises(ValueError):
        recovery_motion_metrics(
            *values, timestep_s=0.02, event_frame=8, forward_direction=[1, 0, 0]
        )
    with pytest.raises(ValueError):
        recovery_motion_metrics(
            *values, timestep_s=0.02, event_frame=2, forward_direction=[0, 0, 0]
        )
    values[1] *= 2
    with pytest.raises(ValueError):
        measure(values)


def test_overflow_rejected_not_reported_as_good_motion():
    values = data()
    values[2][:, 0] = np.where(np.arange(10) % 2, 1e308, -1e308)
    with pytest.raises(ValueError, match="finite"):
        measure(values)
