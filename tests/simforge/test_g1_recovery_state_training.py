from __future__ import annotations

from types import SimpleNamespace

import pytest

from rosclaw.simforge.g1_recovery_state_training import _absolute_recovery_margins


def _quality(**changes: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "post_contact_backward_reversal_m": 0.14,
        "tail_wobble_index": 0.20,
        "post_contact_leg_joint_jerk_rms_rad_s3": 800.0,
        "settling_time_sec": 5.0,
        "terminal_bilateral_support": True,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_absolute_recovery_margins_reward_a_bounded_rescue() -> None:
    margins, natural = _absolute_recovery_margins(_quality())

    assert natural
    assert margins == pytest.approx((0.60, 0.60, 0.20))


@pytest.mark.parametrize(
    ("changes", "negative_component"),
    [
        ({"post_contact_backward_reversal_m": 0.40}, 0),
        ({"tail_wobble_index": 0.60}, 1),
        ({"post_contact_leg_joint_jerk_rms_rad_s3": 1_100.0}, 2),
    ],
)
def test_absolute_recovery_margins_reject_excessive_motion(
    changes: dict[str, object],
    negative_component: int,
) -> None:
    margins, natural = _absolute_recovery_margins(_quality(**changes))

    assert not natural
    assert margins[negative_component] < 0.0


def test_absolute_recovery_margins_require_terminal_settling() -> None:
    _, natural = _absolute_recovery_margins(_quality(settling_time_sec=None))

    assert not natural
