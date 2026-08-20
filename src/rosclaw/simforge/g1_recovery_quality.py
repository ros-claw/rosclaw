"""Trajectory-derived post-kick stability metrics and matched A/B gates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class G1RecoveryQuality:
    contact_time_sec: float
    evaluation_start_sec: float
    evaluation_duration_sec: float
    tail_window_sec: float
    torso_angular_velocity_rms_rad_s: float
    torso_tilt_rms_rad: float
    com_lateral_velocity_rms_m_s: float
    pelvis_planar_velocity_rms_m_s: float
    joint_velocity_rms_rad_s: float
    post_contact_joint_acceleration_rms_rad_s2: float
    post_contact_joint_jerk_rms_rad_s3: float
    post_contact_leg_joint_jerk_rms_rad_s3: float
    post_contact_waist_joint_jerk_rms_rad_s3: float
    post_contact_arm_joint_jerk_rms_rad_s3: float
    tail_torso_angular_velocity_rms_rad_s: float
    tail_torso_tilt_rms_rad: float
    tail_com_lateral_velocity_rms_m_s: float
    tail_pelvis_planar_velocity_rms_m_s: float
    tail_joint_velocity_rms_rad_s: float
    tail_joint_acceleration_rms_rad_s2: float
    tail_joint_jerk_rms_rad_s3: float
    tail_wobble_index: float
    post_contact_pelvis_path_length_m: float
    post_contact_pelvis_displacement_m: float
    post_contact_pelvis_max_excursion_m: float
    post_contact_forward_peak_advance_m: float
    post_contact_backward_reversal_m: float
    post_contact_lateral_peak_return_m: float
    post_contact_support_transition_count: int
    post_contact_single_support_duration_sec: float
    bilateral_support_fraction: float
    terminal_bilateral_support: bool
    terminal_stable_duration_sec: float
    settling_time_sec: float | None
    schema_version: str = "rosclaw.g1_goalforge.recovery_quality.v4"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class G1RecoveryComparison:
    passed: bool
    goal_outcome_preserved: bool
    safety_preserved: bool
    terminal_bilateral_support: bool
    tail_angular_velocity_reduction: float
    tail_tilt_reduction: float
    tail_pelvis_velocity_reduction: float
    tail_joint_velocity_reduction: float
    tail_wobble_reduction: float
    reasons: tuple[str, ...]
    schema_version: str = "rosclaw.g1_goalforge.recovery_comparison.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


@dataclass(frozen=True)
class G1MomentumUnloadingComparison:
    """Fail-closed promotion result for a shorter active recovery step."""

    passed: bool
    goal_outcome_preserved: bool
    safety_preserved: bool
    strict_replay_preserved: bool
    pelvis_path_reduction: float
    pelvis_displacement_reduction: float
    pelvis_max_excursion_reduction: float
    tail_wobble_reduction: float
    settling_time_reduction: float
    support_transition_reduction: float
    reasons: tuple[str, ...]
    schema_version: str = "rosclaw.g1_goalforge.momentum_unloading_comparison.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


@dataclass(frozen=True)
class G1NaturalnessComparison:
    """Fail-closed A/B gate for post-contact whole-body motion quality."""

    passed: bool
    goal_outcome_preserved: bool
    safety_preserved: bool
    strict_replay_preserved: bool
    joint_acceleration_reduction: float
    joint_jerk_reduction: float
    leg_joint_jerk_reduction: float
    waist_joint_jerk_reduction: float
    arm_joint_jerk_reduction: float
    tail_joint_jerk_reduction: float
    pelvis_path_regression: float
    pelvis_displacement_regression: float
    tail_wobble_reduction: float
    backward_reversal_reduction: float
    lateral_peak_return_reduction: float
    settling_time_regression: float
    support_transition_regression: float
    reasons: tuple[str, ...]
    schema_version: str = "rosclaw.g1_goalforge.naturalness_comparison.v2"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


@dataclass(frozen=True)
class G1AbsoluteRecoveryThresholds:
    """Frozen absolute gates for one declared recovery evaluation profile."""

    maximum_backward_reversal_m: float = 0.25
    maximum_pelvis_path_m: float = 0.70
    maximum_settling_time_sec: float = 1.50
    minimum_terminal_stable_duration_sec: float = 0.50
    minimum_ball_speed_mps: float = 9.50
    maximum_target_error_m: float = 0.25
    maximum_support_foot_slip_m: float = 0.04
    schema_version: str = "rosclaw.g1_goalforge.absolute_recovery_thresholds.v1"

    def __post_init__(self) -> None:
        values = (
            self.maximum_backward_reversal_m,
            self.maximum_pelvis_path_m,
            self.maximum_settling_time_sec,
            self.minimum_terminal_stable_duration_sec,
            self.minimum_ball_speed_mps,
            self.maximum_target_error_m,
            self.maximum_support_foot_slip_m,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("absolute recovery thresholds must be finite and non-negative")
        if self.maximum_settling_time_sec <= 0.0 or self.minimum_ball_speed_mps <= 0.0:
            raise ValueError("settling-time and ball-speed thresholds must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class G1AbsoluteRecoveryGate:
    """Fail-closed absolute gate; relative improvement alone is insufficient."""

    passed: bool
    goal_quality_passed: bool
    physical_recovery_passed: bool
    safety_passed: bool
    strict_replay_passed: bool
    reasons: tuple[str, ...]
    thresholds: G1AbsoluteRecoveryThresholds
    schema_version: str = "rosclaw.g1_goalforge.absolute_recovery_gate.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        value["evidence_domain"] = "SIM_ONLY"
        value["hardware_authorized"] = False
        return value


def evaluate_g1_absolute_recovery_gate(
    *,
    quality: G1RecoveryQuality,
    result: Mapping[str, Any],
    strict_replay: bool,
    thresholds: G1AbsoluteRecoveryThresholds | None = None,
) -> G1AbsoluteRecoveryGate:
    """Require natural recovery in addition to a successful strong shot.

    Missing result fields, a missing settling time, and any non-finite metric
    fail closed.  This prevents a relative A/B improvement from promoting a
    candidate that remains objectively unstable.
    """

    thresholds = thresholds or G1AbsoluteRecoveryThresholds()
    if not isinstance(quality, G1RecoveryQuality):
        raise ValueError("quality must be a G1RecoveryQuality")
    if not isinstance(thresholds, G1AbsoluteRecoveryThresholds):
        raise ValueError("thresholds must be G1AbsoluteRecoveryThresholds")

    ball_speed = _finite_result_metric(result, "ball_speed_mps")
    target_error = _finite_result_metric(result, "target_error_m")
    support_slip = _finite_result_metric(result, "support_foot_slip_m")
    recovery_metrics_finite = all(
        math.isfinite(value)
        for value in (
            quality.post_contact_backward_reversal_m,
            quality.post_contact_pelvis_path_length_m,
            quality.terminal_stable_duration_sec,
        )
    ) and (quality.settling_time_sec is None or math.isfinite(quality.settling_time_sec))
    goal_quality = bool(
        result.get("success") is True
        and result.get("goal_crossed") is True
        and result.get("target_zone_hit") is True
        and ball_speed is not None
        and ball_speed >= thresholds.minimum_ball_speed_mps - 1e-9
        and target_error is not None
        and target_error <= thresholds.maximum_target_error_m + 1e-9
    )
    physical_recovery = bool(
        recovery_metrics_finite
        and quality.post_contact_backward_reversal_m
        <= thresholds.maximum_backward_reversal_m + 1e-9
        and quality.post_contact_pelvis_path_length_m <= thresholds.maximum_pelvis_path_m + 1e-9
        and quality.settling_time_sec is not None
        and quality.settling_time_sec <= thresholds.maximum_settling_time_sec + 1e-9
        and quality.terminal_stable_duration_sec
        >= thresholds.minimum_terminal_stable_duration_sec - 1e-9
        and quality.terminal_bilateral_support
    )
    safety = bool(
        result.get("post_kick_fall") is False
        and result.get("joint_limit_violation") is False
        and result.get("torque_limit_violation") is False
        and result.get("actuator_saturation") is False
        and support_slip is not None
        and support_slip <= thresholds.maximum_support_foot_slip_m + 1e-9
    )
    reasons: list[str] = []
    if not goal_quality:
        reasons.append("goal_quality_below_absolute_gate")
    if not recovery_metrics_finite:
        reasons.append("recovery_metric_non_finite")
    if quality.post_contact_backward_reversal_m > thresholds.maximum_backward_reversal_m + 1e-9:
        reasons.append("backward_reversal_above_absolute_gate")
    if quality.post_contact_pelvis_path_length_m > thresholds.maximum_pelvis_path_m + 1e-9:
        reasons.append("pelvis_path_above_absolute_gate")
    if quality.settling_time_sec is None:
        reasons.append("settling_time_missing")
    elif quality.settling_time_sec > thresholds.maximum_settling_time_sec + 1e-9:
        reasons.append("settling_time_above_absolute_gate")
    if (
        quality.terminal_stable_duration_sec
        < thresholds.minimum_terminal_stable_duration_sec - 1e-9
    ):
        reasons.append("terminal_stable_duration_below_gate")
    if not quality.terminal_bilateral_support:
        reasons.append("terminal_bilateral_support_missing")
    if not safety:
        reasons.append("safety_below_absolute_gate")
    if not strict_replay:
        reasons.append("strict_replay_missing")
    return G1AbsoluteRecoveryGate(
        passed=not reasons,
        goal_quality_passed=goal_quality,
        physical_recovery_passed=physical_recovery,
        safety_passed=safety,
        strict_replay_passed=bool(strict_replay),
        reasons=tuple(reasons),
        thresholds=thresholds,
    )


def measure_g1_recovery_quality(
    trajectory: Mapping[str, np.ndarray],
    *,
    evaluation_delay_sec: float = 1.0,
    tail_window_sec: float = 2.0,
    stable_dwell_sec: float = 0.5,
) -> G1RecoveryQuality:
    """Measure motion after contact instead of reusing the loose fall gate.

    The wobble index is a deterministic comparison score, not a physical unit:
    angular velocity + planar pelvis velocity + half torso tilt + one quarter
    whole-body joint velocity.  Its components are always reported separately.
    """

    if evaluation_delay_sec < 0.0 or tail_window_sec <= 0.0 or stable_dwell_sec <= 0.0:
        raise ValueError("recovery metric windows must be positive")
    required = {
        "time",
        "torso_quaternion",
        "pelvis_pose",
        "joint_velocity",
        "com_y_relative",
        "left_foot_contact",
        "right_foot_contact",
        "contact_impulse",
        "ball_velocity",
    }
    missing = required.difference(trajectory)
    if missing:
        raise ValueError(f"recovery trajectory is missing fields: {sorted(missing)}")
    time = np.asarray(trajectory["time"], dtype=np.float64)
    quaternion = np.asarray(trajectory["torso_quaternion"], dtype=np.float64)
    pelvis = np.asarray(trajectory["pelvis_pose"], dtype=np.float64)
    joint_velocity = np.asarray(trajectory["joint_velocity"], dtype=np.float64)
    com_y = np.asarray(trajectory["com_y_relative"], dtype=np.float64)
    left_support = np.asarray(trajectory["left_foot_contact"], dtype=bool)
    right_support = np.asarray(trajectory["right_foot_contact"], dtype=bool)
    impulse = np.asarray(trajectory["contact_impulse"], dtype=np.float64)
    ball_velocity = np.asarray(trajectory["ball_velocity"], dtype=np.float64)
    count = len(time)
    expected = {
        "torso_quaternion": (count, 4),
        "pelvis_pose": (count, 7),
        "joint_velocity": (count, 29),
        "com_y_relative": (count,),
        "left_foot_contact": (count,),
        "right_foot_contact": (count,),
        "contact_impulse": (count,),
        "ball_velocity": (count, 3),
    }
    actual_shapes = {
        "torso_quaternion": quaternion.shape,
        "pelvis_pose": pelvis.shape,
        "joint_velocity": joint_velocity.shape,
        "com_y_relative": com_y.shape,
        "left_foot_contact": left_support.shape,
        "right_foot_contact": right_support.shape,
        "contact_impulse": impulse.shape,
        "ball_velocity": ball_velocity.shape,
    }
    invalid = [name for name, shape in expected.items() if actual_shapes[name] != shape]
    if count < 3 or time.ndim != 1 or invalid:
        raise ValueError(f"recovery trajectory shapes are invalid: {invalid}")
    if not np.all(np.diff(time) > 0.0):
        raise ValueError("recovery trajectory time must be strictly increasing")
    if not all(
        np.all(np.isfinite(value))
        for value in (time, quaternion, pelvis, joint_velocity, com_y, impulse, ball_velocity)
    ):
        raise ValueError("recovery trajectory must contain only finite values")
    contact_indices = np.flatnonzero(np.diff(impulse, prepend=0.0) > 1e-9)
    if not len(contact_indices):
        raise ValueError("recovery trajectory does not contain a ball-contact impulse")
    contact_index = int(contact_indices[0])
    contact_time = float(time[contact_index])
    evaluation_start = min(float(time[-1]), contact_time + evaluation_delay_sec)
    evaluation = time >= evaluation_start - 1e-12
    tail_start = max(evaluation_start, float(time[-1]) - tail_window_sec)
    tail = time >= tail_start - 1e-12
    if not np.any(evaluation) or not np.any(tail):
        raise ValueError("recovery trajectory is shorter than its metric windows")

    roll, pitch = _roll_pitch(quaternion)
    roll = np.unwrap(roll)
    pitch = np.unwrap(pitch)
    roll_rate = np.gradient(roll, time)
    pitch_rate = np.gradient(pitch, time)
    angular_speed = np.hypot(roll_rate, pitch_rate)
    tilt = np.hypot(roll, pitch)
    com_velocity = np.gradient(com_y, time)
    pelvis_velocity = np.linalg.norm(
        np.gradient(pelvis[:, :2], axis=0) / np.gradient(time)[:, None],
        axis=1,
    )
    joint_speed = np.sqrt(np.mean(np.square(joint_velocity), axis=1))
    joint_acceleration = np.gradient(joint_velocity, time, axis=0)
    joint_jerk = np.gradient(joint_acceleration, time, axis=0)
    bilateral = left_support & right_support
    support_state = left_support.astype(np.uint8) + 2 * right_support.astype(np.uint8)
    post_xy = pelvis[contact_index:, :2]
    post_excursion = np.linalg.norm(post_xy - post_xy[0], axis=1)
    post_relative_xy = post_xy - post_xy[0]
    forward_axis = _planar_unit_axis(ball_velocity[contact_index, :2])
    lateral_axis = np.asarray((-forward_axis[1], forward_axis[0]), dtype=np.float64)
    forward_progress = post_relative_xy @ forward_axis
    lateral_progress = post_relative_xy @ lateral_axis
    forward_peak = float(np.max(forward_progress))
    backward_reversal = max(0.0, forward_peak - float(forward_progress[-1]))
    lateral_peak_return = max(
        float(np.max(lateral_progress) - lateral_progress[-1]),
        float(lateral_progress[-1] - np.min(lateral_progress)),
    )
    post_path = float(np.linalg.norm(np.diff(post_xy, axis=0), axis=1).sum())
    post_displacement = float(np.linalg.norm(post_xy[-1] - post_xy[0]))
    post_support = support_state[contact_index:]
    post_single_support = np.logical_xor(
        left_support[contact_index:], right_support[contact_index:]
    )
    post_dt = np.diff(time[contact_index:])

    tail_angular = _rms(angular_speed[tail])
    tail_tilt = _rms(tilt[tail])
    tail_com_velocity = _rms(com_velocity[tail])
    tail_pelvis_velocity = _rms(pelvis_velocity[tail])
    tail_joint_velocity = _rms(joint_speed[tail])
    wobble = tail_angular + tail_pelvis_velocity + 0.50 * tail_tilt + 0.25 * tail_joint_velocity
    stable = (
        (angular_speed <= 0.35) & (tilt <= 0.35) & (pelvis_velocity <= 0.25) & (joint_speed <= 0.55)
    )
    first_evaluation_index = int(np.flatnonzero(evaluation)[0])
    final_stable_start = len(time)
    for index in range(len(time) - 1, first_evaluation_index - 1, -1):
        if not stable[index]:
            break
        final_stable_start = index
    terminal_duration = (
        float(time[-1] - time[final_stable_start]) if final_stable_start < len(time) else 0.0
    )
    settling_time = (
        max(0.0, float(time[final_stable_start] - contact_time))
        if terminal_duration + 1e-12 >= stable_dwell_sec
        else None
    )
    return G1RecoveryQuality(
        contact_time_sec=contact_time,
        evaluation_start_sec=evaluation_start,
        evaluation_duration_sec=float(time[-1] - evaluation_start),
        tail_window_sec=float(time[-1] - tail_start),
        torso_angular_velocity_rms_rad_s=_rms(angular_speed[evaluation]),
        torso_tilt_rms_rad=_rms(tilt[evaluation]),
        com_lateral_velocity_rms_m_s=_rms(com_velocity[evaluation]),
        pelvis_planar_velocity_rms_m_s=_rms(pelvis_velocity[evaluation]),
        joint_velocity_rms_rad_s=_rms(joint_speed[evaluation]),
        post_contact_joint_acceleration_rms_rad_s2=_rms(joint_acceleration[evaluation]),
        post_contact_joint_jerk_rms_rad_s3=_rms(joint_jerk[evaluation]),
        post_contact_leg_joint_jerk_rms_rad_s3=_rms(joint_jerk[evaluation, :12]),
        post_contact_waist_joint_jerk_rms_rad_s3=_rms(joint_jerk[evaluation, 12:15]),
        post_contact_arm_joint_jerk_rms_rad_s3=_rms(joint_jerk[evaluation, 15:]),
        tail_torso_angular_velocity_rms_rad_s=tail_angular,
        tail_torso_tilt_rms_rad=tail_tilt,
        tail_com_lateral_velocity_rms_m_s=tail_com_velocity,
        tail_pelvis_planar_velocity_rms_m_s=tail_pelvis_velocity,
        tail_joint_velocity_rms_rad_s=tail_joint_velocity,
        tail_joint_acceleration_rms_rad_s2=_rms(joint_acceleration[tail]),
        tail_joint_jerk_rms_rad_s3=_rms(joint_jerk[tail]),
        tail_wobble_index=wobble,
        post_contact_pelvis_path_length_m=post_path,
        post_contact_pelvis_displacement_m=post_displacement,
        post_contact_pelvis_max_excursion_m=float(np.max(post_excursion)),
        post_contact_forward_peak_advance_m=forward_peak,
        post_contact_backward_reversal_m=backward_reversal,
        post_contact_lateral_peak_return_m=lateral_peak_return,
        post_contact_support_transition_count=int(np.count_nonzero(np.diff(post_support))),
        post_contact_single_support_duration_sec=float(np.sum(post_dt * post_single_support[:-1])),
        bilateral_support_fraction=float(np.mean(bilateral[evaluation])),
        terminal_bilateral_support=bool(bilateral[-1]),
        terminal_stable_duration_sec=terminal_duration,
        settling_time_sec=settling_time,
    )


def compare_g1_momentum_unloading(
    *,
    parent: G1RecoveryQuality,
    candidate: G1RecoveryQuality,
    parent_result: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
    parent_strict_replay: bool,
    candidate_strict_replay: bool,
    maximum_target_error_m: float = 0.18,
    minimum_path_reduction: float = 0.40,
    minimum_displacement_reduction: float = 0.50,
    minimum_max_excursion_reduction: float = 0.50,
    minimum_wobble_reduction: float = 0.25,
    minimum_settling_time_reduction: float = 0.10,
    minimum_support_transition_reduction: float = 0.05,
) -> G1MomentumUnloadingComparison:
    """Compare an evolved recovery action with its retained parent.

    Unlike :func:`compare_g1_recovery`, this gate intentionally allows a
    bounded precision trade within the declared target zone.  It requires
    equal-or-better ball speed and large, measured reductions in wandering,
    wobble, settling time, and support chatter.  Both sides must strictly
    replay, so a noisy candidate cannot promote itself.
    """

    thresholds = (
        maximum_target_error_m,
        minimum_path_reduction,
        minimum_displacement_reduction,
        minimum_max_excursion_reduction,
        minimum_wobble_reduction,
        minimum_settling_time_reduction,
        minimum_support_transition_reduction,
    )
    if not all(math.isfinite(value) for value in thresholds):
        raise ValueError("momentum-unloading thresholds must be finite")
    if maximum_target_error_m <= 0.0 or any(not 0.0 <= value < 1.0 for value in thresholds[1:]):
        raise ValueError("momentum-unloading thresholds are outside their bounds")

    goal_preserved = bool(
        candidate_result.get("success")
        and candidate_result.get("goal_crossed")
        and candidate_result.get("target_zone_hit")
        and float(candidate_result.get("target_error_m", math.inf)) <= maximum_target_error_m
        and float(candidate_result.get("ball_speed_mps", 0.0))
        >= float(parent_result.get("ball_speed_mps", 0.0)) - 1e-9
    )
    safety_preserved = bool(
        not candidate_result.get("post_kick_fall")
        and not candidate_result.get("joint_limit_violation")
        and not candidate_result.get("torque_limit_violation")
        and not candidate_result.get("actuator_saturation")
        and float(candidate_result.get("support_foot_slip_m", math.inf)) <= 0.04
        and candidate.terminal_bilateral_support
    )
    strict_replay = bool(parent_strict_replay and candidate_strict_replay)
    path_reduction = _reduction(
        parent.post_contact_pelvis_path_length_m,
        candidate.post_contact_pelvis_path_length_m,
    )
    displacement_reduction = _reduction(
        parent.post_contact_pelvis_displacement_m,
        candidate.post_contact_pelvis_displacement_m,
    )
    excursion_reduction = _reduction(
        parent.post_contact_pelvis_max_excursion_m,
        candidate.post_contact_pelvis_max_excursion_m,
    )
    wobble_reduction = _reduction(parent.tail_wobble_index, candidate.tail_wobble_index)
    settling_reduction = _optional_time_reduction(
        parent.settling_time_sec, candidate.settling_time_sec
    )
    transition_reduction = _reduction(
        float(parent.post_contact_support_transition_count),
        float(candidate.post_contact_support_transition_count),
    )
    reasons = []
    if not goal_preserved:
        reasons.append("goal_or_precision_envelope_regressed")
    if not safety_preserved:
        reasons.append("safety_regressed")
    if not strict_replay:
        reasons.append("strict_replay_missing")
    for reduction, minimum, reason in (
        (path_reduction, minimum_path_reduction, "pelvis_path_reduction_below_gate"),
        (
            displacement_reduction,
            minimum_displacement_reduction,
            "pelvis_displacement_reduction_below_gate",
        ),
        (
            excursion_reduction,
            minimum_max_excursion_reduction,
            "pelvis_max_excursion_reduction_below_gate",
        ),
        (wobble_reduction, minimum_wobble_reduction, "tail_wobble_reduction_below_gate"),
        (
            settling_reduction,
            minimum_settling_time_reduction,
            "settling_time_reduction_below_gate",
        ),
        (
            transition_reduction,
            minimum_support_transition_reduction,
            "support_transition_reduction_below_gate",
        ),
    ):
        if reduction < minimum:
            reasons.append(reason)
    return G1MomentumUnloadingComparison(
        passed=not reasons,
        goal_outcome_preserved=goal_preserved,
        safety_preserved=safety_preserved,
        strict_replay_preserved=strict_replay,
        pelvis_path_reduction=path_reduction,
        pelvis_displacement_reduction=displacement_reduction,
        pelvis_max_excursion_reduction=excursion_reduction,
        tail_wobble_reduction=wobble_reduction,
        settling_time_reduction=settling_reduction,
        support_transition_reduction=transition_reduction,
        reasons=tuple(reasons),
    )


def compare_g1_naturalness(
    *,
    parent: G1RecoveryQuality,
    candidate: G1RecoveryQuality,
    parent_result: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
    parent_strict_replay: bool,
    candidate_strict_replay: bool,
    minimum_joint_acceleration_reduction: float = 0.01,
    minimum_joint_jerk_reduction: float = 0.01,
    minimum_leg_jerk_reduction: float = 0.0,
    minimum_waist_jerk_reduction: float = 0.02,
    minimum_arm_jerk_reduction: float = 0.05,
    minimum_tail_jerk_reduction: float = 0.10,
    minimum_tail_wobble_reduction: float = 0.05,
    minimum_backward_reversal_reduction: float = 0.10,
    minimum_lateral_peak_return_reduction: float = 0.15,
    maximum_path_regression: float = 0.02,
    maximum_displacement_regression: float = 0.05,
    maximum_settling_time_regression: float = 0.02,
) -> G1NaturalnessComparison:
    """Promote smooth follow-through only when stability remains intact.

    Jerk is derived from recorded physical joint velocity and starts one
    second after ball contact, so impact itself cannot dominate the score.
    The immediate retained parent must be beaten on tail wobble, backward
    reversal, and lateral peak return.  Path, settling time, and support
    chatter retain fail-closed non-regression bounds.
    """

    thresholds = (
        minimum_joint_acceleration_reduction,
        minimum_joint_jerk_reduction,
        minimum_leg_jerk_reduction,
        minimum_waist_jerk_reduction,
        minimum_arm_jerk_reduction,
        minimum_tail_jerk_reduction,
        minimum_tail_wobble_reduction,
        minimum_backward_reversal_reduction,
        minimum_lateral_peak_return_reduction,
        maximum_path_regression,
        maximum_displacement_regression,
        maximum_settling_time_regression,
    )
    if not all(math.isfinite(value) and 0.0 <= value < 1.0 for value in thresholds):
        raise ValueError("naturalness thresholds must be finite and in [0, 1)")

    goal_preserved = bool(
        candidate_result.get("success")
        and candidate_result.get("goal_crossed") == parent_result.get("goal_crossed")
        and candidate_result.get("target_zone_hit") == parent_result.get("target_zone_hit")
        and float(candidate_result.get("target_error_m", math.inf))
        <= float(parent_result.get("target_error_m", math.inf)) + 1e-9
        and float(candidate_result.get("ball_speed_mps", 0.0))
        >= float(parent_result.get("ball_speed_mps", 0.0)) - 1e-9
    )
    safety_preserved = bool(
        not candidate_result.get("post_kick_fall")
        and not candidate_result.get("joint_limit_violation")
        and not candidate_result.get("torque_limit_violation")
        and not candidate_result.get("actuator_saturation")
        and float(candidate_result.get("support_foot_slip_m", math.inf))
        <= max(0.04, float(parent_result.get("support_foot_slip_m", 0.0)) + 1e-9)
        and candidate.terminal_bilateral_support
    )
    strict_replay = bool(parent_strict_replay and candidate_strict_replay)
    acceleration_reduction = _reduction(
        parent.post_contact_joint_acceleration_rms_rad_s2,
        candidate.post_contact_joint_acceleration_rms_rad_s2,
    )
    jerk_reduction = _reduction(
        parent.post_contact_joint_jerk_rms_rad_s3,
        candidate.post_contact_joint_jerk_rms_rad_s3,
    )
    leg_reduction = _reduction(
        parent.post_contact_leg_joint_jerk_rms_rad_s3,
        candidate.post_contact_leg_joint_jerk_rms_rad_s3,
    )
    waist_reduction = _reduction(
        parent.post_contact_waist_joint_jerk_rms_rad_s3,
        candidate.post_contact_waist_joint_jerk_rms_rad_s3,
    )
    arm_reduction = _reduction(
        parent.post_contact_arm_joint_jerk_rms_rad_s3,
        candidate.post_contact_arm_joint_jerk_rms_rad_s3,
    )
    tail_reduction = _reduction(
        parent.tail_joint_jerk_rms_rad_s3,
        candidate.tail_joint_jerk_rms_rad_s3,
    )
    path_regression = _regression(
        parent.post_contact_pelvis_path_length_m,
        candidate.post_contact_pelvis_path_length_m,
    )
    displacement_regression = _regression(
        parent.post_contact_pelvis_displacement_m,
        candidate.post_contact_pelvis_displacement_m,
    )
    wobble_reduction = _reduction(parent.tail_wobble_index, candidate.tail_wobble_index)
    backward_reversal_reduction = _reduction(
        parent.post_contact_backward_reversal_m,
        candidate.post_contact_backward_reversal_m,
    )
    lateral_peak_return_reduction = _reduction(
        parent.post_contact_lateral_peak_return_m,
        candidate.post_contact_lateral_peak_return_m,
    )
    settling_regression = _optional_time_regression(
        parent.settling_time_sec,
        candidate.settling_time_sec,
    )
    transition_regression = _regression(
        float(parent.post_contact_support_transition_count),
        float(candidate.post_contact_support_transition_count),
    )
    reasons = []
    if not goal_preserved:
        reasons.append("goal_outcome_regressed")
    if not safety_preserved:
        reasons.append("safety_regressed")
    if not strict_replay:
        reasons.append("strict_replay_missing")
    for actual, threshold, reason in (
        (
            acceleration_reduction,
            minimum_joint_acceleration_reduction,
            "joint_acceleration_reduction_below_gate",
        ),
        (jerk_reduction, minimum_joint_jerk_reduction, "joint_jerk_reduction_below_gate"),
        (leg_reduction, minimum_leg_jerk_reduction, "leg_jerk_regressed"),
        (
            waist_reduction,
            minimum_waist_jerk_reduction,
            "waist_jerk_reduction_below_gate",
        ),
        (arm_reduction, minimum_arm_jerk_reduction, "arm_jerk_reduction_below_gate"),
        (tail_reduction, minimum_tail_jerk_reduction, "tail_jerk_reduction_below_gate"),
        (
            wobble_reduction,
            minimum_tail_wobble_reduction,
            "tail_wobble_reduction_below_gate",
        ),
        (
            backward_reversal_reduction,
            minimum_backward_reversal_reduction,
            "backward_reversal_reduction_below_gate",
        ),
        (
            lateral_peak_return_reduction,
            minimum_lateral_peak_return_reduction,
            "lateral_peak_return_reduction_below_gate",
        ),
    ):
        if actual < threshold:
            reasons.append(reason)
    for actual, maximum, reason in (
        (path_regression, maximum_path_regression, "pelvis_path_regression_above_gate"),
        (
            displacement_regression,
            maximum_displacement_regression,
            "pelvis_displacement_regression_above_gate",
        ),
        (
            settling_regression,
            maximum_settling_time_regression,
            "settling_time_regression_above_gate",
        ),
        (transition_regression, 0.0, "support_transition_regressed"),
    ):
        if actual > maximum + 1e-12:
            reasons.append(reason)
    return G1NaturalnessComparison(
        passed=not reasons,
        goal_outcome_preserved=goal_preserved,
        safety_preserved=safety_preserved,
        strict_replay_preserved=strict_replay,
        joint_acceleration_reduction=acceleration_reduction,
        joint_jerk_reduction=jerk_reduction,
        leg_joint_jerk_reduction=leg_reduction,
        waist_joint_jerk_reduction=waist_reduction,
        arm_joint_jerk_reduction=arm_reduction,
        tail_joint_jerk_reduction=tail_reduction,
        pelvis_path_regression=path_regression,
        pelvis_displacement_regression=displacement_regression,
        tail_wobble_reduction=wobble_reduction,
        backward_reversal_reduction=backward_reversal_reduction,
        lateral_peak_return_reduction=lateral_peak_return_reduction,
        settling_time_regression=settling_regression,
        support_transition_regression=transition_regression,
        reasons=tuple(reasons),
    )


def compare_g1_recovery(
    *,
    baseline: G1RecoveryQuality,
    candidate: G1RecoveryQuality,
    baseline_result: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
    minimum_wobble_reduction: float = 0.10,
) -> G1RecoveryComparison:
    if not 0.0 <= minimum_wobble_reduction < 1.0:
        raise ValueError("minimum_wobble_reduction must be in [0, 1)")
    goal_preserved = bool(
        candidate_result.get("success")
        and candidate_result.get("goal_crossed") == baseline_result.get("goal_crossed")
        and candidate_result.get("target_zone_hit") == baseline_result.get("target_zone_hit")
        and float(candidate_result.get("target_error_m", math.inf))
        <= float(baseline_result.get("target_error_m", math.inf)) + 1e-9
        and float(candidate_result.get("ball_speed_mps", 0.0))
        >= float(baseline_result.get("ball_speed_mps", 0.0)) - 1e-9
    )
    safety_preserved = bool(
        not candidate_result.get("post_kick_fall")
        and not candidate_result.get("joint_limit_violation")
        and not candidate_result.get("torque_limit_violation")
        and float(candidate_result.get("support_foot_slip_m", math.inf))
        <= max(0.08, float(baseline_result.get("support_foot_slip_m", 0.0)) + 0.005)
    )
    angular_reduction = _reduction(
        baseline.tail_torso_angular_velocity_rms_rad_s,
        candidate.tail_torso_angular_velocity_rms_rad_s,
    )
    tilt_reduction = _reduction(
        baseline.tail_torso_tilt_rms_rad,
        candidate.tail_torso_tilt_rms_rad,
    )
    pelvis_reduction = _reduction(
        baseline.tail_pelvis_planar_velocity_rms_m_s,
        candidate.tail_pelvis_planar_velocity_rms_m_s,
    )
    joint_reduction = _reduction(
        baseline.tail_joint_velocity_rms_rad_s,
        candidate.tail_joint_velocity_rms_rad_s,
    )
    wobble_reduction = _reduction(
        baseline.tail_wobble_index,
        candidate.tail_wobble_index,
    )
    reasons = []
    if not goal_preserved:
        reasons.append("goal_outcome_regressed")
    if not safety_preserved:
        reasons.append("safety_regressed")
    if not candidate.terminal_bilateral_support:
        reasons.append("terminal_bilateral_support_missing")
    if angular_reduction < 0.05:
        reasons.append("tail_angular_velocity_reduction_below_5pct")
    # Tilt is a posture term rather than an oscillation term.  A quiet,
    # bounded lean is acceptable; angular/pelvis/joint velocity must still
    # improve and the composite wobble gate remains mandatory.
    if candidate.tail_torso_tilt_rms_rad > 0.35:
        reasons.append("tail_tilt_exceeds_stable_bound")
    if pelvis_reduction < 0.05:
        reasons.append("tail_pelvis_velocity_reduction_below_5pct")
    if joint_reduction < 0.0:
        reasons.append("tail_joint_velocity_regressed")
    if wobble_reduction < minimum_wobble_reduction:
        reasons.append("tail_wobble_reduction_below_gate")
    return G1RecoveryComparison(
        passed=not reasons,
        goal_outcome_preserved=goal_preserved,
        safety_preserved=safety_preserved,
        terminal_bilateral_support=candidate.terminal_bilateral_support,
        tail_angular_velocity_reduction=angular_reduction,
        tail_tilt_reduction=tilt_reduction,
        tail_pelvis_velocity_reduction=pelvis_reduction,
        tail_joint_velocity_reduction=joint_reduction,
        tail_wobble_reduction=wobble_reduction,
        reasons=tuple(reasons),
    )


def _roll_pitch(quaternion_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, x, y, z = (quaternion_wxyz[:, index] for index in range(4))
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def _planar_unit_axis(value: np.ndarray) -> np.ndarray:
    """Normalize the observed ball travel direction into the field XY plane."""

    axis = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        raise ValueError("ball contact velocity has no finite planar travel direction")
    return axis / norm


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value))))


def _finite_result_metric(result: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(result[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _reduction(baseline: float, candidate: float) -> float:
    if baseline <= 1e-12:
        return 0.0 if candidate <= baseline + 1e-12 else -math.inf
    return (baseline - candidate) / baseline


def _regression(baseline: float, candidate: float) -> float:
    return -_reduction(baseline, candidate)


def _optional_time_reduction(baseline: float | None, candidate: float | None) -> float:
    if candidate is None:
        return -math.inf
    if baseline is None:
        return 1.0
    return _reduction(baseline, candidate)


def _optional_time_regression(baseline: float | None, candidate: float | None) -> float:
    if candidate is None:
        return math.inf
    if baseline is None:
        return -1.0
    return _regression(baseline, candidate)


__all__ = [
    "G1AbsoluteRecoveryGate",
    "G1AbsoluteRecoveryThresholds",
    "G1RecoveryComparison",
    "G1RecoveryQuality",
    "G1MomentumUnloadingComparison",
    "G1NaturalnessComparison",
    "compare_g1_momentum_unloading",
    "compare_g1_naturalness",
    "compare_g1_recovery",
    "evaluate_g1_absolute_recovery_gate",
    "measure_g1_recovery_quality",
]
