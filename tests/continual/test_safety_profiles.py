from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw.continual.safety_profiles import (
    GrowthSafetyObservation,
    GrowthSafetyProfile,
    GrowthSafetyUse,
    evaluate_growth_safety,
    validate_profile_pair,
)


def _profile(use: GrowthSafetyUse) -> GrowthSafetyProfile:
    exploration = use is GrowthSafetyUse.EXPLORATION_SIM
    return GrowthSafetyProfile(
        profile_id="g1.controlled-fall." + use.value.lower(),
        use=use,
        maximum_joint_limit_excess_rad=0.08 if exploration else 0.02,
        maximum_normalized_actuator_command=1.20 if exploration else 1.0,
        maximum_head_impact_speed_mps=0.20 if exploration else 0.10,
        maximum_root_angular_speed_rad_s=8.0 if exploration else 3.5,
        maximum_self_penetration_m=0.01 if exploration else 0.002,
        always_allowed_contacts=("left_foot", "right_foot"),
        hard_fail_contacts=("head", "neck"),
        phase_contact_permissions={
            "ready": (),
            "landing": ("hand", "forearm", "lateral_thigh", "side_torso")
            if exploration
            else ("hand", "forearm", "lateral_thigh"),
        },
    )


def _observation() -> GrowthSafetyObservation:
    return GrowthSafetyObservation(
        phase="landing",
        finite_state=True,
        joint_limit_excess_rad=0.0,
        normalized_actuator_command=0.9,
        head_impact_speed_mps=0.0,
        root_angular_speed_rad_s=3.0,
        self_penetration_m=0.0,
        contacts=("forearm", "lateral_thigh"),
    )


def test_exploration_can_pass_without_becoming_promotion_evidence() -> None:
    decision = evaluate_growth_safety(_profile(GrowthSafetyUse.EXPLORATION_SIM), _observation())

    assert decision.passed
    assert not decision.promotion_eligible


def test_promotion_is_tighter_and_can_be_eligible() -> None:
    exploration = _profile(GrowthSafetyUse.EXPLORATION_SIM)
    promotion = _profile(GrowthSafetyUse.PROMOTION_SIM)
    validate_profile_pair(exploration, promotion)

    decision = evaluate_growth_safety(promotion, _observation())

    assert decision.passed
    assert decision.promotion_eligible


def test_head_contact_and_non_finite_state_fail_in_exploration() -> None:
    profile = _profile(GrowthSafetyUse.EXPLORATION_SIM)

    head = evaluate_growth_safety(profile, replace(_observation(), contacts=("head",)))
    non_finite = evaluate_growth_safety(profile, replace(_observation(), finite_state=False))

    assert head.hard_failure and "HARD_CONTACT:head" in head.reasons
    assert non_finite.hard_failure and "NON_FINITE_STATE" in non_finite.reasons


def test_promotion_pair_rejects_a_looser_threshold() -> None:
    exploration = _profile(GrowthSafetyUse.EXPLORATION_SIM)
    promotion = replace(
        _profile(GrowthSafetyUse.PROMOTION_SIM),
        maximum_root_angular_speed_rad_s=9.0,
    )

    with pytest.raises(ValueError, match="looser"):
        validate_profile_pair(exploration, promotion)
