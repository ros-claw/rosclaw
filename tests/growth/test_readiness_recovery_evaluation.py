from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.growth.readiness_recovery_evaluation import (
    _episode_passed,
    evaluate_g1_readiness_recovery,
)


def test_readiness_recovery_episode_contract_rejects_motion_or_saturation() -> None:
    config = {
        "minimum_pelvis_height_m": 0.65,
        "maximum_peak_tilt_rad": 0.65,
        "maximum_final_speed_mps": 0.20,
        "maximum_final_joint_velocity_rms_rad_s": 0.50,
    }
    result = {
        "readiness_abstained": True,
        "finite_state": True,
        "post_abstention_fall": False,
        "joint_limit_violation": False,
        "torque_limit_violation": False,
        "actuator_saturation_steps": 0,
        "recovery_min_pelvis_height_m": 0.72,
        "recovery_peak_tilt_rad": 0.30,
        "final_pelvis_height_m": 0.78,
        "final_speed_mps": 0.08,
        "final_joint_velocity_rms_rad_s": 0.10,
    }

    assert _episode_passed(result, config)
    assert not _episode_passed({**result, "final_speed_mps": 0.201}, config)
    assert not _episode_passed({**result, "actuator_saturation_steps": 1}, config)
    assert not _episode_passed({**result, "readiness_abstained": False}, config)


def test_readiness_recovery_evaluation_requires_three_episodes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least three"):
        evaluate_g1_readiness_recovery(
            evidence_paths=(tmp_path / "one.json", tmp_path / "two.json"),
            router_path=tmp_path / "router.json",
            gate_path=tmp_path / "gate.json",
            output_path=tmp_path.parent / "report.json",
            source_checkout=tmp_path,
        )
