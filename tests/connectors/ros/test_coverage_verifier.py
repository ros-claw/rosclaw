"""Independent geometry checks, denominator invariance and false-proof refusal."""

import math

import pytest

from rosclaw.connectors.ros.verification.coverage import CleaningPose, CoverageVerifier
from rosclaw.connectors.ros.verification.mission import verify_mission


def grid():
    return {
        "width": 10,
        "height": 10,
        "resolution": 0.1,
        "accessible_cells": list(range(100)),
        "cleaning_polygon": [(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)],
    }


def cover(verifier):
    stamp = 0
    for row in range(10):
        columns = range(10) if row % 2 == 0 else reversed(range(10))
        for col in columns:
            verifier.observe(
                CleaningPose((col + 0.5) * 0.1, (row + 0.5) * 0.1, 0, stamp, True), frame_id="map"
            )
            stamp += 0.1


def test_full_sweep_independent_of_action_status():
    verifier = CoverageVerifier(**grid())
    cover(verifier)
    result = verifier.result()
    assert result["coverage_ratio"] == 1
    assert result["accessible_area_m2"] == pytest.approx(1)
    assert result["missed_regions"] == []
    assert result["mission_success"] is None


def test_temporary_obstacle_never_shrinks_denominator_and_can_be_recovered():
    verifier = CoverageVerifier(**grid())
    verifier.set_temporary_blocked(list(range(40, 60)))
    cover(verifier)
    blocked = verifier.result()
    assert blocked["coverage_ratio"] == 0.8
    assert blocked["accessible_area_m2"] == pytest.approx(1)
    assert blocked["missed_regions"][0]["reason"] == "TEMP_BLOCKED"
    verifier.set_temporary_blocked([])
    for row in [4, 5]:
        for col in range(10):
            stamp = 20 + row + col * 0.1
            verifier.observe(
                CleaningPose((col + 0.5) * 0.1, (row + 0.5) * 0.1, 0, stamp, True), frame_id="map"
            )
    assert verifier.result()["coverage_ratio"] == 1


def test_disabled_cleaning_and_trace_gaps_do_not_fill_space():
    verifier = CoverageVerifier(**grid())
    verifier.observe(CleaningPose(0.05, 0.05, 0, 0, False), frame_id="map")
    assert verifier.result()["coverage_ratio"] == 0
    verifier.observe(CleaningPose(0.05, 0.05, 0, 1, True), frame_id="map")
    verifier.observe(CleaningPose(0.95, 0.95, 0, 10, True), frame_id="map")
    result = verifier.result()
    assert result["coverage_ratio"] < 0.2
    assert result["trace_gaps"] == 1
    with pytest.raises(ValueError, match="strictly increase"):
        verifier.observe(CleaningPose(0, 0, 0, 10, True), frame_id="map")
    with pytest.raises(ValueError, match="frame mismatch"):
        verifier.observe(CleaningPose(0, 0, 0, 11, True), frame_id="odom")


def test_rotation_uses_cleaning_polygon_not_robot_width():
    verifier = CoverageVerifier(
        **{**grid(), "cleaning_polygon": [(-0.2, -0.05), (0.2, -0.05), (0.2, 0.05), (-0.2, 0.05)]}
    )
    verifier.observe(CleaningPose(0.5, 0.5, math.pi / 2, 0, True), frame_id="map")
    cleaned = [i for i, v in enumerate(verifier.result()["mask"]) if v == 1]
    assert len({i // 10 for i in cleaned}) > len({i % 10 for i in cleaned})


def test_forged_complete_trace_never_becomes_a_mission_pass():
    evidence = {
        "mission_id": "m1",
        "body_id": "b1",
        "frame_id": "map",
        "grid": grid(),
        "trajectory": [{"x": 0.5, "y": 0.5, "yaw": 0, "time_sec": 0, "cleaning_enabled": True}],
        "collision": {"collision_count": 0, "observation_complete": True},
    }
    evidence["grid"]["cleaning_polygon"] = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
    result = verify_mission(evidence)
    assert result["calculation_pass"] is True
    assert result["verification_status"] == "NOT_VERIFIED"
    assert result["success"] is False
    assert result["missing_evidence"]


@pytest.mark.parametrize("invalid", [0, float("nan"), float("inf")])
def test_invalid_grid_rejected(invalid):
    with pytest.raises(ValueError):
        CoverageVerifier(**{**grid(), "resolution": invalid})


def test_recovery_waits_for_obstacle_and_has_bounded_idempotent_attempts():
    from rosclaw.connectors.ros.mission.recovery import MissedRegionRecovery

    verifier = CoverageVerifier(**grid())
    verifier.set_temporary_blocked(list(range(100)))
    queue = MissedRegionRecovery(verifier, max_attempts=2)
    assert queue.propose()["status"] == "WAITING_FOR_OBSTACLE"
    assert not queue.propose()["ready"]
    verifier.set_temporary_blocked([])
    region = queue.propose()["ready"][0]
    queue.record_attempt(region["cells"], action_id="attempt_1")
    queue.record_attempt(region["cells"], action_id="attempt_1")
    assert queue.propose()["ready"][0]["retry_count"] == 1
    queue.record_attempt(region["cells"], action_id="attempt_2")
    assert queue.propose()["status"] == "REQUIRES_OPERATOR"
    assert verifier.result()["coverage_ratio"] == 0
    assert queue.propose()["dispatched"] is False


def test_stationary_sampling_does_not_inflate_overlap():
    verifier = CoverageVerifier(**grid())
    for step in range(20):
        verifier.observe(CleaningPose(0.5, 0.5, 0, step * 0.1, True), frame_id="map")
    assert verifier.result()["coverage_ratio"] > 0
    assert verifier.result()["overlap_ratio"] == 0


@pytest.mark.parametrize(
    "mode,domain,expected",
    [
        ("FIXTURE", "FIXTURE", "NOT_VERIFIED"),
        ("SHADOW", "SHADOW", "NOT_VERIFIED"),
        ("SIMULATION", "HARDWARE", "NOT_VERIFIED"),
        ("SIMULATION", "SIMULATION", "PASS"),
    ],
)
def test_mission_requires_exact_independent_binding_and_preserves_domain(mode, domain, expected):
    from rosclaw.contracts.common import content_hash

    evidence = {
        "mission_id": "m1",
        "body_id": "b1",
        "frame_id": "map",
        "grid": grid(),
        "trajectory": [{"x": 0.5, "y": 0.5, "yaw": 0, "time_sec": 0, "cleaning_enabled": True}],
        "collision": {"collision_count": 0, "observation_complete": True},
        "action_ids": ["a1"],
    }
    evidence["grid"]["cleaning_polygon"] = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
    frozen_hash = content_hash("rosmissionevidence", evidence)

    class TrustedDaemonFixture:
        def get_execution_receipt(self, action_id):
            return {
                "receipt": {
                    "action_id": action_id,
                    "body_id": "b1",
                    "final_state": "COMPLETED",
                    "mode": mode,
                    "evidence_domain": domain,
                    "evidence_level": "TASK_VERIFIED",
                    "verification_result": {
                        "mission_id": "m1",
                        "independent_evidence_hashes": [frozen_hash],
                    },
                }
            }

    daemon = TrustedDaemonFixture()
    result = verify_mission(evidence, daemon=daemon)
    assert result["verification_status"] == expected
    assert result["hardware_verified"] is False
    evidence["collision"]["collision_count"] = 1
    altered = verify_mission(evidence, daemon=daemon)
    assert altered["verification_status"] == "NOT_VERIFIED"
    assert altered["success"] is False
