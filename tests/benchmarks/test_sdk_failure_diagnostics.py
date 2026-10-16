import hashlib
import json

import pytest

from benchmarks.ros_expert_ablation.sdk_failure import failure_feedback, safe_failure_feedback


def test_recorded_continuous_geometry_failure_does_not_erase_initial_l1(tmp_path):
    world = tmp_path / "world"
    (world / "execution").mkdir(parents=True)
    (world / "actual-live-L1-readiness-review.json").write_text(
        json.dumps({"status": "READY_FOR_SIM_TEST", "source_binding": {"result": "MATCHED"}})
    )
    (world / "execution/governed-journey-original-outcome.json").write_text(
        json.dumps(
            {
                "goals": [
                    {
                        "action_id": "canonical-1",
                        "final_state": "FAILED",
                        "verification": {"physical_stop_verified": False},
                    }
                ]
            }
        )
    )
    reason = "EFFECTIVE_FOOTPRINT_UNDERSIZES_BODY:/robot/global_costmap/costmap"
    health = json.dumps({"health": {"blocked_reasons": [reason], "unknown_reasons": []}})
    log = world / "execution/governed-daemon-original.log"
    log.write_text(
        "current ROS/source expired or rejected: "
        + health
        + "\ncurrent ROS/source expired or rejected: "
        + health
        + "\n"
    )
    before = log.read_bytes()
    feedback = failure_feedback(world, root=tmp_path)
    assert (
        feedback["initial_L1"] == "READY_FOR_SIM_TEST"
        and feedback["initial_source_binding"] == "MATCHED"
    )
    assert feedback["failure_stage"] == "CANONICAL_EXECUTION_NOT_COMPLETED"
    assert feedback["recorded_runtime_health_blocked_reasons"] == [reason]
    assert feedback["canonical_goals"][0]["physical_stop_verified"] is False
    assert feedback["verified_success"] is False and feedback["current_authority"] == "NONE"
    assert (
        "TF time alignment" in feedback["next_check"]
        and feedback["safety_thresholds_changed"] is False
    )
    assert log.read_bytes() == before
    for ref in feedback["original_diagnostic_refs"]:
        assert hashlib.sha256((tmp_path / ref["path"]).read_bytes()).hexdigest() == ref["sha256"]


def test_absent_artifacts_remain_unknown(tmp_path):
    world = tmp_path / "world"
    world.mkdir()
    feedback = failure_feedback(world, root=tmp_path)
    assert feedback["initial_L1"] == "UNKNOWN_NOT_RECORDED"
    assert not feedback["canonical_goals"] and not feedback["original_diagnostic_refs"]
    assert feedback["verified_success"] is False


def test_operator_artifact_escape_and_symlink_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError):
        failure_feedback(outside, root=root)
    world = root / "world"
    world.mkdir()
    target = outside / "secret"
    target.write_text("{}")
    (world / "actual-live-L1-readiness-review.json").symlink_to(target)
    with pytest.raises(ValueError, match="bounded original"):
        failure_feedback(world, root=root)


def test_diagnostic_error_preserves_failed_sdk_verdict(tmp_path):
    world = tmp_path / "world"
    world.mkdir()
    (world / "actual-live-L1-readiness-review.json").write_text("invalid original JSON")
    feedback = safe_failure_feedback(world, root=tmp_path)
    assert feedback["failure_stage"] == "SDK_WORLD_FAILED_DIAGNOSTIC_UNAVAILABLE"
    assert (
        feedback["verified_success"] is False and feedback["initial_L1"] == "UNKNOWN_NOT_REVIEWED"
    )
