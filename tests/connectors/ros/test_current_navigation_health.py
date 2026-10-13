"""Current ROS subcheck contracts; no physical or motion acceptance."""

import hashlib
from datetime import timedelta

import pytest

from rosclaw.connectors.ros.context.navigation_readiness import (
    review_current_sim_navigation_health,
)
from rosclaw.contracts.common import content_hash
from tests.connectors.ros.test_navigation_body_workspace import NOW
from tests.connectors.ros.test_navigation_readiness import setup
from tests.connectors.ros.test_sim_source_binding import pack


def prepared(tmp_path):
    parts, policy = setup(tmp_path)
    root, original, _ = pack(parts)
    request = {
        "schema_version": "rosclaw.current_sim_navigation_health_request.v1",
        **{
            key: original[key]
            for key in ("manifest_hash", "body_snapshot_hash", "sim_run_id", "max_age_ms")
        },
    }
    return root, request, policy, parts[-1]


def review(parts, now=NOW):
    root, request, policy, model = parts
    model.seal()
    return review_current_sim_navigation_health(
        root,
        request,
        model=model,
        policy=policy,
        policy_hash=content_hash("navreadinesspolicy", policy),
        now=now,
    )


def test_healthy_observation_does_not_reuse_stopped_capture_or_grant_authority(tmp_path):
    parts = prepared(tmp_path)
    root = parts[0]
    before = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()
    }
    result = review(parts)
    assert result["status"] == "HEALTHY_SIM_OBSERVATION_ONLY"
    assert result["source_binding"] == result["daemon_generation"] == "NOT_EVALUATED"
    assert result["execution_authority"] == "NONE" and result["L2"] == "NOT_RUN"
    assert before == {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()
    }


def test_historical_good_health_cannot_cover_current_stale_ros(tmp_path):
    parts = prepared(tmp_path)
    assert review(parts)["status"] == "HEALTHY_SIM_OBSERVATION_ONLY"
    result = review(parts, NOW + timedelta(seconds=10))
    assert result["status"] != "HEALTHY_SIM_OBSERVATION_ONLY"
    assert "STALE_L1_CAPTURE" in result["unknown_reasons"]


@pytest.mark.parametrize("field", ["body_snapshot_hash", "manifest_hash", "sim_run_id"])
def test_immutable_context_mismatch_is_never_healthy(tmp_path, field):
    parts = prepared(tmp_path)
    parts[1][field] = "wrong"
    assert review(parts)["status"] != "HEALTHY_SIM_OBSERVATION_ONLY"


def test_tiny_negative_variance_is_still_rejected_without_threshold_change(tmp_path):
    parts = prepared(tmp_path)
    q = parts[-1].observations["localization_quality"][0]
    q["covariance"][0] = q["x_variance"] = -1e-18
    result = review(parts)
    assert "UNSAFE_PLANAR_LOCALIZATION_COVARIANCE" in result["blocked_reasons"]


def test_actual_effective_footprint_must_still_contain_body(tmp_path):
    parts = prepared(tmp_path)
    footprint = next(iter(parts[-1].observations["costmap_footprints"].values()))
    footprint["points"] = [[-0.01, -0.01, 0], [0.01, -0.01, 0], [0.01, 0.01, 0], [-0.01, 0.01, 0]]
    result = review(parts)
    assert any(
        r.startswith("EFFECTIVE_FOOTPRINT_UNDERSIZES_BODY:") for r in result["blocked_reasons"]
    )


def test_health_request_cannot_accept_caller_supplied_ready_or_generation(tmp_path):
    parts = prepared(tmp_path)
    parts[1]["READY"] = True
    with pytest.raises(ValueError, match="closed independent"):
        review(parts)
