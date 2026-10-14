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


def test_conservative_age_margin_keeps_actual_review_clock_and_original_limits(tmp_path):
    root, request, policy, model = prepared(tmp_path)
    model.seal()
    kwargs = {
        "model": model,
        "policy": policy,
        "policy_hash": content_hash("navreadinesspolicy", policy),
    }
    strict = review_current_sim_navigation_health(
        root, request, now=NOW, freshness_margin_ms=100, **kwargs
    )
    explicit = review_current_sim_navigation_health(
        root, request, now=NOW + timedelta(milliseconds=100), **kwargs
    )
    assert strict["reviewed_at"] == NOW.isoformat()
    assert strict["age_evaluation_at"] == explicit["reviewed_at"]
    assert strict["freshness_margin_ms"] == 100
    for field in explicit:
        if field != "reviewed_at":
            assert strict[field] == explicit[field]
    assert strict["policy_hash"] == content_hash("navreadinesspolicy", policy)


@pytest.mark.parametrize("margin", [-1, 101, True, 0.1])
def test_freshness_margin_is_bounded_and_cannot_relax_age(tmp_path, margin):
    root, request, policy, model = prepared(tmp_path)
    with pytest.raises(ValueError, match="conservative freshness margin"):
        review_current_sim_navigation_health(
            root,
            request,
            model=model,
            policy=policy,
            policy_hash=content_hash("navreadinesspolicy", policy),
            now=NOW,
            freshness_margin_ms=margin,
        )


@pytest.mark.parametrize(
    "topic",
    [
        "/sample/position_feedback",
        "/sample/scan2",
        "/sample/estimated_pose",
        "/sample/near_grid",
        "/sample/far_grid",
    ],
)
@pytest.mark.parametrize("message_age,elapsed", [(501, 0), (450, 51)])
def test_each_live_stream_uses_policy_and_elapsed_age(tmp_path, topic, message_age, elapsed):
    parts = prepared(tmp_path)
    parts[2]["max_age_ms"] = 500
    signal = next(s for s in parts[-1].signals if s.topic == topic)
    signal.last_message_age_ms = message_age
    result = review(parts, NOW + timedelta(milliseconds=elapsed))
    assert "CURRENT_LIVE_STREAM_AGE_EXCEEDED:" + topic in result["unknown_reasons"]
    assert result["status"] == "UNKNOWN"


@pytest.mark.parametrize("bad_age", [None, -1, float("nan"), float("inf")])
def test_missing_or_invalid_original_stream_age_fails_closed(tmp_path, bad_age):
    parts = prepared(tmp_path)
    signal = next(s for s in parts[-1].signals if s.topic == "/sample/scan2")
    signal.last_message_age_ms = bad_age
    result = review(parts)
    assert result["status"] == "UNKNOWN"
    assert any(
        r in result["unknown_reasons"]
        for r in (
            "CURRENT_LIVE_STREAM_AGE_EXCEEDED:/sample/scan2",
            "CURRENT_UNIQUE_LIVE_INTERFACE_REQUIRED:sensing.lidar",
        )
    )


@pytest.mark.parametrize("future_fact", ["snapshot", "signal", "localization", "tf", "parameter"])
def test_safety_horizon_cannot_hide_future_observations(tmp_path, future_fact):
    root, request, policy, model = prepared(tmp_path)
    future = NOW + timedelta(milliseconds=50)
    if future_fact == "snapshot":
        model.captured_at = future
    elif future_fact == "signal":
        model.signals[0].captured_at = future
    elif future_fact == "localization":
        model.observations["localization_quality"][0]["captured_at"] = future.isoformat()
    elif future_fact == "tf":
        next(r for r in model.observations["transform_values"] if not r["static"])[
            "captured_at"
        ] = future.isoformat()
    else:
        node = next(iter(model.observations["parameter_captured_at"]))
        model.observations["parameter_captured_at"][node] = future.isoformat()
    model.seal()
    result = review_current_sim_navigation_health(
        root,
        request,
        model=model,
        policy=policy,
        policy_hash=content_hash("navreadinesspolicy", policy),
        now=NOW,
        freshness_margin_ms=100,
    )
    assert result["status"] == "UNKNOWN", result


def test_margin_tightens_stream_age_without_requiring_periodic_map_or_commands(tmp_path):
    root, request, policy, model = prepared(tmp_path)
    policy["max_age_ms"] = 500
    for signal in model.signals:
        if signal.topic in {"/sample/floor", "/sample/unfiltered", "/sample/filtered"}:
            signal.last_message_age_ms = 900
    scan = next(s for s in model.signals if s.topic == "/sample/scan2")
    scan.last_message_age_ms = 401
    assert review((root, request, policy, model))["status"] == "HEALTHY_SIM_OBSERVATION_ONLY"
    result = review_current_sim_navigation_health(
        root,
        request,
        model=model,
        policy=policy,
        policy_hash=content_hash("navreadinesspolicy", policy),
        now=NOW,
        freshness_margin_ms=100,
    )
    assert "CURRENT_LIVE_STREAM_AGE_EXCEEDED:/sample/scan2" in result["unknown_reasons"]
