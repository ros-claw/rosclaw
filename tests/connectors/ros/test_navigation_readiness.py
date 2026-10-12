"""Synthetic L1 contracts, never held-out or live SIM acceptance."""

import json
from datetime import timedelta

import pytest

from rosclaw.connectors.ros.context.navigation_readiness import review_sim_navigation_readiness
from rosclaw.contracts.common import content_hash
from tests.connectors.ros.test_navigation_body_workspace import NOW
from tests.connectors.ros.test_sim_source_binding import fixture, pack


def setup(tmp_path):
    parts = fixture(tmp_path)
    root, _, _, capture, _, _, _, model = parts
    requirements = json.loads((root / "navigation-requirements.json").read_text())
    policy = {
        "schema_version": "rosclaw.sim_navigation_readiness_policy.v1",
        "policy_id": "synthetic-precision-policy",
        "rationale": "synthetic planar std < Body goal error; no physical claim",
        "max_age_ms": 1000,
        "variance_limits": {"x": 0.001, "y": 0.001, "yaw": 0.001},
        "costmaps": {topic: topic + "_node" for topic in requirements["costmap_topics"]},
        "footprint_topics": {
            topic: topic + "_footprint" for topic in requirements["costmap_topics"]
        },
    }
    quality = model.observations["localization_quality"][0]
    covariance = [0.0] * 36
    for key, index in (("x", 0), ("y", 7), ("yaw", 35)):
        covariance[index] = quality[key + "_variance"] = 0.0001
    quality["covariance"] = covariance
    model.observations["transform_values"] = [
        {
            "parent": t.parent,
            "child": t.child,
            "source": t.source,
            "captured_at": t.captured_at.isoformat(),
            "static": t.static,
            "stamp_sec": 5,
            "translation": [0, 0, 0],
            "rotation_xyzw": [0, 0, 0, 1],
        }
        for t in model.transforms
    ]
    footprints = model.observations["costmap_footprints"] = {}
    for topic, node in policy["costmaps"].items():
        next(t for t in model.graph["topics"] if t["name"] == topic)["publishers"] = [node]
        model.observations["node_parameters"][node] = {
            "use_sim_time": True,
            "robot_base_frame": "chassis",
            "global_frame": "world",
            "plugins": ["range"],
            "range.plugin": "nav2_costmap_2d::ObstacleLayer",
            "range.enabled": True,
            "range.observation_sources": "lidar",
            "range.lidar.topic": "/sample/scan2",
            "robot_radius": 0.3,
            "footprint": "[]",
        }
        model.observations["parameter_captured_at"][node] = NOW.isoformat()
        ft = policy["footprint_topics"][topic]
        model.graph["topics"].append(
            {
                "name": ft,
                "msg_type": "geometry_msgs/msg/PolygonStamped",
                "publishers": [node],
                "subscribers": [],
            }
        )
        footprints[ft] = {
            "source": ft,
            "captured_at": NOW.isoformat(),
            "frame_id": "world",
            "stamp_sec": 5,
            "points": [[-0.3, -0.3, 0], [0.3, -0.3, 0], [0.3, 0.3, 0], [-0.3, 0.3, 0]],
        }
        model.observations["message_frames"][ft] = {"stamp_age_ms": 50}
    capture["physics_witness"]["sample"].update(
        body_id="synthetic-observed-base",
        evidence_domain="GAZEBO_PHYSICS",
        collision_count=0,
        linear_speed_mps=0,
        angular_speed_rps=0,
    )
    model.seal()
    return parts, policy


def run(parts, policy, **overrides):
    parts[-1].seal()
    root, request, inputs = pack(parts)
    return review_sim_navigation_readiness(
        root,
        request,
        policy=policy,
        policy_hash=content_hash("navreadinesspolicy", policy),
        **{**inputs, **overrides},
    )


def test_ready_is_readonly_point_in_time_with_no_authority(tmp_path):
    parts, policy = setup(tmp_path)
    before = {
        str(p): (
            p.read_bytes() if p.is_file() else None,
            p.stat().st_mtime_ns,
            p.stat().st_ctime_ns,
        )
        for p in [parts[0], *parts[0].rglob("*")]
    }
    result = run(parts, policy)
    assert result["status"] == "READY_FOR_SIM_TEST", result
    assert result["execution_authority"] == "NONE" and result["real_authority"] is False
    assert result["L2"] == "NOT_RUN"
    assert before == {
        str(p): (
            p.read_bytes() if p.is_file() else None,
            p.stat().st_mtime_ns,
            p.stat().st_ctime_ns,
        )
        for p in [parts[0], *parts[0].rglob("*")]
    }
    assert run(parts, policy, now=NOW + timedelta(seconds=2))["status"] != "READY_FOR_SIM_TEST"


@pytest.mark.parametrize(
    "fault",
    [
        "covariance_missing",
        "negative_variance",
        "indefinite",
        "asymmetric",
        "summary_mismatch",
        "loose_policy",
        "stale_localization",
        "inactive",
        "clock",
        "tf",
        "scan",
        "qos",
        "duplicate_publisher",
        "undersized_circle",
        "undersized_polygon",
        "missing_footprint",
        "effective_small",
        "wrong_footprint_frame",
        "wrong_footprint_publisher",
        "missing_numeric_tf",
        "bad_quaternion",
        "stale_numeric_tf",
        "moving",
        "lease",
        "contact",
        "stale_parameters",
        "disabled_obstacle",
        "wrong_lidar",
        "wrong_base",
        "wrong_body",
        "source_pin",
    ],
)
def test_incomplete_or_unsafe_evidence_cannot_promote(tmp_path, fault):
    parts, policy = setup(tmp_path)
    model, capture = parts[-1], parts[3]
    quality = model.observations["localization_quality"][0]
    topic, node = next(iter(policy["costmaps"].items()))
    params = model.observations["node_parameters"][node]
    ft = policy["footprint_topics"][topic]
    if fault == "covariance_missing":
        quality.pop("covariance")
    elif fault == "negative_variance":
        quality["covariance"][0] = quality["x_variance"] = -0.001
    elif fault == "indefinite":
        quality["covariance"][1] = quality["covariance"][6] = 0.001
    elif fault == "asymmetric":
        quality["covariance"][1] = 0.00001
    elif fault == "summary_mismatch":
        quality["x_variance"] = 0.0002
    elif fault == "loose_policy":
        policy["variance_limits"]["x"] = 0.02
    elif fault == "stale_localization":
        quality["captured_at"] = (NOW - timedelta(seconds=10)).isoformat()
    elif fault == "inactive":
        model.lifecycle[0].state = "INACTIVE"
    elif fault == "clock":
        model.observations["clock_advancing"] = False
    elif fault == "tf":
        model.transforms.pop(0)
    elif fault == "scan":
        next(s for s in model.signals if s.topic == "/sample/scan2").last_message_age_ms = 10000
    elif fault == "qos":
        model.qos["incompatible_pairs"] = [{"topic": "/sample/scan2"}]
    elif fault == "duplicate_publisher":
        next(t for t in model.graph["topics"] if t["name"] == "/sample/filtered")[
            "publishers"
        ].append("/bypass")
    elif fault == "undersized_circle":
        params["robot_radius"] = 0.01
    elif fault == "undersized_polygon":
        params["footprint"] = "[[-0.1,-0.1],[0.1,-0.1],[0.1,0.1],[-0.1,0.1]]"
    elif fault == "missing_footprint":
        model.observations["costmap_footprints"].pop(ft)
    elif fault == "effective_small":
        model.observations["costmap_footprints"][ft]["points"] = [
            [x / 10, y / 10, z]
            for x, y, z in model.observations["costmap_footprints"][ft]["points"]
        ]
    elif fault == "wrong_footprint_frame":
        model.observations["costmap_footprints"][ft]["frame_id"] = "unobserved"
    elif fault == "wrong_footprint_publisher":
        next(t for t in model.graph["topics"] if t["name"] == ft)["publishers"] = ["/other"]
    elif fault == "missing_numeric_tf":
        model.observations.pop("transform_values")
    elif fault == "bad_quaternion":
        model.observations["transform_values"][0]["rotation_xyzw"] = [0, 0, 0, 2]
    elif fault == "stale_numeric_tf":
        model.observations["transform_values"][0]["stamp_sec"] = 1
    elif fault == "moving":
        capture["physics_witness"]["sample"]["linear_speed_mps"] = 0.01
    elif fault == "lease":
        capture["physics_witness"]["sample"]["lease_enabled"] = True
    elif fault == "contact":
        capture["physics_witness"]["sample"]["collision_count"] = 1
    elif fault == "stale_parameters":
        model.observations["parameter_captured_at"][node] = (
            NOW - timedelta(seconds=10)
        ).isoformat()
    elif fault == "disabled_obstacle":
        params["range.enabled"] = False
    elif fault == "wrong_lidar":
        params["range.lidar.topic"] = "/wrong"
    elif fault == "wrong_base":
        params["robot_base_frame"] = "invented"
    elif fault == "wrong_body":
        capture["physics_witness"]["body_hash"] = "f" * 64
    elif fault == "source_pin":
        parts[2]["asset_sha256"]["robot.urdf"] = "f" * 64
    result = run(parts, policy)
    assert result["status"] in {"UNKNOWN", "BLOCKED"}, result
    assert result["unknown_reasons"] or result["blocked_reasons"]
    assert result["execution_authority"] == "NONE"


def test_policy_pin_is_independent(tmp_path):
    parts, policy = setup(tmp_path)
    root, request, inputs = pack(parts)
    with pytest.raises(ValueError, match="operator readiness policy"):
        review_sim_navigation_readiness(root, request, policy=policy, policy_hash="wrong", **inputs)


def test_covariance_serialization_roundoff_is_not_sensor_uncertainty(tmp_path):
    parts, policy = setup(tmp_path)
    covariance = parts[-1].observations["localization_quality"][0]["covariance"]
    covariance[1] = -5.271526297516281e-6
    covariance[6] = -5.271526297516282e-6
    assert run(parts, policy)["status"] == "READY_FOR_SIM_TEST"


@pytest.mark.parametrize("offset,expected", [(0, "READY_FOR_SIM_TEST"), (0.2, "BLOCKED")])
def test_costmap_base_uses_original_fixed_transform(tmp_path, offset, expected):
    from rosclaw.connectors.ros.intelligence.system_model import Transform

    parts, policy = setup(tmp_path)
    model = parts[-1]
    edge = Transform(
        parent="chassis", child="nav_base", source="/tf_static", captured_at=NOW, static=True
    )
    model.transforms.append(edge)
    model.observations["transform_values"].append(
        {
            "parent": edge.parent,
            "child": edge.child,
            "source": edge.source,
            "captured_at": NOW.isoformat(),
            "static": True,
            "stamp_sec": 0,
            "translation": [offset, 0, 0.01],
            "rotation_xyzw": [0, 0, 0, 1],
        }
    )
    for node in policy["costmaps"].values():
        model.observations["node_parameters"][node]["robot_base_frame"] = "nav_base"
    assert run(parts, policy)["status"] == expected
