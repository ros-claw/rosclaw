"""Synthetic contracts, never Nav2 execution or unseen robot acceptance."""

from copy import deepcopy
from datetime import timedelta

import pytest

from rosclaw.connectors.ros.context.discovery import discover_body_candidate
from rosclaw.connectors.ros.context.navigation_review import review_navigation_candidate
from rosclaw.connectors.ros.intelligence.system_model import Lifecycle, Signal
from tests.connectors.ros.test_body_discovery_candidate import NOW, fixture


def navigation_fixture(namespace="/sample", base="chassis", sensor="laser_mount"):
    model, raw = fixture(namespace, base, sensor)
    nodes = {
        r: namespace + "/" + r
        for r in ("localization", "planner", "controller", "collision_monitor", "navigator")
    }
    requirements = {
        "interface_names": {"mapping.occupancy_map": namespace + "/floor"},
        "lifecycle_nodes": nodes,
        "costmap_topics": [namespace + "/near_grid", namespace + "/far_grid"],
        "localization_topic": namespace + "/estimated_pose",
        "velocity_input_topic": namespace + "/unfiltered",
        "velocity_output_topic": namespace + "/filtered",
        "velocity_consumer_node": namespace + "/drive_adapter",
    }
    model.environment.update(ros_generation="ros2", use_sim_time=True)
    model.completeness.update(time=True, lifecycle=True, qos=True, signals=True)
    model.observations.update(clock_advancing=True, clock_last_receive_age_ms=20)
    model.lifecycle = [
        Lifecycle(name=n, source=n + "/get_state", state="ACTIVE", captured_at=NOW)
        for n in nodes.values()
    ]
    model.observations["localization_quality"] = [
        {
            "source": requirements["localization_topic"],
            "captured_at": NOW.isoformat(),
            "x_variance": 0.01,
            "y_variance": 0.01,
            "yaw_variance": 0.01,
        }
    ]
    model.observations["node_parameters"] = {
        nodes["collision_monitor"]: {
            "use_sim_time": True,
            "observation_sources": ["range_source"],
            "range_source.topic": namespace + "/scan2",
            "polygons": ["stop_zone"],
            "cmd_vel_in_topic": requirements["velocity_input_topic"],
            "cmd_vel_out_topic": requirements["velocity_output_topic"],
        }
    }
    model.observations["parameter_captured_at"] = {nodes["collision_monitor"]: NOW.isoformat()}
    for name, kind, publisher, subscriber in [
        (
            requirements["localization_topic"],
            "geometry_msgs/msg/PoseWithCovarianceStamped",
            nodes["localization"],
            None,
        ),
        *[
            (n, "nav_msgs/msg/OccupancyGrid", nodes["planner"], None)
            for n in requirements["costmap_topics"]
        ],
        (
            requirements["velocity_input_topic"],
            "geometry_msgs/msg/Twist",
            nodes["controller"],
            nodes["collision_monitor"],
        ),
        (
            requirements["velocity_output_topic"],
            "geometry_msgs/msg/Twist",
            nodes["collision_monitor"],
            requirements["velocity_consumer_node"],
        ),
    ]:
        model.graph["topics"].append(
            {
                "name": name,
                "msg_type": kind,
                "publishers": [publisher],
                "subscribers": [subscriber] if subscriber else [],
            }
        )
        model.signals.append(
            Signal(
                topic=name,
                source="native",
                captured_at=NOW,
                publisher_count=1,
                last_message_age_ms=50,
            )
        )
        model.observations["message_frames"][name] = {
            "source": name,
            "captured_at": NOW.isoformat(),
            "frame_id": "world",
        }
    # Validate the synthetic model through its public schema, not list mutation.
    return type(model)(**model.model_dump()).seal(), raw, requirements


@pytest.mark.parametrize(
    "namespace,base,sensor",
    [("/sample", "chassis", "laser_mount"), ("/renamed/robot", "platform", "range_frame")],
)
def test_navigation_review_is_independent_of_cleaning_and_has_no_authority(namespace, base, sensor):
    model, raw, requirements = navigation_fixture(namespace, base, sensor)
    before = model.to_dict()
    result = review_navigation_candidate(model, raw, requirements, now=NOW)
    assert result["status"] == "PROPOSED" and result["unknown_fields"] == []
    assert result["candidate"]["frames"]["base"] == base
    assert result["candidate"]["cleaning_attachment"] is None
    assert not result["authorization"] and not result["binding_verified"]
    assert not result["usable_for_real_execution"] and result["capabilities_granted"] == []
    assert result["physical_acceptance_level"] == "NOT_RUN"
    assert model.to_dict() == before


def test_costmap_type_ambiguity_requires_explicit_map_selection():
    model, raw, requirements = navigation_fixture()
    assert discover_body_candidate(model, raw, now=NOW)["status"] == "UNKNOWN"
    selected = discover_body_candidate(
        model, raw, now=NOW, interface_names=requirements["interface_names"]
    )
    assert selected["status"] == "PROPOSED"
    requirements["interface_names"]["mapping.occupancy_map"] = "/missing/map"
    assert review_navigation_candidate(model, raw, requirements, now=NOW)["status"] == "UNKNOWN"


@pytest.mark.parametrize(
    "fault",
    [
        "clock_stopped",
        "clock_stale",
        "not_sim",
        "time_incomplete",
        "lifecycle_incomplete",
        "inactive",
        "lifecycle_stale",
        "lifecycle_wrong_source",
        "lifecycle_duplicate",
        "localization_stale",
        "localization_bad_covariance",
        "localization_negative_covariance",
        "localization_missing_axis",
        "localization_wrong_publisher",
        "localization_wrong_frame",
        "costmap_stale",
        "costmap_missing",
        "costmap_duplicate_publisher",
        "costmap_wrong_type",
        "monitor_parameters_stale",
        "monitor_relative_sensor",
        "monitor_sensor_mismatch",
        "monitor_input_mismatch",
        "monitor_no_polygon",
        "velocity_bypass",
        "velocity_wrong_publisher",
        "velocity_mixed_types",
        "velocity_duplicate_signal",
        "qos_incomplete",
        "qos_incompatible",
        "candidate_stale",
        "unknown_geometry",
    ],
)
def test_incomplete_or_faulty_evidence_never_proposes_navigation(fault):
    model, raw, req = navigation_fixture()
    monitor = req["lifecycle_nodes"]["collision_monitor"]
    params = model.observations["node_parameters"][monitor]
    quality = model.observations["localization_quality"][0]
    costmap = req["costmap_topics"][0]

    def signal(name):
        return next(s for s in model.signals if s.topic == name)

    def topic(name):
        return next(t for t in model.graph["topics"] if t["name"] == name)

    old = (NOW - timedelta(seconds=6)).isoformat()
    if fault == "clock_stopped":
        model.observations["clock_advancing"] = False
    elif fault == "clock_stale":
        model.observations["clock_last_receive_age_ms"] = 1001
    elif fault == "not_sim":
        model.environment["use_sim_time"] = False
    elif fault.endswith("_incomplete"):
        model.completeness[fault.removesuffix("_incomplete")] = False
    elif fault == "inactive":
        model.lifecycle[0].state = "INACTIVE"
    elif fault == "lifecycle_stale":
        model.lifecycle[0].captured_at = NOW - timedelta(seconds=6)
    elif fault == "lifecycle_wrong_source":
        model.lifecycle[0].source = "/invented/get_state"
    elif fault == "lifecycle_duplicate":
        model.lifecycle.append(model.lifecycle[0].model_copy())
    elif fault == "localization_stale":
        quality["captured_at"] = old
    elif fault == "localization_bad_covariance":
        quality["yaw_variance"] = 0.26
    elif fault == "localization_negative_covariance":
        quality["x_variance"] = -0.01
    elif fault == "localization_missing_axis":
        quality.pop("y_variance")
    elif fault == "localization_wrong_publisher":
        topic(req["localization_topic"])["publishers"] = ["/initial_pose_source"]
    elif fault == "localization_wrong_frame":
        model.observations["message_frames"][req["localization_topic"]]["frame_id"] = "local"
    elif fault == "costmap_stale":
        signal(costmap).last_message_age_ms = 1001
    elif fault == "costmap_missing":
        model.signals.remove(signal(costmap))
    elif fault == "costmap_duplicate_publisher":
        signal(costmap).publisher_count = 2
    elif fault == "costmap_wrong_type":
        topic(costmap)["msg_type"] = "std_msgs/msg/String"
    elif fault == "monitor_parameters_stale":
        model.observations["parameter_captured_at"][monitor] = old
    elif fault == "monitor_relative_sensor":
        params["range_source.topic"] = "scan2"
    elif fault == "monitor_sensor_mismatch":
        params["range_source.topic"] = "/other/scan"
    elif fault == "monitor_input_mismatch":
        params["cmd_vel_in_topic"] = "/other/input"
    elif fault == "monitor_no_polygon":
        params["polygons"] = []
    elif fault == "velocity_bypass":
        topic(req["velocity_input_topic"])["subscribers"].append(req["velocity_consumer_node"])
    elif fault == "velocity_wrong_publisher":
        topic(req["velocity_output_topic"])["publishers"] = [req["lifecycle_nodes"]["controller"]]
    elif fault == "velocity_mixed_types":
        topic(req["velocity_output_topic"])["msg_type"] = "geometry_msgs/msg/TwistStamped"
    elif fault == "velocity_duplicate_signal":
        model.signals.append(signal(req["velocity_input_topic"]).model_copy())
    elif fault == "qos_incompatible":
        model.qos["incompatible_pairs"] = [{"topic": costmap}]
    elif fault == "candidate_stale":
        model.captured_at = NOW - timedelta(seconds=6)
    elif fault == "unknown_geometry":
        raw = b'<robot name="x"><link name="chassis"/></robot>'
    else:
        raise AssertionError(fault)
    result = review_navigation_candidate(model.seal(), raw, req, now=NOW)
    assert result["status"] == "UNKNOWN" and result["unknown_fields"]
    assert not result["authorization"] and result["capabilities_granted"] == []


@pytest.mark.parametrize(
    "fault",
    [
        "extra_key",
        "missing_key",
        "duplicate_roles",
        "relative_name",
        "same_velocity_topic",
        "unknown_interface_role",
    ],
)
def test_malformed_requirements_rejected(fault):
    model, raw, req = navigation_fixture()
    req = deepcopy(req)
    if fault == "extra_key":
        req["approved"] = True
    elif fault == "missing_key":
        req.pop("localization_topic")
    elif fault == "duplicate_roles":
        req["lifecycle_nodes"]["planner"] = req["lifecycle_nodes"]["controller"]
    elif fault == "relative_name":
        req["costmap_topics"][0] = "near_grid"
    elif fault == "same_velocity_topic":
        req["velocity_output_topic"] = req["velocity_input_topic"]
    else:
        req["interface_names"]["cleaning"] = "/attachment"
    with pytest.raises(ValueError):
        review_navigation_candidate(model, raw, req, now=NOW)


def test_tampered_model_rejected():
    model, raw, req = navigation_fixture()
    model.environment["use_sim_time"] = False
    with pytest.raises(ValueError, match="hash"):
        review_navigation_candidate(model, raw, req, now=NOW)


def test_readonly_cli_reports_review_without_binding(tmp_path, monkeypatch, capsys):
    import json
    from datetime import datetime
    from types import SimpleNamespace

    from rosclaw.connectors.ros.cli.ros_cli import cmd_ros_expert
    from rosclaw.connectors.ros.context import navigation_review

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW

    monkeypatch.setattr(navigation_review, "datetime", FixedDatetime)
    model, raw, requirements = navigation_fixture()
    (tmp_path / "robot.urdf").write_bytes(raw)
    (tmp_path / "snapshot.json").write_text(json.dumps(model.to_dict()))
    (tmp_path / "requirements.json").write_text(json.dumps(requirements))
    args = SimpleNamespace(
        ros_command="review-navigation",
        urdf=str(tmp_path / "robot.urdf"),
        requirements=str(tmp_path / "requirements.json"),
        snapshot=str(tmp_path / "snapshot.json"),
        graph=None,
        native=None,
        body=None,
        json=True,
        output=str(tmp_path / "review.json"),
    )
    assert cmd_ros_expert(args) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "PROPOSED"
    assert response["authorization"] is False and response["binding_verified"] is False
    assert response["source_snapshot_hash"] == model.snapshot_hash
    assert json.loads((tmp_path / "review.json").read_text())["status"] == "PROPOSED"
