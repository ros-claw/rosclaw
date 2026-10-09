"""Synthetic renamed streams exercise discovery without selecting a holdout."""

import ast
import hashlib
import time
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw.connectors.ros.context.discovery import discover_body_candidate
from rosclaw.connectors.ros.intelligence import RosSystemModel

NOW = datetime(2026, 10, 8, tzinfo=UTC)


def fixture(namespace="/sample", base="chassis", sensor="laser_mount"):
    data = (
        f'<robot name="anonymous"><link name="{base}"><collision><geometry>'
        '<box size="0.4 0.2 0.1"/></geometry></collision></link>'
        f'<link name="{sensor}"/><joint name="sensor_mount" type="fixed">'
        f'<parent link="{base}"/><child link="{sensor}"/></joint></robot>'
    ).encode()
    roles = [
        (namespace + "/position_feedback", "nav_msgs/msg/Odometry", "local", base),
        (namespace + "/floor", "nav_msgs/msg/OccupancyGrid", "world", None),
        (namespace + "/scan2", "sensor_msgs/msg/LaserScan", sensor, None),
    ]
    node = namespace + "/description_source"
    model = RosSystemModel(
        robot_id="unknown",
        snapshot_id="",
        captured_at=NOW,
        graph={
            "topics": [{"name": n, "msg_type": t} for n, t, _, _ in roles],
            "actions": [
                {"name": namespace + "/go", "action_type": "nav2_msgs/action/NavigateToPose"}
            ],
        },
        signals=[
            {
                "topic": n,
                "source": "native",
                "captured_at": NOW,
                "publisher_count": 1,
                "last_message_age_ms": 50,
            }
            for n, _, _, _ in roles
        ],
        transforms=[
            {"parent": a, "child": b, "source": "/tf", "captured_at": NOW, "age_ms": 50}
            for a, b in [("world", "local"), ("local", base), (base, sensor)]
        ],
        observations={
            "message_frames": {
                n: {
                    "source": n,
                    "captured_at": NOW.isoformat(),
                    "frame_id": f,
                    "child_frame_id": c,
                    "stamp_sec": 5,
                }
                for n, _, f, c in roles
            },
            "urdf_descriptions": {
                node: {
                    "source": node + "/get_parameters",
                    "captured_at": NOW.isoformat(),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                    "complete": True,
                }
            },
        },
        completeness={"graph": True, "tf": True},
    ).seal()
    return model, data


@pytest.mark.parametrize(
    "namespace,base,sensor",
    [
        ("/sample", "chassis", "laser_mount"),
        ("/other/robot", "renamed_body", "range_frame"),
    ],
)
def test_namespace_sensor_and_frame_rename_need_no_profile(namespace, base, sensor):
    model, data = fixture(namespace, base, sensor)
    before = model.to_dict()
    result = discover_body_candidate(model, data, now=NOW)
    assert result["status"] == "PROPOSED"
    assert result["frames"]["base"] == base
    assert result["frames"]["lidar"] == sensor
    assert result["geometry"]["complete"]
    assert result["unknown_fields"] == []
    assert result["interfaces"]["sensing.lidar"][0]["name"] == namespace + "/scan2"
    assert not result["authorization"] and not result["binding_verified"]
    assert result["capabilities_granted"] == [] and result["cleaning_attachment"] is None
    assert result["physical_acceptance_level"] == "NOT_RUN"
    assert model.to_dict() == before


@pytest.mark.parametrize("age", [None, 90000, -10000])
def test_unrelated_unknown_stale_or_future_tf_does_not_replace_required_chain(age):
    model, raw = fixture()
    data = model.to_dict()
    data["transforms"].append(
        {
            "parent": "unrelated_world",
            "child": "independent_model_truth",
            "source": "/unrelated_truth",
            "captured_at": (NOW - timedelta(hours=1)).isoformat(),
            "age_ms": age,
        }
    )
    model = RosSystemModel(**data).seal()
    before = model.to_dict()
    result = discover_body_candidate(model, raw, now=NOW)
    assert result["status"] == "PROPOSED" and result["unknown_fields"] == []
    assert not result["authorization"] and not result["binding_verified"]
    assert result["capabilities_granted"] == [] and model.to_dict() == before


@pytest.mark.parametrize("edge_index", [0, 1, 2])
@pytest.mark.parametrize("fault", ["unknown_age", "stale_age", "future_age", "old_receipt"])
def test_every_required_tf_edge_retains_its_freshness_guard(edge_index, fault):
    model, raw = fixture()
    data = model.to_dict()
    edge = data["transforms"][edge_index]
    if fault == "old_receipt":
        edge["captured_at"] = (NOW - timedelta(seconds=6)).isoformat()
    else:
        edge["age_ms"] = {"unknown_age": None, "stale_age": 1001, "future_age": -101}[fault]
    result = discover_body_candidate(RosSystemModel(**data).seal(), raw, now=NOW)
    assert result["status"] == "UNKNOWN"
    assert any(key.startswith("tf.") for key in result["unknown_fields"])
    assert not result["authorization"] and not result["binding_verified"]


@pytest.mark.parametrize("fault", ["conflicting_parent", "duplicate_edge", "cycle", "self_loop"])
def test_required_tf_chain_conflicts_and_cycles_refused(fault):
    model, raw = fixture()
    data = model.to_dict()
    if fault in {"conflicting_parent", "duplicate_edge"}:
        extra = dict(data["transforms"][1])
        if fault == "conflicting_parent":
            extra["parent"] = "foreign_odom"
        data["transforms"].append(extra)
    else:
        data["transforms"][1]["parent"] = "laser_mount" if fault == "cycle" else "chassis"
    result = discover_body_candidate(RosSystemModel(**data).seal(), raw, now=NOW)
    assert result["status"] == "UNKNOWN"
    assert "tf.odom_to_base" in result["unknown_fields"]
    assert not result["authorization"] and not result["binding_verified"]


@pytest.mark.parametrize(
    "fault",
    [
        "stale_graph",
        "stale_signal",
        "stale_header",
        "missing_header",
        "wrong_source",
        "stale_tf",
        "unknown_tf",
        "missing_tf",
        "wrong_urdf",
        "stale_description",
        "wrong_description_source",
        "ambiguous_description",
        "ambiguous_sensor",
        "untyped_navigation",
    ],
)
def test_missing_ambiguous_or_stale_provenance_never_promotes(fault):
    model, data = fixture()
    frames = model.observations["message_frames"]
    descriptions = model.observations["urdf_descriptions"]
    description = next(iter(descriptions.values()))
    if fault == "stale_graph":
        model.captured_at = NOW - timedelta(seconds=6)
    elif fault == "stale_signal":
        model.signals[0].last_message_age_ms = 5000
    elif fault == "stale_header":
        frames["/sample/scan2"]["captured_at"] = (NOW - timedelta(seconds=6)).isoformat()
    elif fault == "missing_header":
        frames.clear()
    elif fault == "wrong_source":
        frames["/sample/scan2"]["source"] = "/another/sensor"
    elif fault == "stale_tf":
        model.transforms[0].age_ms = 1500
    elif fault == "unknown_tf":
        model.completeness["tf"] = False
    elif fault == "missing_tf":
        model.transforms.pop()
    elif fault == "wrong_urdf":
        description["sha256"] = "0" * 64
    elif fault == "stale_description":
        description["captured_at"] = (NOW - timedelta(seconds=6)).isoformat()
    elif fault == "wrong_description_source":
        description["source"] = "/somewhere/get_parameters"
    elif fault == "ambiguous_description":
        descriptions["/another"] = {**description, "source": "/another/get_parameters"}
    elif fault == "ambiguous_sensor":
        model.graph["topics"].append({"name": "/extra", "msg_type": "sensor_msgs/msg/LaserScan"})
        model.signals.append(model.signals[-1].model_copy(update={"topic": "/extra"}))
        frames["/extra"] = {**frames["/sample/scan2"], "source": "/extra"}
    else:
        model.graph["actions"][0]["action_type"] = "unknown/Action"
    result = discover_body_candidate(model.seal(), data, now=NOW)
    assert result["status"] == "UNKNOWN" and result["unknown_fields"]
    assert result["capabilities_granted"] == [] and not result["binding_verified"]


def test_tampered_snapshot_is_rejected_before_discovery():
    model, data = fixture()
    model.robot_id = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        discover_body_candidate(model, data, now=NOW)


def test_probe_captures_actual_message_frames_and_original_receive_time():
    path = Path(__file__).resolve().parents[3] / "integrations/ros_probe/ros2/probe.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ReadOnlyProbe")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "observe")
    scope = {"time": time, "deque": deque, "utc_now": lambda: NOW.isoformat()}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    probe = SimpleNamespace(samples={}, message_frames={})
    message = SimpleNamespace(
        header=SimpleNamespace(frame_id="local", stamp=SimpleNamespace(sec=7, nanosec=500000000)),
        child_frame_id="chassis",
    )
    scope["observe"](probe, "/renamed/feedback", message)
    assert probe.message_frames["/renamed/feedback"] == {
        "source": "/renamed/feedback",
        "frame_id": "local",
        "child_frame_id": "chassis",
        "stamp_sec": 7.5,
        "captured_at": NOW.isoformat(),
    }
