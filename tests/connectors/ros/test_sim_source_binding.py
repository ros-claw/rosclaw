"""Synthetic source-binding contracts, never actual Gazebo/L1/L2 acceptance."""

import hashlib
import json
from copy import deepcopy
from datetime import timedelta
from xml.etree import ElementTree as ET

import pytest

from rosclaw.connectors.ros.context.navigation_review import review_navigation_candidate
from rosclaw.connectors.ros.context.navigation_workspace import compile_navigation_body_workspace
from rosclaw.connectors.ros.context.sim_source_binding import (
    review_sim_source_binding,
    sdf_inventory,
)
from tests.connectors.ros.test_navigation_body_workspace import NOW, source


def encoded(value):
    return json.dumps(value, allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def fixture(tmp_path):
    model, raw, requirements, policy = source()
    model.environment["domain_id"] = "101"
    model.observations["clock_time_sec"] = 0.3
    model.observations["clock_received_monotonic_ns"] = 600_000_000
    model.seal()
    policy["source_snapshot_hash"] = model.snapshot_hash
    root = tmp_path / "compiled"
    manifest = compile_navigation_body_workspace(
        root, model, raw, requirements, policy=policy, now=NOW
    )
    robot = ET.fromstring(raw).get("name")
    converted = f'<sdf version="1.9"><model name="{robot}"><link name="chassis"><collision name="base"><geometry><box><size>0.4 0.3 0.2</size></box></geometry></collision></link></model></sdf>'.encode()
    world = converted.replace(b"<model ", b'<world name="owned_world"><model ').replace(
        b"</sdf>", b"</world></sdf>"
    )
    assets = {"robot.urdf": sha(raw), "converted.sdf": sha(converted), "world.sdf": sha(world)}
    launch = {
        "schema_version": "rosclaw.sim_source_launch.v1",
        "world_id": "owned_world",
        "run_id": policy["witness_declaration"]["run_id"],
        "robot_entity": robot,
        "ros_domain": 101,
        "namespace": "/sample",
        "image_digest": "sha256:" + "a" * 64,
        "observer_library_sha256": "b" * 64,
        "physics_witness_identity": "/sample/independent_truth",
        "physics_witness_implementation_sha256": "d" * 64,
        "physics_world_frame_alignment": "IDENTITY",
        "geometry_pose_tolerance_m": 0.00001,
        "geometry_rotation_matrix_tolerance": 0.001,
        "asset_sha256": assets,
        "conversion": {
            "argv": ["gz", "sdf", "-p", "/assets/robot.urdf"],
            "exit_code": 0,
            "input_sha256": sha(raw),
            "output_sha256": sha(converted),
        },
    }
    container = {
        "Id": "c" * 64,
        "Image": launch["image_digest"],
        "State": {"Running": True, "Pid": 1234},
        "HostConfig": {"ReadonlyRootfs": True, "NetworkMode": "none"},
        "Mounts": [
            {"Destination": "/assets", "RW": False},
            {"Destination": "/observer", "RW": False},
        ],
    }
    capture = {
        "schema_version": "rosclaw.sim_source_capture.v1",
        **{
            key: launch[key]
            for key in ["world_id", "run_id", "robot_entity", "ros_domain", "namespace"]
        },
        "observed_at": NOW.isoformat(),
        "host_monotonic_ns": 700_000_000,
        "assets_before": deepcopy(assets),
        "assets_after": deepcopy(assets),
        "container_before": deepcopy(container),
        "container_after": deepcopy(container),
        "observer": {
            "library_sha256": launch["observer_library_sha256"],
            "server_count": 1,
            "alive": True,
        },
        "physics_witness": {
            "identity": launch["physics_witness_identity"],
            "alive": True,
            "container_id": container["Id"],
            "run_id": launch["run_id"],
            "body_hash": manifest["body_snapshot_hash"],
            "entity_id": 2,
            "world_entity_id": 1,
            "implementation_sha256": launch["physics_witness_implementation_sha256"],
            "sample": {
                "captured_at": NOW.isoformat(),
                "time_sec": 0.3,
                "observation_complete": True,
                "source_only_observer": True,
                "lease_enabled": False,
                "source_received_control_messages": 0,
                "source_published_velocity_messages": 0,
                "body_snapshot_hash": manifest["body_snapshot_hash"],
                "run_id": launch["run_id"],
                "frame_id": "world",
                "x": 0,
                "y": 0,
                "yaw": 0,
            },
        },
    }
    samples = []
    for index in range(3):
        samples.append(
            {
                "schema_version": "rosclaw.gazebo_ecm_source_sample.v1",
                "inventory_complete": True,
                "paused": False,
                "reset_epoch": 0,
                "observer_run_label": launch["run_id"],
                "sim_time_ns": (index + 1) * 100_000_000,
                "iterations": index + 1,
                "observer_monotonic_ns": (index + 1) * 200_000_000,
                "sample": index,
                "worlds": [{"name": launch["world_id"], "entity": 1}],
                "models": [{"name": robot, "entity": 2, "world_pose": [0] * 6, "static": False}],
                "collisions": [
                    {
                        "entity": 4,
                        "name": "base",
                        "pose": [0] * 6,
                        "world_pose": [0] * 6,
                        "geometry_sdf": "<geometry><box><size>0.4 0.3 0.2</size></box></geometry>",
                        "ancestry": [
                            {"entity": 3, "name": "chassis", "model": False, "world": False},
                            {"entity": 2, "name": robot, "model": True, "world": False},
                            {
                                "entity": 1,
                                "name": launch["world_id"],
                                "model": False,
                                "world": True,
                            },
                        ],
                    }
                ],
                "sensors": [],
            }
        )
    request = {
        "schema_version": "rosclaw.sim_source_binding_request.v1",
        "manifest_hash": manifest["manifest_hash"],
        "body_snapshot_hash": manifest["body_snapshot_hash"],
        "source_asset_ref": "robot.urdf",
        "source_asset_digest": sha(raw),
        "sim_world_id": launch["world_id"],
        "sim_run_id": launch["run_id"],
        "expected_robot_entity": robot,
        "expected_ros_domain": 101,
        "expected_namespace": "/sample",
        "expected_frames": review_navigation_candidate(model, raw, requirements, now=NOW)[
            "candidate"
        ]["frames"],
        "max_age_ms": 1000,
    }
    return root, request, launch, capture, samples, converted, world, model


def pack(parts):
    root, request, launch, capture, samples, converted, world, model = parts
    ecm = b"\n".join(encoded(row) for row in samples) + b"\n"
    launch_raw = encoded(launch)
    capture.update(
        launch_sha256=sha(launch_raw), ecm_sha256=sha(ecm), ros_snapshot_hash=model.snapshot_hash
    )
    capture_raw = encoded(capture)
    request.update(launch_sha256=sha(launch_raw), capture_sha256=sha(capture_raw))
    return (
        root,
        request,
        {
            "launch_bytes": launch_raw,
            "capture_bytes": capture_raw,
            "converted_sdf_bytes": converted,
            "world_sdf_bytes": world,
            "ecm_bytes": ecm,
            "model": model,
            "now": NOW,
        },
    )


def test_pinned_complete_capture_matches_without_authority_or_writes(tmp_path):
    root, request, kwargs = pack(fixture(tmp_path))
    before = {
        str(p.relative_to(root)): (
            p.read_bytes() if p.is_file() else None,
            p.stat().st_mtime_ns,
            p.stat().st_ctime_ns,
        )
        for p in [root, *root.rglob("*")]
    }
    result = review_sim_source_binding(root, request, **kwargs)
    assert result["result"] == "MATCHED", result
    assert result["execution_authority"] == "NONE" and result["real_authority"] is False
    assert result["L1"] == result["L2"] == "NOT_RUN" and result["workspace_modified"] is False
    assert before == {
        str(p.relative_to(root)): (
            p.read_bytes() if p.is_file() else None,
            p.stat().st_mtime_ns,
            p.stat().st_ctime_ns,
        )
        for p in [root, *root.rglob("*")]
    }


@pytest.mark.parametrize(
    "fault",
    [
        "asset_replace",
        "collision_shrink",
        "collision_move",
        "collision_extra",
        "run",
        "world",
        "body",
        "manifest",
        "witness_name",
        "witness_exit",
        "witness_entity",
        "witness_body",
        "observer_exit",
        "observer_replace",
        "observer_multiple",
        "domain",
        "namespace",
        "duplicate_entity",
        "reset",
        "clock_backwards",
        "clock_freeze",
        "wall_clock_freeze",
        "paused",
        "incomplete",
        "entity_replace",
        "container_replace",
        "server_restart",
        "container_exit",
        "writable_source",
        "missing_mount",
        "unisolated",
        "stale",
        "future",
        "conversion_failure",
        "conversion_wrong_input",
        "duplicate_publisher",
        "native_wrong_domain",
        "missing_ecm_geometry",
        "unexpected_model",
        "effective_link_displacement",
        "witness_sample_stale",
        "witness_sample_body",
        "witness_sample_run",
        "witness_sample_frame",
        "witness_sample_incomplete",
        "witness_sample_time",
        "witness_sample_pose",
        "witness_sample_lease",
        "witness_sample_control",
        "witness_sample_velocity",
        "witness_implementation",
        "world_alignment",
        "observer_stale",
        "ros_clock_diverged",
    ],
)
def test_observed_faults_are_refused_even_when_capture_is_resealed(tmp_path, fault):
    parts = fixture(tmp_path)
    root, request, launch, capture, samples, converted, world, model = parts
    if fault == "asset_replace":
        capture["assets_after"]["robot.urdf"] = "bad"
    elif fault == "collision_shrink":
        samples[-1]["collisions"][0]["geometry_sdf"] = (
            "<geometry><box><size>0.2 0.3 0.2</size></box></geometry>"
        )
    elif fault == "collision_move":
        samples[-1]["collisions"][0]["pose"][0] = 1
    elif fault == "collision_extra":
        samples[-1]["collisions"].append(
            {**deepcopy(samples[-1]["collisions"][0]), "name": "extra", "entity": 99}
        )
    elif fault == "run":
        request["sim_run_id"] = "other"
    elif fault == "world":
        request["sim_world_id"] = "other"
    elif fault == "body":
        request["body_snapshot_hash"] = "wrong"
    elif fault == "manifest":
        request["manifest_hash"] = "wrong"
    elif fault == "witness_name":
        capture["physics_witness"]["identity"] = "renamed"
    elif fault == "witness_exit":
        capture["physics_witness"]["alive"] = False
    elif fault == "witness_entity":
        capture["physics_witness"]["entity_id"] = 99
    elif fault == "witness_body":
        capture["physics_witness"]["body_hash"] = "wrong"
    elif fault == "observer_exit":
        capture["observer"]["alive"] = False
    elif fault == "observer_replace":
        capture["observer"]["library_sha256"] = "wrong"
    elif fault == "observer_multiple":
        capture["observer"]["server_count"] = 2
    elif fault == "domain":
        request["expected_ros_domain"] = 102
    elif fault == "namespace":
        request["expected_namespace"] = "/other"
    elif fault == "duplicate_entity":
        samples[-1]["models"].append(deepcopy(samples[-1]["models"][0]))
    elif fault == "reset":
        samples[-1]["reset_epoch"] = 1
    elif fault == "clock_backwards":
        samples[-1]["sim_time_ns"] = 0
    elif fault == "clock_freeze":
        samples[-1]["sim_time_ns"] = samples[-2]["sim_time_ns"]
    elif fault == "wall_clock_freeze":
        samples[-1]["observer_monotonic_ns"] = samples[-2]["observer_monotonic_ns"]
    elif fault == "paused":
        samples[-1]["paused"] = True
    elif fault == "incomplete":
        samples[-1]["inventory_complete"] = False
    elif fault == "entity_replace":
        samples[-1]["models"][0]["entity"] = 22
    elif fault == "container_replace":
        capture["container_after"]["Id"] = "wrong"
    elif fault == "server_restart":
        capture["container_after"]["State"]["Pid"] += 1
    elif fault == "container_exit":
        capture["container_after"]["State"]["Running"] = False
    elif fault == "writable_source":
        capture["container_after"]["Mounts"][0]["RW"] = True
    elif fault == "missing_mount":
        capture["container_after"]["Mounts"][0]["Destination"] = "/extra"
    elif fault == "unisolated":
        capture["container_after"]["HostConfig"]["NetworkMode"] = "host"
    elif fault == "stale":
        capture["observed_at"] = (NOW - timedelta(seconds=2)).isoformat()
    elif fault == "future":
        capture["observed_at"] = (NOW + timedelta(seconds=2)).isoformat()
    elif fault == "conversion_failure":
        launch["conversion"]["exit_code"] = 1
    elif fault == "conversion_wrong_input":
        launch["conversion"]["input_sha256"] = "wrong"
    elif fault == "duplicate_publisher":
        topic = next(x for x in model.graph["topics"] if x["name"] == "/sample/filtered")
        topic["publishers"].append(topic["publishers"][0])
        model.seal()
    elif fault == "native_wrong_domain":
        model.environment["domain_id"] = "102"
        model.seal()
    elif fault == "missing_ecm_geometry":
        samples[-1]["collisions"][0].pop("geometry_sdf")
    elif fault == "unexpected_model":
        samples[-1]["models"].append({"name": "unexpected", "entity": 99, "world_pose": [0] * 6})
    elif fault == "effective_link_displacement":
        samples[-1]["collisions"][0]["world_pose"][0] = 0.1
    elif fault == "witness_sample_stale":
        capture["physics_witness"]["sample"]["captured_at"] = (
            NOW - timedelta(seconds=2)
        ).isoformat()
    elif fault in {"witness_sample_body", "witness_sample_run", "witness_sample_frame"}:
        key = {
            "witness_sample_body": "body_snapshot_hash",
            "witness_sample_run": "run_id",
            "witness_sample_frame": "frame_id",
        }[fault]
        capture["physics_witness"]["sample"][key] = "wrong"
    elif fault == "witness_sample_incomplete":
        capture["physics_witness"]["sample"]["observation_complete"] = False
    elif fault == "witness_sample_time":
        capture["physics_witness"]["sample"]["time_sec"] = 40
    elif fault == "witness_sample_pose":
        capture["physics_witness"]["sample"]["x"] = 0.1
    elif fault == "witness_sample_lease":
        capture["physics_witness"]["sample"]["lease_enabled"] = True
    elif fault in {"witness_sample_control", "witness_sample_velocity"}:
        key = (
            "source_received_control_messages"
            if fault == "witness_sample_control"
            else "source_published_velocity_messages"
        )
        capture["physics_witness"]["sample"][key] = 1
    elif fault == "witness_implementation":
        capture["physics_witness"]["implementation_sha256"] = "wrong"
    elif fault == "world_alignment":
        launch["physics_world_frame_alignment"] = "UNVERIFIED"
    elif fault == "observer_stale":
        capture["host_monotonic_ns"] += 1_000_000_000
    elif fault == "ros_clock_diverged":
        model.observations["clock_time_sec"] = 40
        model.seal()
    root, request, kwargs = pack(parts)
    result = review_sim_source_binding(root, request, **kwargs)
    assert result["result"] in {"MISMATCH", "UNKNOWN"}, result
    assert result["source_provenance_class"] == "NOT_VERIFIED"
    assert result["execution_authority"] == "NONE" and result["real_authority"] is False


@pytest.mark.parametrize("pin", ["launch_sha256", "capture_sha256"])
def test_self_reported_capture_cannot_supply_its_own_trust_pin(tmp_path, pin):
    root, request, kwargs = pack(fixture(tmp_path))
    request[pin] = "f" * 64
    result = review_sim_source_binding(root, request, **kwargs)
    assert (
        result["result"] == "MISMATCH"
        and "INDEPENDENT_OPERATOR_PIN_MISMATCH" in result["mismatch_reasons"]
    )


@pytest.mark.parametrize(
    "xml",
    [
        b'<!DOCTYPE sdf [<!ENTITY bad "x">]><sdf/>',
        b"<sdf><include/></sdf>",
        b'<sdf><model name="fixture"><link name="a"><collision name="c"><geometry><mesh><uri>missing</uri></mesh></geometry></collision></link></model></sdf>',
    ],
)
def test_unresolved_sources_never_get_default_geometry(xml):
    with pytest.raises(ValueError):
        sdf_inventory(xml, robot_entity="fixture")


def test_source_match_preserves_blocked_localization_review_without_claiming_l1(tmp_path):
    parts = fixture(tmp_path)
    model = parts[-1]
    for quality in model.observations["localization_quality"]:
        if quality["source"] == "/sample/estimated_pose":
            quality["x_variance"] = -0.01
    model.seal()
    root, request, kwargs = pack(parts)
    result = review_sim_source_binding(root, request, **kwargs)
    assert result["result"] == "MATCHED", result
    assert result["navigation_observation_review"]["status"] == "UNKNOWN"
    assert "fresh_localization_estimate_and_covariance" in result["deferred_L1_checks"]
    assert result["current_navigation_readiness"] == "NOT_EVALUATED"
    assert result["L1"] == result["L2"] == "NOT_RUN"


def test_localization_publisher_ambiguity_is_never_deferred_to_l1(tmp_path):
    parts = fixture(tmp_path)
    model = parts[-1]
    topic = next(t for t in model.graph["topics"] if t["name"] == "/sample/estimated_pose")
    topic["publishers"].append("/sample/untrusted_localization")
    model.seal()
    root, request, kwargs = pack(parts)
    result = review_sim_source_binding(root, request, **kwargs)
    assert result["result"] == "UNKNOWN"
    assert "ROS_SOURCE:unique_typed_localization_publisher" in result["unknown_reasons"]


@pytest.mark.parametrize("fault", ["none", "plane_size", "plane_pose", "model_pose", "dynamic"])
def test_complete_static_world_plane_and_effective_frames(tmp_path, fault):
    parts = list(fixture(tmp_path))
    _, _, launch, capture, samples, _, world, _ = parts
    tree = ET.fromstring(world)
    floor = ET.fromstring(
        '<model name="floor"><static>true</static><link name="ground"><collision name="surface"><geometry><plane><normal>0 0 1</normal><size>10 10</size></plane></geometry></collision></link></model>'
    )
    tree.find("world").append(floor)
    if fault == "dynamic":
        floor.find("static").text = "false"
    parts[6] = ET.tostring(tree)
    launch["asset_sha256"]["world.sdf"] = sha(parts[6])
    capture["assets_before"] = deepcopy(launch["asset_sha256"])
    capture["assets_after"] = deepcopy(launch["asset_sha256"])
    for sample in samples:
        sample["models"].append(
            {"name": "floor", "entity": 10, "static": fault != "dynamic", "world_pose": [0] * 6}
        )
        sample["collisions"].append(
            {
                "entity": 12,
                "name": "surface",
                "pose": [0] * 6,
                "world_pose": [0] * 6,
                "geometry_sdf": "<geometry><plane><normal>0 0 1</normal><size>10 10</size></plane></geometry>",
                "ancestry": [
                    {"entity": 11, "name": "ground", "model": False, "world": False},
                    {"entity": 10, "name": "floor", "model": True, "world": False},
                    {"entity": 1, "name": "owned_world", "model": False, "world": True},
                ],
            }
        )
    if fault == "plane_size":
        samples[-1]["collisions"][-1]["geometry_sdf"] = samples[-1]["collisions"][-1][
            "geometry_sdf"
        ].replace("10 10", "5 5")
    elif fault == "plane_pose":
        samples[-1]["collisions"][-1]["world_pose"][0] = 0.1
    elif fault == "model_pose":
        samples[-1]["models"][-1]["world_pose"][0] = 0.1
    root, request, kwargs = pack(parts)
    result = review_sim_source_binding(root, request, **kwargs)
    assert result["result"] == ("MATCHED" if fault == "none" else "MISMATCH"), result


@pytest.mark.parametrize(
    "loaded_kind,expected", [("navsat", "MATCHED"), ("imu", "MISMATCH"), ("UNKNOWN", "UNKNOWN")]
)
@pytest.mark.parametrize("declared_kind", ["navsat", "camera", "depth_camera", "rgbd_camera"])
def test_declared_sensor_requires_exact_loaded_type(tmp_path, loaded_kind, expected, declared_kind):
    parts = list(fixture(tmp_path))
    root, request, launch, capture, samples, converted, world, model = parts
    sensor = (
        f'<sensor name="source_sensor" type="{declared_kind}"><pose>0 0 0.1 0 0 0</pose></sensor>'
    ).encode()
    converted = converted.replace(b"</link>", sensor + b"</link>")
    world = world.replace(b"</link>", sensor + b"</link>")
    launch["asset_sha256"].update({"converted.sdf": sha(converted), "world.sdf": sha(world)})
    launch["conversion"]["output_sha256"] = sha(converted)
    capture["assets_before"] = deepcopy(launch["asset_sha256"])
    capture["assets_after"] = deepcopy(launch["asset_sha256"])
    for sample in samples:
        sample["sensors"] = [
            {
                "entity": 5,
                "name": "source_sensor",
                "kind": declared_kind if loaded_kind == "navsat" else loaded_kind,
                "pose": [0, 0, 0.1, 0, 0, 0],
                "world_pose": [0, 0, 0.1, 0, 0, 0],
                "ancestry": deepcopy(sample["collisions"][0]["ancestry"]),
            }
        ]
    parts[5:7] = [converted, world]
    root, request, kwargs = pack(parts)
    result = review_sim_source_binding(root, request, **kwargs)
    assert result["result"] == expected, result
    assert result["execution_authority"] == "NONE"
