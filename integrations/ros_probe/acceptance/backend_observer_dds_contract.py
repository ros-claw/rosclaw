"""Actual isolated DDS lifecycle over explicitly SYNTHETIC component sources.

No Gazebo world, plugin loading, scene service, actuator or Native task. Valid
synthetic streams must remain UNKNOWN without a measured probe intervention;
a bad stream must latch. This checks transport/lifecycle, not physical evidence.
"""

import argparse
import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from actuator_observation_constraint import ActuatorObservationConstraint
from backend_probe_cdr_contract import fixture as probe_fixture
from geometry_msgs.msg import TransformStamped
from native_contact_cdr_contract import fixture as robot_fixture
from native_contact_evidence import prepare_native_policy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--with-spatial", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    (args.output / "original-contract-driver.py").write_bytes(Path(__file__).read_bytes())
    repo = Path(__file__).resolve().parents[3]
    capture = args.output / "original-source-snapshot"
    capture.mkdir()
    inputs = [
        *sorted((repo / "src").rglob("*.py")),
        *sorted(Path(__file__).parent.glob("*.py")),
        repo / "tests/connectors/ros/fixtures/passive-native-contact-contract-packets.jsonl",
        repo / "tests/connectors/ros/fixtures/passive-ecm-body-v2-contract-packets.jsonl",
    ]
    source_hashes = {}
    for path in inputs:
        if path.is_symlink() or not path.is_file():
            raise ValueError("original contract source must be a regular non-symlink file")
        raw = path.read_bytes()
        relative = str(path.relative_to(repo))
        target = capture / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        source_hashes[relative] = hashlib.sha256(raw).hexdigest()
    manifest = json.dumps(source_hashes, sort_keys=True, indent=2).encode() + b"\n"
    (capture / "source-manifest.json").write_bytes(manifest)
    robot_root, probe_root, output = [
        args.output / name for name in ("robot", "instrument", "observer")
    ]
    output.mkdir()
    robot_policy, _, _ = robot_fixture(robot_root, args.plugin, all_steps=True)
    probe_policy, _ = probe_fixture(probe_root, args.plugin)
    # Both are explicitly derived fixtures. Rebind the instrument source to
    # the same synthetic run and preserve exclusive source artifacts/IDs.
    run_id = robot_policy["contact_policy"]["run_id"]
    for name in ("brush_binding.json", "physics_binding.json"):
        path = probe_root / name
        value = json.loads(path.read_text())
        value["run_id"] = run_id
        path.write_text(json.dumps(value) + "\n")
    native = prepare_native_policy(
        probe_root,
        plugin_path=probe_root / "libcontacts.so",
        support_topics=["/instrument/contact"],
        ground_collisions=["support_plane::floor_link::floor_collision"],
        pose_topic="/instrument/pose",
        component_topic="/instrument/components",
        component_gz_topic="/rosclaw_sim/backend_probe_components",
    )
    probe_policy["native_policy"] = native
    probe_path = probe_root / "probe-policy.json"
    probe_path.write_text(json.dumps(probe_policy) + "\n")
    script = Path(__file__).with_name("backend_observer.py")
    argv = [
        sys.executable,
        str(script),
        "--directory",
        str(output),
        "--robot-directory",
        str(robot_root),
        "--probe-directory",
        str(probe_root),
        "--robot-policy",
        str(robot_root / "native-policy.json"),
        "--probe-policy",
        str(probe_path),
        "--robot-plugin",
        str(robot_root / "libcontacts.so"),
        "--probe-plugin",
        str(probe_root / "libcontacts.so"),
        "--robot-pose-frame",
        "fixture_world",
        "--probe-pose-frame",
        "fixture_world",
        "--duration",
        "60",
    ]
    scene = None
    if args.with_spatial:
        sources = Path(__file__).resolve().parents[3] / "tests/connectors/ros/fixtures"
        scene = next(
            json.loads(line)["packet"]
            for line in (sources / "passive-ecm-body-v2-contract-packets.jsonl")
            .read_text()
            .splitlines()
            if json.loads(line)["case"].startswith("v2_")
        )
        b = robot_policy["contact_policy"]
        declaration = json.loads((probe_root / "probe-fixture.json").read_text())["declaration"]
        declaration.update(run_id=b["run_id"], robot_model_name=b["model_name"])
        scene.update({k: b[k] for k in ("run_id", "body_snapshot_hash", "attachment_hash")})
        scene.update(world_name=robot_policy["world_name"], paused=False)
        scene["body"]["collision_geometry"][0]["entity_id"] = 6
        scene["obstacles"][0]["world_pose"][:2] = [3, 3]
        scene["obstacles"].append(
            {
                "model_name": "instrument_probe",
                "entity_id": 40,
                "world_pose": [6, 0, 0.05, 1, 0, 0, 0],
                "collision_geometry": [
                    {
                        "entity_id": 46,
                        "kind": "sphere",
                        "radius": 0.05,
                        "enclosing_radius_m": 0.05,
                        "model_relative_pose": [0, 0, 0, 1, 0, 0, 0],
                    }
                ],
            }
        )
        scene["scene_models"].append({"model_name": "instrument_probe", "entity_id": 40})
        scene_binding = {
            **{k: b[k] for k in ("run_id", "body_snapshot_hash", "attachment_hash")},
            "body_model_name": b["model_name"],
            "world_name": robot_policy["world_name"],
            "obstacle_names": ["anonymous_blocker", "instrument_probe"],
            "scene_model_names": [b["model_name"], "anonymous_blocker", "instrument_probe"],
            "world_to_map_xyyaw": [0, 0, 0],
            "map_world_identity_approved": True,
            "frame_transform_source": "simulator_operator_fixture_policy",
            "grid": {"cleaning_polygon": declaration["cleaning_polygon"]},
        }
        scene_dir = args.output / "explicit-synthetic-spatial-sources"
        scene_dir.mkdir()
        for name, value in (("binding.json", scene_binding), ("declaration.json", declaration)):
            (scene_dir / name).write_text(json.dumps(value) + "\n")
        argv += [
            "--scene-binding",
            str(scene_dir / "binding.json"),
            "--probe-declaration",
            str(scene_dir / "declaration.json"),
            "--scene-directory",
            str(scene_dir),
        ]
    subprocess.run([*argv, "--prepare-only"], check=True, timeout=20)
    config = json.loads((output / "backend_actor_constraint.json").read_text())
    binding = {k: robot_policy["contact_policy"][k] for k in ("run_id", "body_snapshot_hash")}
    receiver = ActuatorObservationConstraint(config, binding)
    templates = [
        json.loads(line)
        for line in (
            Path(__file__).resolve().parents[3]
            / "tests/connectors/ros/fixtures/passive-native-contact-contract-packets.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    robot = next(
        row["packet"] for row in templates if row["case"] == "native_actual_contact_entity_names"
    )
    probe = copy.deepcopy(robot)
    probe.update(
        {
            k: native["contact_policy"][k]
            for k in ("run_id", "body_snapshot_hash", "attachment_hash", "producer_id")
        }
    )
    probe.update(body_model_name="instrument_probe", body_model_entity_id=40)
    source = probe["contact_sources"][0]
    source.update(
        sensor_entity_id=48,
        sensor_name="probe_contact",
        link_entity_id=44,
        link_name="probe_link",
        gz_topic="/rosclaw_sim/backend_probe_contact",
        collision_entity_ids=[46],
    )
    collision = "instrument_probe::probe_link::probe_collision"
    probe["collision_contacts"][0].update(collision_entity_id=46, collision_name=collision)
    probe["collision_contacts"][0]["contacts"][0].update(
        collision1_entity_id=46, collision1_name=collision
    )
    import rclpy

    rclpy.init()
    node, child, child_log = None, None, None
    received = []
    try:
        node = Node(
            "explicit_synthetic_DDS_contract_source",
            enable_rosout=False,
            start_parameter_services=False,
        )
        qos = QoSProfile(depth=128, reliability=ReliabilityPolicy.BEST_EFFORT)
        robot_pose = node.create_publisher(
            TFMessage, robot_policy["contact_policy"]["pose_topic"], qos
        )
        robot_components = node.create_publisher(String, robot_policy["component_topic"], qos)
        probe_pose = node.create_publisher(TFMessage, native["contact_policy"]["pose_topic"], qos)
        probe_components = node.create_publisher(String, native["component_topic"], qos)
        scene_components = (
            node.create_publisher(String, "/rosclaw_sim/physics_snapshot", qos)
            if scene is not None
            else None
        )

        def envelope(message):
            received.append(json.loads(message.data))
            receiver.receive(message.data, time.monotonic())

        node.create_subscription(
            String, "/rosclaw_sim/backend_observation_constraint", envelope, 128
        )
        child_log = (args.output / "actual-observer.log").open("x")
        child = subprocess.Popen(
            argv, stdout=child_log, stderr=subprocess.STDOUT, env=os.environ.copy()
        )
        deadline = time.monotonic() + 15
        source_publishers = [robot_pose, robot_components, probe_pose, probe_components]
        if scene_components is not None:
            source_publishers.append(scene_components)
        while any(p.get_subscription_count() < 1 for p in source_publishers):
            if child.poll() is not None or time.monotonic() >= deadline:
                raise ValueError("actual observation subscriptions failed to discover")
            rclpy.spin_once(node, timeout_sec=0.02)

        # Discovery can precede the initial BEST_EFFORT data connection.
        # Settle without publishing packets or consuming source sequences.
        # This setup interval never extends a mission deadline.
        settle_until = time.monotonic() + 0.5
        while time.monotonic() < settle_until:
            rclpy.spin_once(node, timeout_sec=0.01)

        def frame(index, bad=False):
            sim = 1 + index * 0.01
            if scene_components is not None and index % 5 == 0:
                value = copy.deepcopy(scene)
                value.update(
                    sequence=index // 5,
                    physics_iteration=index + 1,
                    sim_time_sec=round(sim * 1e9) / 1e9,
                    captured_at_unix_ns=time.time_ns(),
                )
                scene_components.publish(String(data=json.dumps(value)))
            for packet, pose_pub, component_pub, xyz in (
                (robot, robot_pose, robot_components, [0, 0, 0]),
                (probe, probe_pose, probe_components, [6, 0, 0.05]),
            ):
                if packet is probe and index % 5:
                    continue
                item = TransformStamped()
                item.header.frame_id, item.child_frame_id = (
                    "fixture_world",
                    packet["body_model_name"],
                )
                item.header.stamp.sec, item.header.stamp.nanosec = divmod(
                    round(sim * 1e9), 1_000_000_000
                )
                (
                    item.transform.translation.x,
                    item.transform.translation.y,
                    item.transform.translation.z,
                ) = [float(v) for v in xyz]
                item.transform.rotation.w = 1.0
                pose_pub.publish(TFMessage(transforms=[item]))
                value = copy.deepcopy(packet)
                value.update(
                    sequence=index // 5 if packet is probe else index,
                    iterations=index + 1,
                    sim_time_sec=item.header.stamp.sec + item.header.stamp.nanosec / 1e9,
                    physics_step_dt_sec=0.01,
                    body_world_pose=[*xyz, 1, 0, 0, 0],
                    captured_at_unix_ns=time.time_ns(),
                )
                if bad and packet is robot:
                    value["sequence"] = 0
                component_pub.publish(String(data=json.dumps(value)))
            until = time.monotonic() + 0.01
            while time.monotonic() < until:
                rclpy.spin_once(node, timeout_sec=0.005)

        for index in range(400):
            frame(index)
        valid = [
            r
            for r in received
            if r["robot_sim_time_sec"] is not None
            and r["probe_sim_time_sec"] is not None
            and r["source_fault"] is None
        ]
        if not valid or any(
            r["live_source_constraint_satisfied"] or r["authorization"] or r["source_fault"]
            for r in received
        ):
            raise ValueError(
                "actual DDS observer admitted an unqualified synthetic probe or had no intact paired source"
            )
        frame(400, bad=True)
        for index in range(401, 448):
            frame(index)
        faulted = [r for r in received if r["source_fault"]]
        if (
            not faulted
            or any(r["live_source_constraint_satisfied"] for r in faulted)
            or receiver.fault is None
        ):
            raise ValueError("actual DDS malformed-source rejection did not remain latched")
        child.send_signal(signal.SIGINT)
        code = child.wait(timeout=10)
        if code != 0:
            raise ValueError("actual independent observer did not close cleanly")
        summary = json.loads((output / "backend-observation-events.jsonl.summary.json").read_text())
        if (
            summary["complete"] is not True
            or summary["writer_stopped"] is not True
            or summary["dropped_events"] != 0
            or summary["writer_error"] is not None
        ):
            raise ValueError("actual DDS original observation writer did not close losslessly")
        events = [
            json.loads(line)
            for line in (output / "backend-observation-events.jsonl").read_text().splitlines()
        ]
        rejected = [
            event for event in events if event["kind"] == "backend_observation_source_rejected"
        ]
        accepted_robot = [
            event
            for event in events
            if event["kind"] == "backend_robot_components"
            and not event["payload"]["projection"].get("startup_pose_pending")
        ]
        if not rejected or accepted_robot[-1]["payload"]["projection"]["sequence"] != 399:
            raise ValueError("DDS valid 100Hz source prefix did not reach its intended final frame")
        first_bad = json.loads(
            __import__("base64").b64decode(rejected[0]["payload"]["original_source_base64"])
        )
        expected_error = (
            "original spatial source step/sequence missing or regressed"
            if args.with_spatial
            else "repeated/regressed"
        )
        if (
            first_bad["sequence"] != 0
            or first_bad["iterations"] != 401
            or rejected[0]["payload"]["source_kind"] != "backend_robot_components"
            or expected_error not in rejected[0]["payload"]["error"]
        ):
            raise ValueError("DDS first rejection was not the retained intentional sequence fault")
        spatial_samples = [
            event["payload"]["projection"]["snapshot"]["scene_geometry_constraint"]
            for event in events
            if args.with_spatial
            and event["kind"] == "backend_observation_sample"
            and event["payload"]["projection"]["snapshot"]["source_fault"] is None
        ]
        original_joins = [
            joined
            for event in events
            for joined in event["payload"].get("projection", {}).get("joined_spatial_sources", [])
        ]
        if args.with_spatial and (
            not any(s["scene_geometry_constraint_satisfied"] for s in spatial_samples)
            or [j["original_sim_step_ns"] for j in original_joins]
            != [1_000_000_000 + i * 50_000_000 for i in range(80)]
        ):
            raise ValueError("actual DDS spatial prefix did not join all 80 intended exact steps")
        if any(
            hashlib.sha256((repo / relative).read_bytes()).hexdigest() != sha
            for relative, sha in source_hashes.items()
        ):
            raise ValueError(
                "original Python/fixture contract sources changed during actual DDS run"
            )
        report = {
            "status": "PASS_ACTUAL_DDS_OVER_EXPLICITLY_SYNTHETIC_SOURCES",
            "received_envelopes": len(received),
            "robot_source_rate_hz": 100,
            "last_accepted_robot_sequence": 399,
            "first_rejected_robot_iteration": 401,
            "first_rejection_matches_retained_intentional_fault": True,
            "probe_source_rate_hz": 20,
            "spatial_source_join_required": args.with_spatial,
            "spatial_source_rate_hz": 20 if args.with_spatial else None,
            "first_iteration_positive_even_at_initial_sequence_zero": True,
            "completed_spatial_joins_before_intentional_fault": len(original_joins)
            if args.with_spatial
            else None,
            "transport_settle_interval_sec": 0.5,
            "original_source_files_captured": len(source_hashes),
            "original_source_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "original_source_scope": "ALL_SRC_PYTHON_ACCEPTANCE_PYTHON_AND_TWO_SDK_DERIVED_FIXTURE_INPUTS",
            "source_bytes_unchanged_during_run": True,
            "intact_unqualified_observations": len(valid),
            "faulted_observations": len(faulted),
            "observer_exit_code": code,
            "summary": summary,
            "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
            "source_domain": "SYNTHETIC_SDK_DERIVED_NOT_GAZEBO_PHYSICS",
            "actual_Gazebo_world_or_plugin_load": False,
            "scene_service_or_actuator_started": False,
            "native_tasks": 0,
            "physical_acceptance": "NOT_RUN",
            "backend_health_admitted": False,
            "authorization": False,
        }
        (args.output / "DDS-contract-review.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        try:
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
        finally:
            if child_log is not None:
                child_log.close()
            if node is not None:
                node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
