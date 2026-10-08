"""Independent SIM robot/probe observer; only observation publisher, no services.

No scene operation is executed here. The owned instrument controller must
supply retained original lift request/reply evidence. Without a measured cache
cycle the additional actor interlock stays closed. This entry is not yet joined
to a qualified physical launcher or canonical Native acceptance.
"""

import argparse
import base64
import hashlib
import json
import signal
import time
from pathlib import Path

from backend_observer_replay import BackendObserverReplay
from backend_probe_evidence import probe_policy
from native_contact_evidence import reopen_native_policy
from probe_controller_ipc import ProbeControllerIPC, decode_controller_packet
from probe_scene_geometry import decode_scene_json

from rosclaw.connectors.ros.diagnosis.coverage_audit import CoverageAuditLog


def owned_json(path, directory):
    path, directory = Path(path), Path(directory)
    if (
        path.is_symlink()
        or not path.is_file()
        or not 0 < path.stat().st_size <= 65536
        or not path.resolve().is_relative_to(directory.resolve())
    ):
        raise ValueError("bounded owned regular frozen observer policy required")
    raw = path.read_bytes()
    return raw, decode_scene_json(raw)


def retained(raw, source_type, wall, unix):
    if not 0 < len(raw) <= 262144:
        raise ValueError("bounded original independent observation bytes required")
    return {
        "original_source_base64": base64.b64encode(raw).decode("ascii"),
        "original_source_sha256": hashlib.sha256(raw).hexdigest(),
        "original_size_bytes": len(raw),
        "source_bytes_complete": True,
        "source_type": source_type,
        "received_monotonic_sec": wall,
        "received_unix_ns": unix,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "directory",
        "robot-directory",
        "probe-directory",
        "robot-policy",
        "probe-policy",
        "robot-plugin",
        "probe-plugin",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--robot-pose-frame", required=True)
    parser.add_argument("--probe-pose-frame", required=True)
    parser.add_argument("--duration", type=int, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--controller-pid", type=int)
    parser.add_argument("--controller-uid", type=int)
    for name in ("scene-binding", "probe-declaration", "scene-directory"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if (args.controller_pid is None) != (args.controller_uid is None):
        raise ValueError("both pinned instrument controller PID and UID required")
    if args.controller_pid is not None and args.scene_binding is None:
        raise ValueError("instrument IPC requires complete spatial observer sources")
    if (
        not 60 <= args.duration <= 1920
        or args.directory.is_symlink()
        or not args.directory.is_dir()
    ):
        raise ValueError("owned observer output and immutable bounded deadline required")
    robot_raw, robot = owned_json(args.robot_policy, args.robot_directory)
    probe_raw, probe = owned_json(args.probe_policy, args.probe_directory)
    robot = reopen_native_policy(args.robot_directory, robot, plugin_path=args.robot_plugin)
    probe_policy(probe)
    reopen_native_policy(
        args.probe_directory, probe["native_policy"], plugin_path=args.probe_plugin
    )
    spatial_args = (args.scene_binding, args.probe_declaration, args.scene_directory)
    if any(v is not None for v in spatial_args) and any(v is None for v in spatial_args):
        raise ValueError("complete owned spatial observer source arguments required")
    spatial_options, frozen_spatial_sources = {}, []
    if args.scene_binding is not None:
        for name, path in (
            ("scene_binding", args.scene_binding),
            ("probe_declaration", args.probe_declaration),
        ):
            raw, value = owned_json(path, args.scene_directory)
            spatial_options[name] = value
            frozen_spatial_sources.append((path, raw))
        observed_topics = {
            topic
            for p in (robot, probe["native_policy"])
            for topic in (
                p["component_topic"],
                p["contact_policy"]["pose_topic"],
                *p["contact_policy"]["contacts"],
            )
        }
        if "/rosclaw_sim/physics_snapshot" in observed_topics:
            raise ValueError("independent scene source aliases a robot/instrument role")
    engine = BackendObserverReplay(
        robot,
        probe,
        robot_pose_frame=args.robot_pose_frame,
        probe_pose_frame=args.probe_pose_frame,
        **spatial_options,
    )
    config = {
        "run_id": engine.binding["run_id"],
        "body_snapshot_hash": engine.binding["body_snapshot_hash"],
        "constraint_policy_hash": engine.gate.policy_hash,
    }
    config_path = args.directory / "backend_actor_constraint.json"
    if args.prepare_only:
        with config_path.open("x") as stream:
            stream.write(json.dumps(config) + "\n")
        return
    _, prepared_config = owned_json(config_path, args.directory)
    if prepared_config != config:
        raise ValueError("prepared actor constraint differs from frozen source policies")
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rclpy.serialization import serialize_message
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage

    rclpy.init()
    node, audit, controller_ipc = None, None, None
    try:
        node = Node(
            "independent_backend_observer",
            enable_rosout=False,
            start_parameter_services=False,
            enable_logger_service=False,
            use_global_arguments=False,
            parameter_overrides=[Parameter("start_type_description_service", value=False)],
        )
        for publisher in tuple(node.publishers):
            if publisher.topic_name != "/parameter_events":
                raise ValueError("unexpected publisher on independent backend observer")
            node.destroy_publisher(publisher)
        if tuple(node.services):
            raise ValueError("unexpected service on independent backend observer")
        audit = CoverageAuditLog(
            args.directory / "backend-observation-events.jsonl",
            capacity=2048,
            context={
                **config,
                "source": "independent_SIM_backend_original_sources",
                "evidence_domain": "SIMULATION",
            },
        )
        if args.controller_pid is not None:
            controller_ipc = ProbeControllerIPC(
                args.directory / "probe-controller.sock",
                controller_pid=args.controller_pid,
                controller_uid=args.controller_uid,
            )
        publisher = node.create_publisher(
            String, "/rosclaw_sim/backend_observation_constraint", 128
        )
        component_qos = QoSProfile(depth=128, reliability=ReliabilityPolicy.BEST_EFFORT)
        pose_qos = QoSProfile(depth=32, reliability=ReliabilityPolicy.BEST_EFFORT)

        def receive(kind, message):
            wall, unix = time.monotonic(), time.time_ns()
            payload = {"received_monotonic_sec": wall, "received_unix_ns": unix}
            try:
                is_pose = kind.endswith("_pose")
                raw = serialize_message(message) if is_pose else message.data.encode("utf-8")
                payload = retained(
                    raw,
                    "tf2_msgs/msg/TFMessage"
                    if is_pose
                    else "gazebo_ecm_postupdate_observation_json"
                    if kind == "backend_scene_components"
                    else "gazebo_ecm_contact_sensor_data_json",
                    wall,
                    unix,
                )
                projection = engine.apply(kind, payload)
                audit.emit(
                    kind,
                    {**payload, "projection": projection},
                    sim_time=projection.get("sim_time_sec"),
                )
            except (ValueError, KeyError, TypeError, UnicodeError) as exc:
                engine.gate.fault = engine.gate.fault or str(exc)
                audit.emit(
                    "backend_observation_source_rejected",
                    {**payload, "source_kind": kind, "error": str(exc)[:512]},
                )

        for prefix, policy in (("robot", robot), ("probe", probe["native_policy"])):
            node.create_subscription(
                TFMessage,
                policy["contact_policy"]["pose_topic"],
                lambda m, p=prefix: receive("backend_" + p + "_pose", m),
                pose_qos,
            )
            node.create_subscription(
                String,
                policy["component_topic"],
                lambda m, p=prefix: receive("backend_" + p + "_components", m),
                component_qos,
            )
        if engine.spatial is not None:
            node.create_subscription(
                String,
                "/rosclaw_sim/physics_snapshot",
                lambda m: receive("backend_scene_components", m),
                component_qos,
            )
        stop = [False]
        signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
        signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
        deadline, next_tick, next_source_check = time.monotonic() + args.duration, 0, 0
        subscriber_seen = False
        while not stop[0] and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.01)
            if controller_ipc is not None:
                wall, unix = time.monotonic(), time.time_ns()
                transport = None
                try:
                    received = controller_ipc.poll()
                    if received is not None:
                        wall, unix = time.monotonic(), time.time_ns()
                        raw, peer = received
                        transport = retained(
                            raw, "linux_unix_seqpacket_probe_controller", wall, unix
                        )
                        transport["kernel_peer_credentials"] = peer
                        kind, payload = decode_controller_packet(
                            raw,
                            run_id=engine.binding["run_id"],
                            constraint_policy_hash=engine.gate.policy_hash,
                        )
                        payload.update(
                            received_monotonic_sec=wall,
                            received_unix_ns=unix,
                            controller_transport=transport,
                        )
                        if kind == "backend_probe_lift_begin":
                            robot_source = engine.gate.robot.snapshot(wall)
                            spatial_source = engine.spatial.snapshot(wall)
                            if (
                                not robot_source["observation_complete"]
                                or robot_source["collision_count"] != 0
                                or not spatial_source["scene_geometry_constraint_satisfied"]
                                or spatial_source["source_fault"]
                                or spatial_source["join_source_fault"]
                            ):
                                raise ValueError(
                                    "original instrument lift requires fresh complete robot/spatial sources"
                                )
                        projection = engine.apply(kind, payload)
                        audit.emit(kind, {**payload, "projection": projection})
                        if audit.dropped or audit.error:
                            raise ValueError("original instrument IPC audit incomplete")
                        controller_ipc.finish(
                            json.dumps(
                                {
                                    "accepted_original_source_event": True,
                                    "transaction_id": payload["transaction_id"],
                                    "event": kind,
                                    "world_source_ownership_admitted": False,
                                    "authorization": False,
                                }
                            ).encode()
                        )
                except (ValueError, OSError) as exc:
                    engine.gate.fault = engine.gate.fault or str(exc)
                    audit.emit(
                        "backend_probe_controller_rejected",
                        {
                            "received_monotonic_sec": wall,
                            "received_unix_ns": unix,
                            "controller_transport": transport,
                            "error": str(exc)[:512],
                        },
                    )
                    controller_ipc.finish(
                        b'{"accepted_original_source_event":false,"authorization":false}'
                    )
            wall = time.monotonic()
            if wall < next_tick:
                continue
            if engine.gate.fault is None:
                try:
                    if (
                        owned_json(args.robot_policy, args.robot_directory)[0] != robot_raw
                        or owned_json(args.probe_policy, args.probe_directory)[0] != probe_raw
                        or any(
                            owned_json(path, args.scene_directory)[0] != raw
                            for path, raw in frozen_spatial_sources
                        )
                    ):
                        raise ValueError("frozen independent observer policy changed")
                except (ValueError, OSError) as exc:
                    engine.gate.fault = str(exc)
                    audit.emit(
                        "backend_observation_policy_rejected",
                        {
                            "received_monotonic_sec": wall,
                            "error": str(exc)[:512],
                        },
                    )
            if audit.dropped or audit.error:
                engine.gate.fault = (
                    engine.gate.fault or "original backend observation audit incomplete"
                )
            if wall >= next_source_check:
                try:
                    reopen_native_policy(args.robot_directory, robot, plugin_path=args.robot_plugin)
                    reopen_native_policy(
                        args.probe_directory, probe["native_policy"], plugin_path=args.probe_plugin
                    )
                except ValueError as exc:
                    engine.gate.fault = engine.gate.fault or str(exc)
                next_source_check = wall + 5
            if publisher.get_subscription_count() < 1:
                if subscriber_seen:
                    engine.gate.fault = (
                        engine.gate.fault or "actor observation subscription disappeared"
                    )
                # Do not skip the genesis sequence while discovering the actor.
                next_tick = wall + 0.05
                continue
            subscriber_seen = True
            # Original controller IPC never substitutes for measured lift,
            # clear cache, recontact or independent world admission.
            payload = {"received_monotonic_sec": wall}
            projection = (
                engine.failed_sample(wall)
                if engine.gate.fault
                else engine.apply("backend_observation_sample", payload)
            )
            audit.emit(
                "backend_observation_sample",
                {**payload, "projection": projection},
                sim_time=projection["snapshot"]["robot_sim_time_sec"],
            )
            publisher.publish(
                String(data=json.dumps(projection["actor_envelope"], allow_nan=False))
            )
            temporary = args.directory / "backend-observation-latest.json.tmp"
            with temporary.open("x") as stream:
                stream.write(json.dumps(projection, allow_nan=False) + "\n")
            temporary.replace(args.directory / "backend-observation-latest.json")
            next_tick = wall + 0.05
    finally:
        if controller_ipc is not None:
            controller_ipc.close()
        try:
            if node is not None:
                node.destroy_node()
        finally:
            try:
                rclpy.shutdown()
            finally:
                if audit is not None:
                    summary = audit.close()
                    with Path(str(audit.path) + ".summary.json").open("x") as stream:
                        stream.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
