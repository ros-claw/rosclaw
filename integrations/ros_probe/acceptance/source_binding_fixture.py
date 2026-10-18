"""SDK-owned passive known-fixture capture and real-observation Body compilation.

Only the isolated operator fixture launches ROS processes. The physical witness
is read-only. Explicit roles may add a passive daemon or a separate governed
sink; the sink refuses leases until independent operator SIM admission. The
external supervisor separately inspects and pins the running instance.
"""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

from rosclaw.connectors.ros.context.navigation_review import review_navigation_candidate
from rosclaw.connectors.ros.context.navigation_workspace import compile_navigation_body_workspace
from rosclaw.connectors.ros.context.navigation_workspace_review import (
    inspect_navigation_body_workspace,
)
from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot
from rosclaw.connectors.ros.intelligence import build_system_model

OUT = Path("/evidence")
ROOT = Path(__file__).resolve().parent


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def process_fact(pid):
    path = Path("/proc") / str(pid)
    return {
        "pid": pid,
        "cmdline": path.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode(),
        "stat": path.joinpath("stat").read_text(),
    }


def observer_facts():
    servers, loaded = [], []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            maps = (path / "maps").read_text()
            if "libgz-sim8.so" in maps:
                servers.append(process_fact(int(path.name)))
            if "/observer/librosclaw_source_witness.so" in maps:
                loaded.append(process_fact(int(path.name)))
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return {
        "gz_servers": servers,
        "observer_loaded_servers": loaded,
        "library_sha256": hashlib.sha256(
            Path("/observer/librosclaw_source_witness.so").read_bytes()
        ).hexdigest(),
    }


def main():
    settings = json.loads(Path("/settings.json").read_text())
    from simulation_engineering_capture import validate_engineering_capture

    engineering_capture = validate_engineering_capture(settings)
    governed = settings.get("governed_sink")
    if governed is not None and (
        type(governed) is not dict
        or set(governed) != {"schema_version", "P03_review_sha256"}
        or governed["schema_version"] != "rosclaw.governed_sink_bootstrap.v1"
        or type(governed["P03_review_sha256"]) is not str
        or len(governed["P03_review_sha256"]) != 64
    ):
        raise ValueError("explicit closed governed SDK sink bootstrap required")
    fault = settings.get("fault", "none")
    if fault not in {
        "none",
        "world_reset",
        "witness_exit",
        "witness_rename",
        "duplicate_publisher",
        "sensor_stop",
        "clock_freeze",
        "tf_loss",
        "nav2_inactive",
        "qos_mismatch",
        "namespace_mismatch",
    }:
        raise ValueError("unsupported explicit disposable SIM fault")
    requirements = {
        "interface_names": {
            "mapping.occupancy_map": "/map",
            "state.odometry": "/odom",
            "sensing.lidar": "/scan",
            "navigation.navigate_to_pose": "/navigate_to_pose",
        },
        "lifecycle_nodes": {
            "localization": "/amcl",
            "planner": "/planner_server",
            "controller": "/velocity_smoother",
            "collision_monitor": "/collision_monitor",
            "navigator": "/bt_navigator",
        },
        "costmap_topics": ["/local_costmap/costmap", "/global_costmap/costmap"],
        "localization_topic": "/amcl_pose",
        "velocity_input_topic": "/cmd_vel_smoothed",
        "velocity_output_topic": "/nav_cmd_vel",
        "velocity_consumer_node": "/rosclaw_sim_witness",
    }
    prepared_sources = None
    if Path("/assets/bringup.json").is_file():
        requirements = json.loads(Path("/assets/requirements.json").read_text())
        prepared_sources = json.loads(Path("/assets/observer-sources.json").read_text())
    sink_namespace = prepared_sources["observer_node"].rsplit("/", 1)[0] if prepared_sources else ""
    sink_node = sink_namespace + "/rosclaw_governed_sink"
    if governed is not None:
        requirements["velocity_consumer_node"] = sink_node
        control = OUT / "execution-control"
        control.mkdir(mode=0o700)
        key_fd = os.open(control / "fixture.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(key_fd, "wb") as key_file:
            key_file.write(os.urandom(32))
    save(
        "observed-requirements.json" if prepared_sources is not None else "requirements.json",
        requirements,
    )
    stack_log = (OUT / "stack-original.log").open("x")
    argv = [
        "python3",
        str(ROOT / "stack.py"),
        "--navigation-only",
        "--source-only-observer",
        "--external-source-witness",
        "--prepared-assets",
        "/assets",
        "--expected-prepared-assets-sha256",
        settings["inventory_sha256"],
        "--run-id",
        settings["run_id"],
        "--seed",
        str(settings.get("seed", 101201)),
    ]
    if "probe_snapshot_period_sec" in settings:
        period = settings["probe_snapshot_period_sec"]
        if type(period) not in (int, float) or not 0.1 <= period <= 1.0:
            raise ValueError("explicit read-only probe cadence must be 0.1..1.0 seconds")
        argv.extend(["--probe-snapshot-period-sec", str(period)])
    if (
        settings.get("live_dual_supervision", False)
        or governed is not None
        or fault
        in {
            "sensor_stop",
            "clock_freeze",
            "tf_loss",
            "nav2_inactive",
            "qos_mismatch",
            "namespace_mismatch",
        }
    ):
        argv.append("--readiness-fault-fixture")
    if prepared_sources is not None:
        argv = ["python3", str(ROOT / "prepared_source_stack.py")]
    save("stack-argv.json", argv)
    stack = subprocess.Popen(
        argv, stdout=stack_log, stderr=subprocess.STDOUT, start_new_session=True
    )
    witness = None
    auxiliary = None
    passive_daemon = None
    passive_log = None
    governed_sink = None
    governed_log = None
    consumer_witness = None
    consumer_log = None
    witness_logs = []
    if Path("/assets/collision-asset-originals.json").is_file():
        from simulation_collision_resources import prewarm_prepared_collision_facts

        prewarm_prepared_collision_facts()
    rclpy.init(args=["--ros-args", "-p", "use_sim_time:=true"])
    node = Node("rosclaw_source_capture")
    state = {"native": None, "model": None, "review": None, "physics": None}
    reviews = []
    native_sequence = 0

    def observed(message):
        nonlocal native_sequence
        received_ns = time.monotonic_ns()
        from simulation_native_capture import persist_native_capture

        index = native_sequence
        native_sequence += 1
        native = persist_native_capture(
            OUT,
            message.data.encode("utf-8"),
            index=index,
            received_monotonic_ns=received_ns,
            publish_pointer=bool(
                settings.get("live_dual_supervision", False)
                or settings.get("live_current_facts", False)
            ),
        )
        graph = RosGraphSnapshot.from_dict(
            {
                "ros_version": "ros2",
                "distro": "jazzy",
                "endpoint": "fixture://actual-gazebo-source",
                "captured_at": native["captured_at"],
                **native["graph"],
            }
        )
        model = build_system_model(graph, native=native)
        review = review_navigation_candidate(
            model, Path("/assets/robot.urdf").read_bytes(), requirements
        )
        save(f"sealed-system-{index:03}.json", model.to_dict())
        reviews.append({"index": index, "captured_at": native["captured_at"], "review": review})
        state.update(native=native, model=model, review=review)

    def physics(message):
        state["physics"] = json.loads(message.data)

    node.create_subscription(
        String,
        "/rosclaw_probe/snapshot",
        observed,
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
    )
    node.create_subscription(
        String,
        prepared_sources["topics"]["observation"]
        if prepared_sources
        else "/rosclaw_sim/observation",
        physics,
        10,
    )

    def launch_witness(name, workspace, config):
        nonlocal governed_sink, governed_log
        log = (OUT / f"{name}-original.log").open("x")
        witness_logs.append(log)
        args = [
            "python3",
            str(ROOT / "witness.py"),
            "--ros-args",
            "-p",
            "use_sim_time:=true",
            "-p",
            "read_only_source_observer:=true",
            "-p",
            "body_workspace:=" + str(workspace),
            "-p",
            "execution_config_path:=" + str(config),
        ]
        if settings.get("generic_source_observer") is True:
            from simulation_witness_sources import _ground_names

            if prepared_sources is not None:
                inputs = dict(prepared_sources)
            else:
                world_bytes = Path("/assets/world.sdf").read_bytes()
                urdf_bytes = Path("/assets/robot.urdf").read_bytes()
                import xml.etree.ElementTree as ET

                world = ET.fromstring(world_bytes).find("world")
                robots = [
                    m.get("name")
                    for m in world.findall("model")
                    if m.findtext("static", "false") not in {"true", "1"}
                ]
                if len(robots) != 1:
                    raise ValueError("known-source generic observer fixture requires one robot")
                contacts = json.loads(Path("/assets/contact_topics.json").read_text())
                import yaml

                nav = yaml.safe_load(Path("/assets/nav2.yaml").read_text())
                map_frame = nav["global_costmap"]["global_costmap"]["ros__parameters"][
                    "global_frame"
                ]
                if type(map_frame) is not str or not map_frame:
                    raise ValueError("operator-declared global map frame required")
                inputs = {
                    "schema_version": "rosclaw.simulation_witness_sources.v1",
                    "run_id": settings["run_id"],
                    "world_id": world.get("name"),
                    "robot_entity": robots[0],
                    "observer_node": "/rosclaw_sim_witness",
                    "map_frame": map_frame,
                    "source_urdf_sha256": hashlib.sha256(urdf_bytes).hexdigest(),
                    "world_sdf_sha256": hashlib.sha256(world_bytes).hexdigest(),
                    "topics": {
                        "observation": "/rosclaw_sim/observation",
                        "lease": "/rosclaw_sim/lease",
                        "ground_truth": "/rosclaw_sim/ground_truth",
                        "localization": requirements["localization_topic"],
                        "velocity_input": requirements["velocity_output_topic"],
                        "drive_command": "/drive_controller/cmd_vel",
                        "collision_monitor_state": "/collision_monitor_state",
                    },
                    "contact_topics": contacts,
                    "ground_witness_topics": [t for t in contacts if "/wheel_" in t],
                    "ground_collision_names": sorted(
                        _ground_names(world_bytes, world.get("name"), robots[0])
                    ),
                    "drive_twist_stamped": True,
                }
            path = OUT / f"{name}-observer-sources.json"
            save(path.name, inputs)
            args += [
                "-p",
                "observer_sources_path:=" + str(path),
                "-p",
                "observer_sources_sha256:=" + hashlib.sha256(path.read_bytes()).hexdigest(),
            ]
            if governed is not None:
                source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
                metadata = {
                    "schema_version": "rosclaw.external_sim_sink_observation.v1",
                    "sink_node": sink_node,
                    "run_id": settings["run_id"],
                    "source_inputs_sha256": source_sha,
                    "lease_service": inputs["topics"]["lease"],
                    "velocity_input": inputs["topics"]["velocity_input"],
                    "status_file": "governed-sink-status.json",
                }
                metadata_path = OUT / f"{name}-external-sink-observation.json"
                save(metadata_path.name, metadata)
                args += [
                    "-p",
                    "external_sink_observation_path:=" + str(metadata_path),
                    "-p",
                    "external_sink_observation_sha256:="
                    + hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                ]
                if governed_sink is None:
                    sink_argv = [
                        "python3",
                        str(ROOT / "simulation_governed_sink.py"),
                        "--sources-path",
                        str(path),
                        "--sources-sha256",
                        source_sha,
                        "--p03-review-sha256",
                        governed["P03_review_sha256"],
                        "--ros-args",
                        "-p",
                        "use_sim_time:=true",
                        "-r",
                        "__ns:=" + sink_namespace,
                    ]
                    save("governed-sink-argv.json", sink_argv)
                    governed_log = (OUT / "governed-sink-original.log").open("x")
                    governed_sink = subprocess.Popen(
                        sink_argv,
                        stdout=governed_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    save("governed-sink-process.json", process_fact(governed_sink.pid))
                expected_body = (
                    json.loads(config.read_text())["body_snapshot_hash"]
                    if config.exists()
                    else None
                )
                deadline = time.monotonic() + 10
                while True:
                    if governed_sink.poll() is not None:
                        raise RuntimeError("owned governed sink exited before witness enrollment")
                    status_path = OUT / "governed-sink-status.json"
                    if status_path.exists():
                        sink_status = json.loads(status_path.read_text())
                        if sink_status["terminal"]:
                            raise RuntimeError(
                                "owned governed sink became terminal before admission"
                            )
                        if sink_status["body_snapshot_hash"] == expected_body:
                            break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("governed bootstrap Body enrollment timeout")
                    rclpy.spin_once(node, timeout_sec=0.05)
        if prepared_sources is not None:
            observer_namespace, observer_name = prepared_sources["observer_node"].rsplit("/", 1)
            args += ["-r", "__ns:=" + observer_namespace, "-r", "__node:=" + observer_name]
        if settings.get("live_continuity_hold_seconds", 0):
            args += ["-p", "max_contact_segments:=4"]
        if name == "renamed-source-witness":
            args += ["-r", "__node:=rosclaw_renamed_source_witness"]
        save(f"{name}-argv.json", args)
        return subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def spin_until(condition, seconds):
        nonlocal passive_daemon
        deadline = time.monotonic() + seconds
        while not condition():
            if time.monotonic() >= deadline:
                save("incomplete-original-reviews.json", reviews)
                raise RuntimeError("bounded passive capture condition not achieved")
            if stack.poll() is not None or (witness and witness.poll() is not None):
                raise RuntimeError("owned source stack or witness exited unexpectedly")
            if auxiliary and auxiliary.poll() is not None:
                raise RuntimeError("explicit endpoint-only fault process exited unexpectedly")
            restart = OUT / "passive-daemon-restart-request.json"
            if passive_daemon and restart.exists():
                request = json.loads(restart.read_text())
                ready = json.loads(
                    (OUT / "supervision" / f"daemon-ready-{passive_daemon.pid}.json").read_text()
                )
                if (
                    set(request) != {"run_id", "expected_pid", "expected_generation"}
                    or request["run_id"] != settings["run_id"]
                    or request["expected_pid"] != passive_daemon.pid
                    or request["expected_generation"] != ready["daemon_instance_id"]
                    or (OUT / "passive-daemon-restart-consumed.json").exists()
                ):
                    raise ValueError("one explicitly owned passive daemon replacement required")
                restart.rename(OUT / "passive-daemon-restart-consumed.json")
                previous = passive_daemon.pid
                stop(passive_daemon)
                save(
                    "passive-daemon-original-generation-exit.json",
                    {
                        "pid": previous,
                        "returncode": passive_daemon.returncode,
                        "daemon_instance_id": ready["daemon_instance_id"],
                    },
                )
                argv = json.loads((OUT / "passive-daemon-argv.json").read_text())
                passive_daemon = subprocess.Popen(
                    argv, stdout=passive_log, stderr=subprocess.STDOUT, start_new_session=True
                )
                save("passive-daemon-replacement-process.json", process_fact(passive_daemon.pid))
            if passive_daemon and passive_daemon.poll() is not None:
                raise RuntimeError("owned passive rosclawd exited unexpectedly")
            if governed_sink and governed_sink.poll() is not None:
                raise RuntimeError("owned governed sink exited unexpectedly")
            if consumer_witness and consumer_witness.poll() is not None:
                raise RuntimeError("owned read-only costmap consumer witness exited unexpectedly")
            rclpy.spin_once(node, timeout_sec=0.05)

    try:
        if prepared_sources is None:
            spin_until(lambda: (OUT / "execution_config.json").exists(), 20)
        witness = launch_witness(
            "initial-source-witness", OUT / "home", OUT / "execution_config.json"
        )
        if prepared_sources is not None:
            # Controller bringup has its own bounded SDK subprocess wait. Do
            # not spend the lifecycle/physics window before Nav2 is spawned.
            spin_until(lambda: (OUT / "prepared-stack-spawned-original.json").exists(), 75)
            spawned = json.loads((OUT / "prepared-stack-spawned-original.json").read_text())
            if spawned["run_id"] != settings["run_id"] or spawned["owner_pid"] != stack.pid:
                raise ValueError("owned prepared process milestone identity differs")
        consumer = settings.get("costmap_consumer_capture")
        if consumer is not None:
            if (
                not Path("/.dockerenv").is_file()
                or os.environ.get("ROS_LOCALHOST_ONLY") != "1"
                or governed is None
                or type(consumer) is not dict
                or set(consumer) != {"schema_version", "configuration_sha256"}
                or consumer["schema_version"] != "rosclaw.costmap_consumer_capture.v1"
                or type(consumer["configuration_sha256"]) is not str
                or consumer["configuration_sha256"]
                != hashlib.sha256(Path("/assets/costmap-witness.json").read_bytes()).hexdigest()
            ):
                raise ValueError("explicit pinned disposable SDK costmap witness required")
            argv = [
                sys.executable,
                str(ROOT / "costmap_consumer_witness.py"),
                "--configuration-sha256",
                consumer["configuration_sha256"],
                "--run-id",
                settings["run_id"],
            ]
            save("costmap-consumer-owned-argv.json", argv)
            consumer_log = (OUT / "costmap-consumer-owned-original.log").open("x")
            consumer_witness = subprocess.Popen(
                argv, stdout=consumer_log, stderr=subprocess.STDOUT, start_new_session=True
            )
            save("costmap-consumer-owned-process.json", process_fact(consumer_witness.pid))
        if engineering_capture and engineering_capture["case"] in {
            "rsk_controller_plugin_unavailable",
            "driver_onboarding",
        }:
            # Configuration can prevent lifecycle activation itself. Capture
            # source-only raw facts before initialization; never bypass L1 or
            # pretend the inactive stack has been admitted for navigation.
            from simulation_engineering_capture import capture_failure

            save("engineering-original-requirements.json", requirements)
            save(
                "counterfactual-initialization-boundary.json",
                {
                    "initial_pose_setup": "NOT_RUN_PREACTIVATION_RAW_CAPTURE",
                    "L1": "NOT_EVALUATED",
                    "L2": "NOT_RUN",
                    "reason": "driver-only source or configuration may have no localization to initialize",
                },
            )
            capture_failure(OUT, engineering_capture, state, spin_until, save)
            return
        from simulation_prepared_lifecycle_config import preinitialization_nodes

        initialization_required_nodes = preinitialization_nodes(
            requirements,
            json.loads((OUT / "bringup.json").read_text())
            if prepared_sources is not None
            else None,
            engineering_capture,
        )
        spin_until(
            lambda: (
                state["model"]
                and state["physics"]
                and state["physics"]["observation_complete"]
                and all(
                    any(
                        row.name == name and row.state == "ACTIVE"
                        for row in state["model"].lifecycle
                    )
                    # Explicit initial localization may precede the new stack's
                    # activation; this wait never substitutes for later full L1.
                    for name in initialization_required_nodes
                )
            ),
            30,
        )
        initializer_argv = [
            "python3",
            str(ROOT / "source_fixture_initialization.py"),
            "--operator-owned-source-fixture",
            "--inventory-sha256",
            settings["inventory_sha256"],
            "--run-id",
            settings["run_id"],
        ]
        save("operator-initialization-argv.json", initializer_argv)
        with (
            (OUT / "operator-initialization-original.stdout").open("xb") as stdout,
            (OUT / "operator-initialization-original.stderr").open("xb") as stderr,
        ):
            initialized = subprocess.run(initializer_argv, stdout=stdout, stderr=stderr, timeout=25)
        save("operator-initialization-exit.json", {"returncode": initialized.returncode})
        if initialized.returncode:
            raise RuntimeError("explicit operator SIM initialization failed")
        if engineering_capture:
            from simulation_engineering_capture import capture_failure

            save("engineering-original-requirements.json", requirements)
            capture_failure(OUT, engineering_capture, state, spin_until, save)
            return
        spin_until(
            lambda: (
                state["review"]
                and state["review"]["status"] == "PROPOSED"
                and state["physics"]
                and state["physics"]["observation_complete"]
            ),
            65,
        )
        initial_physics = state["physics"]
        save("initial-physics-witness.json", initial_physics)
        if any(
            initial_physics[k] != 0
            for k in ("source_received_control_messages", "source_published_velocity_messages")
        ):
            raise RuntimeError("source-only capture received control messages")
        model, review = state["model"], state["review"]
        policy = {
            "schema_version": "rosclaw.navigation_sim_fixture_policy.v1",
            "source": "simulator_operator_fixture_policy",
            "approved": True,
            "policy_id": "G1-known-source-readonly-capture",
            "body_id": settings.get("observed_body_id", "g1-observed-known-base"),
            "source_snapshot_hash": model.snapshot_hash,
            "source_urdf_sha256": review["candidate"]["source_urdf_sha256"],
            "navigation_requirements_hash": review["requirements_hash"],
            "navigation_limits": {
                "max_goal_distance_m": 0.5,
                "max_duration_sec": 30.0,
                "goal_position_tolerance_m": 0.05,
                "goal_yaw_tolerance_rad": 0.1,
                "max_speed_mps": 0.2,
                "max_yaw_rate_rps": 0.5,
                "stop_speed_mps": 0.005,
                "stop_yaw_rate_rps": 0.01,
            },
            "witness_declaration": {
                "observation_topic": prepared_sources["topics"]["observation"]
                if prepared_sources
                else "/rosclaw_sim/observation",
                "lease_service": prepared_sources["topics"]["lease"]
                if prepared_sources
                else "/rosclaw_sim/lease",
                "run_id": settings["run_id"],
            },
        }
        from navigation_fixture_policy import select_navigation_fixture_policy

        policy = select_navigation_fixture_policy(
            policy,
            settings.get("navigation_fixture_policy"),
            action_name=review["candidate"]["interfaces"]["navigation.navigate_to_pose"][0]["name"],
            frame_id=review["candidate"]["frames"]["map"],
        )
        save(
            "navigation-policy-selection.json",
            {
                "scope": "SDK_OWNED_NEW_BODY_POLICY_NOT_MOTION_APPROVAL",
                "operator_input_present": settings.get("navigation_fixture_policy") is not None,
                "policy_id": policy["policy_id"],
                "source_urdf_sha256": policy["source_urdf_sha256"],
                "navigation_requirements_hash": policy["navigation_requirements_hash"],
                "navigation_limits": policy["navigation_limits"],
                "execution_authority": "NONE",
                "current_L1": "NOT_EVALUATED",
            },
        )
        compiled = OUT / "generic-compiled"
        manifest = compile_navigation_body_workspace(
            compiled, model, Path("/assets/robot.urdf").read_bytes(), requirements, policy=policy
        )
        inspection = inspect_navigation_body_workspace(
            compiled,
            expected_manifest_hash=manifest["manifest_hash"],
            expected_body_hash=manifest["body_snapshot_hash"],
        )
        save("compiled-inspection.json", inspection)
        # The supervisor owns this single intentional witness replacement. The
        # stack never masks a crashed child, and the new process has no actuator.
        save("initial-witness-process.json", process_fact(witness.pid))
        stop(witness)
        save("initial-witness-exit.json", {"returncode": witness.returncode})
        state["physics"] = None
        witness = launch_witness(
            "compiled-source-witness", compiled, compiled / "execution_config.json"
        )
        spin_until(
            lambda: (
                state["physics"]
                and state["physics"].get("body_snapshot_hash") == manifest["body_snapshot_hash"]
                and state["physics"]["observation_complete"]
                and not (
                    set(state["review"]["unknown_fields"])
                    - {
                        "fresh_localization_estimate_and_covariance",
                        "fresh_navigation_frame:" + requirements["localization_topic"],
                        "fresh_unique_stream:" + requirements["localization_topic"],
                    }
                )
            ),
            30,
        )
        witness_process = process_fact(witness.pid)
        witness_alive = True
        if fault == "world_reset":
            before_reset_clock = node.get_clock().now().nanoseconds
            reset_argv = [
                "gz",
                "service",
                "-s",
                "/world/ros_expert/control",
                "--reqtype",
                "gz.msgs.WorldControl",
                "--reptype",
                "gz.msgs.Boolean",
                "--timeout",
                "5000",
                "--req",
                "reset: {all: true}",
            ]
            save("fault-reset-argv.json", reset_argv)
            with (
                (OUT / "fault-reset-original.stdout").open("xb") as stdout,
                (OUT / "fault-reset-original.stderr").open("xb") as stderr,
            ):
                reset = subprocess.run(reset_argv, stdout=stdout, stderr=stderr, timeout=8)
            save("fault-reset-exit.json", {"returncode": reset.returncode})
            if reset.returncode:
                raise RuntimeError("actual reset injection failed")
            spin_until(lambda: node.get_clock().now().nanoseconds < before_reset_clock, 5)
            save(
                "fault-reset-applied.json",
                {
                    "before_ros_clock_ns": before_reset_clock,
                    "after_ros_clock_ns": node.get_clock().now().nanoseconds,
                },
            )
        elif fault == "witness_exit":
            stop(witness)
            save(
                "fault-witness-exit.json",
                {"returncode": witness.returncode, "process": witness_process},
            )
            witness = None
            witness_alive = False
        elif fault == "witness_rename":
            stop(witness)
            state["physics"] = None
            witness = launch_witness(
                "renamed-source-witness", compiled, compiled / "execution_config.json"
            )
            spin_until(lambda: state["physics"] and state["physics"]["observation_complete"], 8)
            witness_process = process_fact(witness.pid)
        elif fault == "duplicate_publisher":
            args = [
                "python3",
                str(ROOT / "source_fixture_endpoint_fault.py"),
                "--inventory-sha256",
                settings["inventory_sha256"],
                "--run-id",
                settings["run_id"],
            ]
            save("fault-endpoint-argv.json", args)
            log = (OUT / "fault-endpoint-original.log").open("x")
            witness_logs.append(log)
            auxiliary = subprocess.Popen(
                args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            save("fault-endpoint-process.json", process_fact(auxiliary.pid))
        elif fault in {
            "sensor_stop",
            "clock_freeze",
            "tf_loss",
            "nav2_inactive",
            "qos_mismatch",
            "namespace_mismatch",
        }:
            args = [
                "python3",
                str(ROOT / "source_readiness_fixture_fault.py"),
                "--inventory-sha256",
                settings["inventory_sha256"],
                "--run-id",
                settings["run_id"],
                "--mode",
                fault,
            ]
            save("readiness-fault-argv.json", args)
            log = (OUT / "readiness-fault-original.log").open("x")
            witness_logs.append(log)
            auxiliary = subprocess.Popen(
                args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            save("readiness-fault-process.json", process_fact(auxiliary.pid))
            spin_until(lambda: (OUT / "readiness-fault-applied.json").exists(), 15)
        if fault != "none":
            end = time.monotonic() + 3
            spin_until(lambda: time.monotonic() >= end, 5)
        if settings.get("live_dual_supervision", False):
            daemon_root = OUT / "supervision"
            (daemon_root / "home").mkdir(parents=True)
            shutil.copytree(compiled / "body", daemon_root / "home/body")
            body_config = json.loads((compiled / "execution_config.json").read_text())
            daemon_config = {
                "schema_version": "rosclaw.passive_daemon_fixture.v1",
                **body_config,
                "executor_kind": "passive_observation",
                "run_id": settings["run_id"],
                "manifest_hash": manifest["manifest_hash"],
            }
            (daemon_root / "execution_config.json").write_text(
                json.dumps(daemon_config, indent=2) + "\n"
            )
            daemon_argv = [
                sys.executable,
                str(ROOT / "daemon.py"),
                "--directory",
                str(daemon_root),
                "--persistent-ledger",
            ]
            save("passive-daemon-argv.json", daemon_argv)
            passive_log = (OUT / "passive-daemon-original.log").open("x")
            passive_daemon = subprocess.Popen(
                daemon_argv, stdout=passive_log, stderr=subprocess.STDOUT, start_new_session=True
            )
            spin_until(
                lambda: (daemon_root / f"daemon-ready-{passive_daemon.pid}.json").exists(), 20
            )
            save("passive-daemon-process.json", process_fact(passive_daemon.pid))
        save("compiled-witness-liveness.json", {"alive": witness_alive})
        save("latest-native.json", state["native"])
        save("latest-system.json", state["model"].to_dict())
        save("latest-navigation-review.json", state["review"])
        save("latest-physics-witness.json", state["physics"])
        save("all-navigation-reviews.json", reviews)
        save("observer-process-facts.json", observer_facts())
        save("compiled-witness-process.json", witness_process)
        save(
            "capture_ready.json",
            {
                "manifest_hash": manifest["manifest_hash"],
                "body_snapshot_hash": manifest["body_snapshot_hash"],
                "native_samples": len(reviews),
            },
        )
        live_seconds = settings.get("live_continuity_hold_seconds", 0)
        if type(live_seconds) is not int or not 0 <= live_seconds <= 120:
            raise ValueError("bounded read-only continuity supervision window required")
        spin_until(lambda: (OUT / "host_capture_ack.json").exists(), 20 + live_seconds)
        print(
            json.dumps(
                {
                    "status": "ACTUAL_SOURCE_CAPTURE_COMPLETE_EXTERNAL_REVIEW_REQUIRED",
                    "native_samples": len(reviews),
                    "compiled_body": manifest["body_snapshot_hash"],
                }
            ),
            flush=True,
        )
    finally:
        if consumer_witness:
            alive_at_shutdown = consumer_witness.poll() is None
            stop(consumer_witness)
            save(
                "costmap-consumer-owned-exit.json",
                {
                    "alive_before_operator_shutdown": alive_at_shutdown,
                    "returncode": consumer_witness.returncode,
                    "monotonic_ns": time.monotonic_ns(),
                    "physical_stop_claimed": False,
                },
            )
        if consumer_log:
            consumer_log.close()
        if passive_daemon:
            stop(passive_daemon)
            save("passive-daemon-exit.json", {"returncode": passive_daemon.returncode})
        if passive_log:
            passive_log.close()
        if auxiliary:
            stop(auxiliary)
            save("auxiliary-process-exit.json", {"returncode": auxiliary.returncode})
        if witness:
            stop(witness)
        if governed_sink:
            stop(governed_sink)
            save("governed-sink-exit.json", {"returncode": governed_sink.returncode})
        if governed_log:
            governed_log.close()
        stop(stack)
        stack_log.close()
        for log in witness_logs:
            log.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
