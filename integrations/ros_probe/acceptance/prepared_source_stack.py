"""SDK-owned bringup from pinned prepared assets, without robot profiles.

No Runtime, navigation client, velocity publisher or lease is created here.
The external source witness and governed sink are separate owned processes.
"""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import yaml
from gazebo_source_witness.prepare import check_prepared_assets

from experiments import gazebo_arguments

OUT = Path("/evidence")
ROOT = Path(__file__).resolve().parent


def navigation_stack_enabled(descriptor, settings):
    start_navigation = descriptor.get("start_navigation_stack", True)
    if type(start_navigation) is not bool:
        raise ValueError("explicit boolean navigation bringup required")
    if not start_navigation:
        capture = settings.get("engineering_capture", {})
        if (
            not Path("/.dockerenv").is_file()
            or os.environ.get("ROS_LOCALHOST_ONLY") != "1"
            or capture.get("case") != "driver_onboarding"
            or capture.get("phase") != "before_repair"
            or capture.get("run_id") != settings["run_id"]
            or "governed_sink" not in settings
        ):
            raise ValueError("driver-only bringup requires closed isolated unadmitted SDK role")
    elif settings.get("engineering_capture", {}).get("case") == "driver_onboarding":
        raise ValueError("driver onboarding cannot start Nav2")
    return start_navigation


def main():
    settings = json.loads(Path("/settings.json").read_text())
    check_prepared_assets("/assets", settings["inventory_sha256"], OUT)
    if (OUT / "run_id.txt").read_text().strip() != settings["run_id"]:
        raise ValueError("pinned prepared assets differ from this actual SIM run")
    descriptor = json.loads(Path("/assets/bringup.json").read_text())
    if (
        set(descriptor)
        - {
            "scan_bridge_state",
            "additional_command_endpoint",
            "counterfactual_sim",
            "start_navigation_stack",
            "initialization_boundary",
        }
        != {
            "schema_version",
            "namespace",
            "controllers",
            "scan_ros_topic",
            "autostart",
            "tf_topics_relative",
        }
        or descriptor["schema_version"] != "rosclaw.prepared_source_bringup.v1"
        or type(descriptor["namespace"]) is not str
        or type(descriptor["autostart"]) is not bool
        or type(descriptor["tf_topics_relative"]) is not bool
        or set(descriptor["controllers"]) != {"manager", "names"}
        or type(descriptor["controllers"]["names"]) is not list
        or not descriptor["controllers"]["names"]
    ):
        raise ValueError("closed pinned operator bringup descriptor required")
    if descriptor.get("initialization_boundary") not in (
        None,
        "LOCALIZATION_ACTIVE_BEFORE_NAVIGATION",
    ):
        raise ValueError("explicit supported initialization boundary required")
    start_navigation = navigation_stack_enabled(descriptor, settings)
    if "counterfactual_sim" in descriptor:
        from simulation_counterfactual_config import counterfactual_clock_parameters

        counterfactual_clock_parameters(descriptor, "controller_server")
        capture = settings.get("engineering_capture", {})
        if (
            not Path("/.dockerenv").is_file()
            or os.environ.get("ROS_LOCALHOST_ONLY") != "1"
            or capture.get("case") != "rsk_node_clock_mismatch"
            or capture.get("phase") not in {"before_repair", "rollback"}
            or capture.get("run_id") != settings["run_id"]
            or "governed_sink" not in settings
        ):
            raise ValueError(
                "clock override requires an explicitly owned isolated counterfactual capture"
            )
    scan_state = descriptor.get("scan_bridge_state", "running")
    if scan_state not in {"running", "paused_after_initialization"}:
        raise ValueError("explicit engineering scan bridge state required")
    if (
        scan_state != "running"
        and settings.get("engineering_capture", {}).get("case") != "scan_stale"
    ):
        raise ValueError("paused scan bridge requires explicit engineering capture role")
    extra = descriptor.get("additional_command_endpoint")
    if extra is not None and (
        type(extra) is not dict
        or set(extra) != {"enabled", "topic", "msg_type"}
        or type(extra["enabled"]) is not bool
        or type(extra["topic"]) is not str
        or not extra["topic"].startswith("/")
        or extra["msg_type"] != "geometry_msgs/msg/Twist"
    ):
        raise ValueError("closed additional command endpoint declaration required")
    if (
        extra
        and extra["enabled"]
        and settings.get("engineering_capture", {}).get("case") != "duplicate_velocity_route"
    ):
        raise ValueError("extra endpoint requires explicit engineering capture role")
    children, logs, owned = [], [], {}

    def start(name, argv):
        log = (OUT / f"{name}.log").open("x")
        logs.append(log)
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        children.append(process)
        owned[name] = {"pid": process.pid, "argv": argv}
        (OUT / "owned-stack-processes.json").write_text(json.dumps(owned, indent=2) + "\n")
        return process

    def interrupt(*_args):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        start("gazebo", gazebo_arguments(OUT / "world.sdf", settings["seed"]))
        time.sleep(3)
        bridge = yaml.safe_load((OUT / "bridge.yaml").read_text())
        scan = [x for x in bridge if x.get("ros_topic_name") == descriptor["scan_ros_topic"]]
        if len(scan) != 1:
            raise ValueError("one declared original scan bridge required")
        (OUT / "scan-bridge.yaml").write_text(yaml.safe_dump(scan))
        (OUT / "main-bridge.yaml").write_text(yaml.safe_dump([x for x in bridge if x not in scan]))
        for name, file in (("scan_bridge", "scan-bridge.yaml"), ("bridge", "main-bridge.yaml")):
            start(
                name,
                [
                    "ros2",
                    "run",
                    "ros_gz_bridge",
                    "parameter_bridge",
                    "--ros-args",
                    "-p",
                    f"config_file:={OUT / file}",
                    "-p",
                    "use_sim_time:=true",
                ],
            )
        state_argv = [
            "ros2",
            "run",
            "robot_state_publisher",
            "robot_state_publisher",
            "--ros-args",
            "--params-file",
            str(OUT / "robot-state-params.yaml"),
            "-r",
            "__ns:=" + descriptor["namespace"],
        ]
        if descriptor["tf_topics_relative"]:
            state_argv += ["-r", "/tf:=tf", "-r", "/tf_static:=tf_static"]
        start("robot_state", state_argv)
        controllers = descriptor["controllers"]
        spawner = start(
            "controller_spawner",
            [
                "ros2",
                "run",
                "controller_manager",
                "spawner",
                *controllers["names"],
                "--controller-manager",
                controllers["manager"],
                "--controller-manager-timeout",
                "35",
            ],
        )
        if spawner.wait(timeout=45) != 0:
            raise RuntimeError("declared source controllers did not initialize")
        children.remove(spawner)
        if start_navigation:
            start("nav2", ["ros2", "launch", str(ROOT / "nav2_launch.py")])
        if start_navigation and Path("/assets/localization-refresh.json").is_file():
            if "governed_sink" not in settings:
                raise ValueError("localization refresh requires explicit governed SIM fixture")
            start(
                "localization_refresh", ["python3", str(ROOT / "localization_refresh_fixture.py")]
            )
        probe_argv = [
            "python3",
            str(ROOT.parent / "ros2/probe.py"),
            "--snapshot-period-sec",
            str(settings.get("probe_snapshot_period_sec", 1.0)),
        ]
        mesh_manifest = Path("/assets/collision-asset-originals.json")
        if mesh_manifest.is_file():
            import hashlib

            probe_argv += [
                "--collision-asset-originals",
                str(mesh_manifest),
                "--collision-asset-originals-sha256",
                hashlib.sha256(mesh_manifest.read_bytes()).hexdigest(),
            ]
        start("probe", probe_argv)
        bridge_command = ["ros2", "run", "rosbridge_server", "rosbridge_websocket"]
        if settings.get("governed_sink") is not None:
            sources = json.loads(Path("/assets/observer-sources.json").read_text())
            if sources["run_id"] != settings["run_id"]:
                raise ValueError("declared SDK lease source run changed")
            bridge_command = [
                "python3",
                str(ROOT / "simulation_persistent_lease_bridge.py"),
                "--lease-service",
                sources["topics"]["lease"],
            ]
        start(
            "rosbridge",
            [
                *bridge_command,
                "--ros-args",
                "-p",
                "port:=9090",
                "-p",
                "use_sim_time:=true",
            ],
        )
        start(
            "rosapi",
            ["ros2", "run", "rosapi", "rosapi_node", "--ros-args", "-p", "use_sim_time:=true"],
        )
        if extra and extra["enabled"]:
            start(
                "engineering_extra_endpoint",
                ["python3", str(ROOT / "simulation_engineering_endpoint_fixture.py")],
            )
        # This is only a process-spawn milestone, never ROS/physics readiness.
        # The capture owner separately requires actual ACTIVE lifecycle and
        # complete physical observations after the components have started.
        (OUT / "prepared-stack-spawned-original.json").write_text(
            json.dumps(
                {
                    "run_id": settings["run_id"],
                    "monotonic_ns": time.monotonic_ns(),
                    "owner_pid": os.getpid(),
                    "children": owned,
                    "ROS_readiness": "NOT_EVALUATED",
                    "execution_authority": "NONE",
                },
                indent=2,
            )
            + "\n"
        )
        while True:
            for process in children:
                if process.poll() is not None:
                    raise RuntimeError("owned prepared source process exited: " + str(process.args))
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        for process in children:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for process in children:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
