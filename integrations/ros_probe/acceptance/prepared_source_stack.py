"""SDK-owned bringup from pinned prepared assets, without robot profiles.

No Runtime, navigation client, velocity publisher or lease is created here.
The external source witness and governed sink are separate owned processes.
"""

import json
import signal
import subprocess
import time
from pathlib import Path

import yaml
from gazebo_source_witness.prepare import check_prepared_assets

from experiments import gazebo_arguments

OUT = Path("/evidence")
ROOT = Path(__file__).resolve().parent


def main():
    settings = json.loads(Path("/settings.json").read_text())
    check_prepared_assets("/assets", settings["inventory_sha256"], OUT)
    if (OUT / "run_id.txt").read_text().strip() != settings["run_id"]:
        raise ValueError("pinned prepared assets differ from this actual SIM run")
    descriptor = json.loads(Path("/assets/bringup.json").read_text())
    if (
        set(descriptor)
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
        start("nav2", ["ros2", "launch", str(ROOT / "nav2_launch.py")])
        start(
            "probe",
            [
                "python3",
                str(ROOT.parent / "ros2/probe.py"),
                "--snapshot-period-sec",
                str(settings.get("probe_snapshot_period_sec", 1.0)),
            ],
        )
        start(
            "rosbridge",
            [
                "ros2",
                "run",
                "rosbridge_server",
                "rosbridge_websocket",
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
