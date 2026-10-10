"""Isolated rosclawd acceptance process; no execution Runtime in the agent."""

import argparse
import hashlib
import json
import logging
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

from experiments import validate_continuous_boundary_experiment, validate_inner_ring_experiment
from rosclaw.connectors.ros.action_client import Ros2ActionClient
from rosclaw.connectors.ros.mission.executor import RosCoverageSimulationExecutor, SimulationWitness
from rosclaw.connectors.ros.mission.remember import VerifiedMissionMemoryExecutor
from rosclaw.connectors.ros.practice import RosPracticeAdapter
from rosclaw.connectors.ros.transport.base import RosbridgeEndpoint
from rosclaw.connectors.ros.transport.rosbridge import RosbridgeTransport
from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.daemon.ledger import DaemonLedger
from rosclaw.daemon.server import RosclawDaemon
from rosclaw.daemon.service import DaemonControlPlane
from rosclaw.kernel import ExecutionMode
from rosclaw.practice.recorder import PracticeRecorder
from rosclaw.runtime.bus import RuntimeBus
from rosclaw.runtime.event import RuntimeEvent


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--endpoint", default="ws://127.0.0.1:19090")
    parser.add_argument("--persistent-ledger", action="store_true")
    args = parser.parse_args()
    root = args.directory.resolve()
    config = json.loads((root / "execution_config.json").read_text())
    validate_inner_ring_experiment(config.get("experiment", {}))
    validate_continuous_boundary_experiment(config.get("experiment", {}))
    runtime = Runtime(
        RuntimeConfig(
            robot_id=config["body_id"],
            workspace_home=str(root / "home"),
            enable_firewall=False,
            enable_memory=True,
            enable_practice=False,
            enable_skill_manager=False,
            enable_knowledge=False,
            enable_how=False,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
            enable_recovery_loop=False,
            enable_event_persistence=False,
            enable_tracing=False,
            seekdb_backend="sqlite",
            seekdb_path=str(root / "memory.sqlite"),
        )
    )
    runtime.initialize()
    runtime.start()
    practice = RosPracticeAdapter(runtime.event_bus)
    practice.initialize()
    recorder_bus = RuntimeBus(event_bus=runtime.event_bus)
    recorder_bus.robot_id = config["body_id"]
    recorder = PracticeRecorder(
        recorder_bus, data_root=root / "practice", publish_to_event_bus=False
    )
    recorder.initialize()
    recorder.start()
    recorder_bus.publish(
        RuntimeEvent(
            type="practice.start",
            source="runtime",
            robot=config["body_id"],
            body_id=config["body_id"],
            payload={
                "practice_id": "gazebo-room-cleaning",
                "episode_id": "gazebo-room-cleaning",
                "metadata": {"evidence_domain": "SIMULATION", "engine": "gazebo"},
            },
        )
    )
    transports = [RosbridgeTransport(RosbridgeEndpoint.from_url(args.endpoint)) for _ in range(4)]
    for transport in transports:
        deadline = time.monotonic() + 20
        while True:
            result = transport.connect()
            if result.ok:
                break
            if time.monotonic() > deadline:
                raise ConnectionError(result.error)
            time.sleep(0.2)
    client = Ros2ActionClient(transports[0])
    witness = SimulationWitness(transports[1])
    deadline = time.monotonic() + 20
    while True:
        try:
            if witness.fresh()["observation_complete"]:
                break
        except RuntimeError:
            pass
        if time.monotonic() > deadline:
            raise RuntimeError("complete independent Gazebo observer did not become ready")
        time.sleep(0.1)
    executor = RosCoverageSimulationExecutor(
        owner="daemon_ros_expert_gazebo",
        client=client,
        control=transports[2],
        witness=witness,
        output=root / "actions",
        body_id=config["body_id"],
        body_snapshot_hash=config["body_snapshot_hash"],
        grid=config["grid"],
        recovery_centers=config["recovery_centers"],
        lease_control=transports[3],
        audit_metadata=freeze_audit_source(root, config),
        boundary_pass=config.get("experiment", {}).get("boundary_pass", False),
        boundary_strategy=config.get("experiment", {}).get("boundary_strategy", "through_poses"),
        boundary_centers=config.get("boundary_centers"),
        boundary_tracking_bt_sha256=config.get("experiment", {}).get("boundary_tracking_bt_sha256"),
        repair_strategy=config.get("repair_strategy", "greedy"),
        repair_swath_yaw=config.get("experiment", {})
        .get("planning_parameters", {})
        .get("default_swath_angle", 0.0),
    )
    for capability in [
        "navigation.navigate_to_pose",
        "coverage.execute",
        "localization.set_initial_pose",
    ]:
        runtime.action_gateway.register_executor(capability, ExecutionMode.SIMULATION, executor)
    runtime.register_driver("gazebo_ros_expert", executor)
    runtime.action_gateway.register_executor(
        "ros.expert.remember",
        ExecutionMode.SIMULATION,
        VerifiedMissionMemoryExecutor(
            runtime, evidence_directory=root / "actions", recorder_bus=recorder_bus
        ),
    )
    state = root / "state" / ("persistent" if args.persistent_ledger else uuid.uuid4().hex)
    ledger = DaemonLedger(state / "control.sqlite") if args.persistent_ledger else None
    daemon = RosclawDaemon(
        service=DaemonControlPlane(runtime=runtime, state_dir=state, ledger=ledger),
        socket_path=root / "run/rosclawd.sock",
    )
    stopped = threading.Event()
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, lambda *_: stopped.set())
    try:
        daemon.start()
        (root / "daemon_ready.json").write_text(
            json.dumps({"pid": __import__("os").getpid(), "body_id": config["body_id"]})
        )
        stopped.wait()
    finally:
        executor.emergency_stop()
        daemon.stop()
        witness.close()
        client.close()
        transports[2].close()
        transports[3].close()
        if recorder.session is not None:
            receipt = runtime.action_gateway.get_receipt("golden-coverage")
            final_state = receipt.final_state.value if receipt is not None else "INTERRUPTED"
            recorder_bus.publish(
                RuntimeEvent(
                    type="practice.stop",
                    source="runtime",
                    robot=config["body_id"],
                    body_id=config["body_id"],
                    payload={
                        "outcome": "PARTIAL" if final_state == "DEGRADED" else "FAILURE",
                        "failure_labels": ["coverage:" + final_state],
                    },
                )
            )
        practice.stop()
        recorder.stop()
        runtime.stop()
        if ledger is not None:
            ledger.close()


def freeze_audit_source(root, config):
    """Freeze only public source/config bytes; never ledger keys or model auth."""
    repository = Path(__file__).resolve().parents[3]
    files = [
        Path(__file__),
        repository / "src/rosclaw/connectors/ros/mission/executor.py",
        repository / "src/rosclaw/connectors/ros/mission/boundary_pass.py",
        repository / "src/rosclaw/connectors/ros/mission/repair_optimizer.py",
        repository / "src/rosclaw/connectors/ros/diagnosis/coverage_audit.py",
        repository / "src/rosclaw/connectors/ros/verification/coverage.py",
        *[
            Path(__file__).with_name(name)
            for name in [
                "stack.py",
                "witness.py",
                "profiles.py",
                "run.py",
                "nav2_launch.py",
                "experiments.py",
                "paired_efficiency.py",
                "cleaning_acceptance.py",
                "precise_through_poses_bt.py",
            ]
        ],
        root / "nav2.yaml",
        root / "execution_config.json",
        root / "robot.urdf",
        root / "measured_map.json",
        root / "world.sdf",
        root / "experiment.json",
        root / "protocol.json",
        *[
            root / name
            for name in (
                "repair-through-poses.original.xml",
                "repair-through-poses.xml",
                "repair-through-poses-source.json",
                "boundary-through-poses.original.xml",
                "boundary-through-poses.xml",
                "boundary-through-poses-source.json",
            )
        ],
    ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repository))
    metadata = {
        "run_id": (root / "run_id.txt").read_text().strip()
        if (root / "run_id.txt").exists()
        else root.name,
        "git_sha": commit,
        "working_tree_dirty": dirty,
        "source_hashes": hashes,
        "map_hash": hashes.get(str(root / "measured_map.json")),
        "geometry_hash": hashes.get(str(root / "robot.urdf")),
        "planner_revision": "65a6598c3587cb947978227c01af421e18576f0a",
    }
    (root / "source-freeze.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Keep high-rate events compact; reference the immutable source manifest.
    return {k: v for k, v in metadata.items() if k != "source_hashes"}


if __name__ == "__main__":
    main()
