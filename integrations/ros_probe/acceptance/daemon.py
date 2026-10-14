"""Isolated rosclawd acceptance process; no execution Runtime in the agent."""

import argparse
import hashlib
import json
import logging
import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

from rosclaw.body.resolver import BodyResolver
from rosclaw.connectors.ros.action_client import Ros2ActionClient
from rosclaw.connectors.ros.mission.executor import RosCoverageSimulationExecutor, SimulationWitness
from rosclaw.connectors.ros.mission.navigation_binding import SimulationNavigationBinding
from rosclaw.connectors.ros.mission.remember import VerifiedMissionMemoryExecutor
from rosclaw.connectors.ros.practice import RosPracticeAdapter
from rosclaw.connectors.ros.transport.base import RosbridgeEndpoint
from rosclaw.connectors.ros.transport.rosbridge import RosbridgeTransport
from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.daemon.ledger import DaemonLedger
from rosclaw.daemon.protocol import PeerCredentials
from rosclaw.daemon.server import RosclawDaemon
from rosclaw.daemon.service import DaemonControlPlane
from rosclaw.kernel import ExecutionMode
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
    kind = config.get("executor_kind", "coverage")
    if kind == "passive_observation":
        serve_passive_observation(root, config, persistent_ledger=args.persistent_ledger)
        return
    from rosclaw.practice.recorder import PracticeRecorder

    if kind not in {"coverage", "navigation_only", "governed_navigation"}:
        raise ValueError("unsupported simulation executor kind")
    navigation_binding = None
    governed_facts = governed_context = p03_raw = None
    if kind in {"navigation_only", "governed_navigation"}:
        resolver = BodyResolver(workspace=root / "home", body_id=config["body_id"])
        if not resolver.effective_body_path.is_file():
            raise ValueError("navigation-only executor requires an already compiled Body")
        effective = resolver.get_effective_body(recompile_if_stale=False)
        navigation_binding = SimulationNavigationBinding.from_body(effective)
        if navigation_binding.body_snapshot_hash != config["body_snapshot_hash"]:
            raise ValueError("navigation execution configuration Body hash mismatch")
        if any(k in config for k in ("grid", "recovery_centers", "boundary_centers", "experiment")):
            raise ValueError(
                "navigation-only configuration must not contain coverage configuration"
            )
        if len(os.fsencode(root / "run/rosclawd.sock")) >= 104:
            raise ValueError("use a shorter isolated directory for the rosclawd Unix socket")
        if kind == "governed_navigation":
            from simulation_current_facts import CurrentSimulationFacts, _read
            from simulation_sink_admission import verify_p03_prerequisite

            if (
                set(config)
                != {
                    "schema_version",
                    "executor_kind",
                    "body_id",
                    "body_snapshot_hash",
                    "supervision",
                }
                or config["schema_version"] != "rosclaw.governed_navigation_fixture.v1"
                or not args.persistent_ledger
                or type(config["supervision"]) is not dict
                or set(config["supervision"])
                != {
                    "schema_version",
                    "facts_enrollment_sha256",
                    "P03_review_sha256",
                    "allowed_action_ids",
                }
                or config["supervision"]["schema_version"]
                != "rosclaw.governed_navigation_supervision.v1"
            ):
                raise ValueError("closed governed SIM role and persistent control ledger required")
            governed_context = config["supervision"]
            actions = governed_context["allowed_action_ids"]
            if (
                type(actions) is not list
                or not 1 <= len(actions) <= 2
                or any(type(x) is not str for x in actions)
                or len(set(actions)) != len(actions)
            ):
                raise ValueError("one or two unique operator-scoped action IDs required")
            p03_raw = _read(Path("/evidence/P03-accepted-review.json"), 128_000)
            verify_p03_prerequisite(p03_raw, expected_sha256=governed_context["P03_review_sha256"])
            governed_facts = CurrentSimulationFacts(
                Path("/evidence"),
                assets_directory=Path("/assets"),
                enrollment_sha256=governed_context["facts_enrollment_sha256"],
            )
            if (
                governed_facts.enrollment["request"]["body_snapshot_hash"]
                != navigation_binding.body_snapshot_hash
                or governed_facts.enrollment["request"]["sim_run_id"] != navigation_binding.run_id
            ):
                governed_facts.close()
                raise ValueError("governed daemon differs from enrolled actual Body/run")
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
                "practice_id": "gazebo-navigation"
                if navigation_binding
                else "gazebo-room-cleaning",
                "episode_id": "gazebo-navigation" if navigation_binding else "gazebo-room-cleaning",
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
    witness = SimulationWitness(transports[1], navigation_binding=navigation_binding)
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
        grid=None if navigation_binding else config["grid"],
        recovery_centers=() if navigation_binding else config["recovery_centers"],
        lease_control=transports[3],
        audit_metadata=freeze_audit_source(root, config),
        boundary_pass=config.get("experiment", {}).get("boundary_pass", False),
        boundary_strategy=config.get("experiment", {}).get("boundary_strategy", "through_poses"),
        boundary_centers=config.get("boundary_centers"),
        navigation_binding=navigation_binding,
    )
    capabilities = (
        ["navigation.navigate_to_pose"]
        if navigation_binding
        else [
            "navigation.navigate_to_pose",
            "coverage.execute",
            "localization.set_initial_pose",
        ]
    )
    for capability in capabilities:
        runtime.action_gateway.register_executor(capability, ExecutionMode.SIMULATION, executor)
    runtime.register_driver("gazebo_ros_expert", executor)
    if not navigation_binding:
        runtime.action_gateway.register_executor(
            "ros.expert.remember",
            ExecutionMode.SIMULATION,
            VerifiedMissionMemoryExecutor(
                runtime, evidence_directory=root / "actions", recorder_bus=recorder_bus
            ),
        )
    state = root / "state" / ("persistent" if args.persistent_ledger else uuid.uuid4().hex)
    ledger = DaemonLedger(state / "control.sqlite") if args.persistent_ledger else None
    service = DaemonControlPlane(runtime=runtime, state_dir=state, ledger=ledger)
    supervision = None
    if kind == "governed_navigation":
        from simulation_current_facts import _read
        from simulation_execution_proof import ExecutionProofPins
        from simulation_lease_observation import ExternalSimulationLeaseObservation
        from simulation_navigation_supervision import SdkNavigationExecutionSupervision

        control = Path("/evidence/execution-control")
        key_info = (control / "fixture.key").stat(follow_symlinks=False)
        if key_info.st_uid != os.getuid() or key_info.st_mode & 0o077:
            raise ValueError("private operator SDK context key required")
        peer = PeerCredentials(os.getpid(), os.getuid(), os.getgid())
        instance = service.get_runtime_status(peer)["daemon_instance_id"]
        # Enrollment occurs before the socket starts serving any action. The
        # SDK operator policy is closed; action arguments cannot replace it.
        for action_id in governed_context["allowed_action_ids"]:
            ExecutionProofPins(
                body_id=navigation_binding.body_id,
                body_snapshot_hash=navigation_binding.body_snapshot_hash,
                manifest_hash=governed_facts.enrollment["request"]["manifest_hash"],
                run_id=navigation_binding.run_id,
                world_sha256=governed_facts.enrollment["world_sha256"],
                observer_epoch=governed_facts.enrollment["source_pins"]["observer_epoch"],
                daemon_instance_id=instance,
                action_id=action_id,
            )
        supervision = SdkNavigationExecutionSupervision(
            binding=navigation_binding,
            facts=governed_facts,
            daemon_instance_id=instance,
            daemon_status=lambda: service.get_runtime_status(peer),
            key=_read(control / "fixture.key", 32),
            proof_file=control / "proof.json",
            audit_file=Path("/evidence") / f"daemon-supervision-{instance}-original.jsonl",
            lease_observation=ExternalSimulationLeaseObservation(
                Path("/evidence"),
                filename="governed-sink-status.json",
                run_id=navigation_binding.run_id,
                body_snapshot_hash=navigation_binding.body_snapshot_hash,
            ),
            p03_review_raw=p03_raw,
            p03_review_sha256=governed_context["P03_review_sha256"],
            allowed_action_ids=tuple(governed_context["allowed_action_ids"]),
        )
        executor.navigation_supervision = supervision
    daemon = RosclawDaemon(
        service=service,
        socket_path=root / "run/rosclawd.sock",
    )
    stopped = threading.Event()
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, lambda *_: stopped.set())
    try:
        daemon.start()
        ready = {"pid": os.getpid(), "body_id": config["body_id"]}
        if kind == "governed_navigation":
            ready.update(
                schema_version="rosclaw.governed_daemon_ready.v1",
                body_snapshot_hash=navigation_binding.body_snapshot_hash,
                run_id=navigation_binding.run_id,
                daemon_instance_id=instance,
                manifest_hash=governed_facts.enrollment["request"]["manifest_hash"],
                uid=os.getuid(),
                execution_config_sha256=hashlib.sha256(
                    (root / "execution_config.json").read_bytes()
                ).hexdigest(),
            )
        (root / "daemon_ready.json").write_text(json.dumps(ready))
        stopped.wait()
    finally:
        executor.emergency_stop()
        daemon.stop()
        if supervision is not None:
            supervision.close()
        witness.close()
        client.close()
        transports[2].close()
        transports[3].close()
        if recorder.session is not None:
            receipt = runtime.action_gateway.get_receipt(
                "golden-navigation" if navigation_binding else "golden-coverage"
            )
            final_state = receipt.final_state.value if receipt is not None else "INTERRUPTED"
            recorder_bus.publish(
                RuntimeEvent(
                    type="practice.stop",
                    source="runtime",
                    robot=config["body_id"],
                    body_id=config["body_id"],
                    payload={
                        "outcome": "SUCCESS"
                        if navigation_binding and final_state == "COMPLETED"
                        else "PARTIAL"
                        if final_state == "DEGRADED"
                        else "FAILURE",
                        "failure_labels": []
                        if navigation_binding and final_state == "COMPLETED"
                        else [("navigation:" if navigation_binding else "coverage:") + final_state],
                    },
                )
            )
        practice.stop()
        recorder.stop()
        runtime.stop()
        if ledger is not None:
            ledger.close()


def serve_passive_observation(root, config, *, persistent_ledger):
    """Existing SDK rosclawd, with no ROS transports/executors/drivers/leases.

    This role only binds an independently compiled Body/run to this actual
    daemon generation. It does not admit SIM actions or enable the generic sink.
    """
    keys = {
        "schema_version",
        "executor_kind",
        "body_id",
        "body_snapshot_hash",
        "run_id",
        "manifest_hash",
    }
    if (
        set(config) != keys
        or not persistent_ledger
        or config["schema_version"] != "rosclaw.passive_daemon_fixture.v1"
        or len(os.fsencode(root / "run/rosclawd.sock")) >= 104
    ):
        raise ValueError("closed passive daemon configuration and persistent ledger required")
    resolver = BodyResolver(workspace=root / "home", body_id=config["body_id"])
    effective = resolver.get_effective_body(recompile_if_stale=False)
    binding = SimulationNavigationBinding.from_body(effective)
    if (
        binding.body_snapshot_hash != config["body_snapshot_hash"]
        or binding.run_id != config["run_id"]
    ):
        raise ValueError("passive daemon immutable Body/run mismatch")
    body_raw = resolver.effective_body_path.read_bytes()
    runtime = Runtime(
        RuntimeConfig(
            robot_id=config["body_id"],
            workspace_home=str(root / "home"),
            enable_firewall=False,
            enable_memory=False,
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
    state = root / "state" / "passive"
    ledger = DaemonLedger(state / "control.sqlite")
    service = DaemonControlPlane(runtime=runtime, state_dir=state, ledger=ledger)
    daemon = RosclawDaemon(service=service, socket_path=root / "run/rosclawd.sock")
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    try:
        daemon.start()
        status = service.get_runtime_status(PeerCredentials(os.getpid(), os.getuid(), os.getgid()))
        builtin_sandbox = {"sandbox.reach:SIMULATION", "sandbox.reach:FIXTURE"}
        if status["drivers"] or set(status["registered_executors"]) - builtin_sandbox:
            raise RuntimeError("passive rosclawd must have zero robot drivers/executors")
        ready = {
            **config,
            "pid": os.getpid(),
            "uid": os.getuid(),
            "daemon_instance_id": status["daemon_instance_id"],
            "effective_body_file_sha256": hashlib.sha256(body_raw).hexdigest(),
            "execution_config_sha256": hashlib.sha256(
                (root / "execution_config.json").read_bytes()
            ).hexdigest(),
            "source_only": True,
            "builtin_sandbox_executors": status["registered_executors"],
            "L2": "NOT_RUN",
            "firewall_fixture_enabled": False,
        }
        with (root / f"daemon-ready-{os.getpid()}.json").open("x") as stream:
            stream.write(json.dumps(ready, indent=2) + "\n")
        stopped.wait()
    finally:
        daemon.stop()
        runtime.stop()
        ledger.close()


def freeze_audit_source(root, config):
    """Freeze only public source/config bytes; never ledger keys or model auth."""
    repository = Path(__file__).resolve().parents[3]
    files = [
        Path(__file__),
        repository / "src/rosclaw/connectors/ros/mission/executor.py",
        repository / "src/rosclaw/connectors/ros/mission/navigation_binding.py",
        repository / "src/rosclaw/connectors/ros/context/navigation_readiness.py",
        repository / "src/rosclaw/connectors/ros/mission/boundary_pass.py",
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
                "simulation_current_facts.py",
                "simulation_current_process.py",
                "simulation_atomic_read.py",
                "simulation_daemon_observation.py",
                "simulation_velocity_gate.py",
                "prepared_source_stack.py",
                "localization_refresh_fixture.py",
                "simulation_execution_proof.py",
                "simulation_navigation_supervision.py",
                "simulation_governed_sink.py",
                "simulation_sink_admission.py",
                "simulation_lease_observation.py",
                "simulation_source_stream.py",
                "simulation_source_continuity.py",
            ]
        ],
        root / "nav2.yaml",
        root / "execution_config.json",
        root / "robot.urdf",
        root / "measured_map.json",
        root / "world.sdf",
        root / "experiment.json",
        root / "protocol.json",
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
        "planner_revision": None
        if config.get("executor_kind") == "governed_navigation"
        else "65a6598c3587cb947978227c01af421e18576f0a",
    }
    if config.get("executor_kind") == "governed_navigation":
        import xml.etree.ElementTree as ET

        package = Path("/opt/ros/jazzy/share/nav2_planner/package.xml")
        metadata["navigation_adapter_revision"] = commit
        metadata["planner_package_version"] = (
            ET.parse(package).getroot().findtext("version") if package.is_file() else None
        )
    (root / "source-freeze.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Keep high-rate events compact; reference the immutable source manifest.
    return {k: v for k, v in metadata.items() if k != "source_hashes"}


if __name__ == "__main__":
    main()
