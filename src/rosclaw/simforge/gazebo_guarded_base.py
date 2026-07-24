"""Gazebo GuardedBase canonical-path and real process-chaos experiment.

This module never targets a real robot. It owns one exact Docker container,
runs Gazebo Fortress plus ROS 2 bridges, and sends motion only through:

    MCP RuntimeClient -> rosclawd -> ActionGateway -> daemon-owned ROS sink
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosclaw.connectors.ros.transport import RosbridgeEndpoint, RosbridgeTransport
from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.daemon.client import DaemonClient
from rosclaw.daemon.server import RosclawDaemon
from rosclaw.daemon.service import DaemonControlPlane
from rosclaw.kernel import ActionEnvelope, EvidenceLevel, ExecutionMode, VerificationPolicy
from rosclaw.mcp.adapters.runtime_client import RuntimeClient
from rosclaw.simforge.proof import (
    CounterfactualMetric,
    CounterfactualRun,
    FaultInjectionResult,
    ModuleEvidenceLevel,
    ModuleProof,
    ProofBundle,
)
from rosclaw.simforge.tasks.guarded_base import (
    GenericMobileBaseSimulationExecutor,
    RosbridgeMobileBaseSink,
    RosbridgeMobileBaseStopDriver,
    read_mobile_base_pose,
)

_BODY_HASH = "sha256:" + "a" * 64
_COMMAND_TOPIC = "/guarded_base/guarded_cmd_vel"
_ODOM_TOPIC = "/guarded_base/odom"
_ODOM_TYPE = "nav_msgs/msg/Odometry"
_SCAN_TOPIC = "/guarded_base/scan"
_SCAN_TYPE = "sensor_msgs/msg/LaserScan"


@dataclass(frozen=True)
class GazeboGuardedBaseResult:
    output_dir: Path
    report_path: Path
    report_hash: str
    canonical_task_verified: bool
    launch_testing_passed: bool
    agent_kill_bounded_stop: bool
    rosbridge_loss_fail_closed: bool
    recovery_no_replay: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "report_path": str(self.report_path),
            "report_hash": self.report_hash,
            "canonical_task_verified": self.canonical_task_verified,
            "launch_testing_passed": self.launch_testing_passed,
            "agent_kill_bounded_stop": self.agent_kill_bounded_stop,
            "rosbridge_loss_fail_closed": self.rosbridge_loss_fail_closed,
            "recovery_no_replay": self.recovery_no_replay,
        }


def run_gazebo_guarded_base(
    *,
    output_dir: Path,
    source_checkout: Path,
    image: str = "rosclaw/ros2-humble-gazebo:latest",
    rosbridge_port: int = 0,
) -> GazeboGuardedBaseResult:
    source = source_checkout.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    if output == source or source in output.parents:
        raise ValueError("raw Gazebo evidence must be outside the source checkout")
    output.mkdir(parents=True, exist_ok=True)
    port = rosbridge_port or _available_loopback_port()
    container = f"rosclaw-phase3-gazebo-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    endpoint_url = f"ws://127.0.0.1:{port}"
    started = False
    daemon: RosclawDaemon | None = None
    transports: list[RosbridgeTransport] = []
    partial: dict[str, Any] = {
        "schema_version": "rosclaw.gazebo_guarded_base.v1",
        "simulation_only": True,
        "real_robot_commands": 0,
        "container": container,
        "rosbridge_endpoint": endpoint_url,
        "started_at_ns": time.time_ns(),
    }
    try:
        launch_result = _run_launch_testing(
            image=image,
            source=source,
            output=output,
        )
        partial["launch_testing"] = launch_result

        _start_stack(
            image=image,
            source=source,
            output=output,
            container=container,
            host_port=port,
        )
        started = True
        required_topics = _wait_for_topics(endpoint_url, timeout_sec=30.0)
        partial["ros_graph"] = {
            "required_topics": sorted(required_topics),
            "all_present": True,
        }

        motion_transport = _transport(endpoint_url)
        stop_transport = _transport(endpoint_url)
        observation_transport = _transport(endpoint_url)
        transports.extend((motion_transport, stop_transport, observation_transport))
        sink = RosbridgeMobileBaseSink(
            motion_transport,
            daemon_owner_id="daemon_gazebo_phase3",
            command_topic=_COMMAND_TOPIC,
            pose_topic=_ODOM_TOPIC,
            pose_message_type=_ODOM_TYPE,
            observation_timeout_sec=3.0,
        )
        stop_driver = RosbridgeMobileBaseStopDriver(
            stop_transport,
            command_topic=_COMMAND_TOPIC,
            pose_topic=_ODOM_TOPIC,
            pose_message_type=_ODOM_TYPE,
            observation_timeout_sec=1.5,
            motion_sink=sink,
        )
        runtime = _runtime()
        runtime.action_gateway.register_executor(
            GenericMobileBaseSimulationExecutor.capability_id,
            ExecutionMode.SHADOW,
            GenericMobileBaseSimulationExecutor(
                sink,
                daemon_instance_id="daemon_gazebo_phase3",
            ),
        )
        runtime.register_driver("gazebo_guarded_base_stop", stop_driver)
        daemon_dir = Path(tempfile.mkdtemp(prefix="rosclaw-gazebo-daemon-"))
        daemon = RosclawDaemon(
            service=DaemonControlPlane(runtime=runtime),
            socket_path=daemon_dir / "rosclawd.sock",
        )
        daemon.start()
        client = DaemonClient(socket_path=daemon.socket_path, timeout_sec=10.0)
        mcp = RuntimeClient(
            project_root=source,
            robot_id="sim_gazebo_guarded_base",
            runtime_profile={},
            daemon_client=client,
        )

        canonical = asyncio.run(
            mcp.request_action(
                capability_id="mobile_base.guarded_move",
                arguments={
                    "linear_x_mps": 0.2,
                    "angular_z_radps": 0.0,
                    "duration_sec": 0.5,
                },
                execution_mode="SHADOW",
                body_snapshot_hash=_BODY_HASH,
                body_id="sim_gazebo_guarded_base",
                action_id="gazebo-canonical-v1",
                required_evidence="TASK_VERIFIED",
                wait_timeout_sec=8.0,
            )
        )
        canonical_receipt = _receipt(canonical)
        _require(
            canonical.get("state") == "FINISHED"
            and canonical_receipt.get("final_state") == "COMPLETED"
            and canonical_receipt.get("evidence_level") == EvidenceLevel.TASK_VERIFIED.value,
            "canonical Gazebo action did not reach TASK_VERIFIED",
        )
        laser = _read_laser(observation_transport)
        partial["canonical_path"] = {
            "route": ["MCP", "rosclawd", "ActionGateway", "daemon_owned_sink", "ROS2", "Gazebo"],
            "direct_provider_publish": False,
            "action_id": "gazebo-canonical-v1",
            "receipt": canonical_receipt,
            "laser": laser,
            "passed": True,
        }

        rosbridge_loss = _run_rosbridge_loss(
            mcp=mcp,
            container=container,
            endpoint_url=endpoint_url,
            output=output,
        )
        partial["rosbridge_loss"] = rosbridge_loss
        _require(
            rosbridge_loss["maximum_evidence"] == EvidenceLevel.DISPATCH_CONFIRMED.value
            and rosbridge_loss["task_verified"] is False
            and rosbridge_loss["deadman_stop_observed"] is True,
            "rosbridge loss did not fail closed",
        )

        # A severed WebSocket is intentionally never trusted after restart.
        # Rebind all daemon-owned southbound handles to the new rosbridge
        # generation before a new Action ID is accepted.
        recovery_motion_transport = _transport(endpoint_url)
        recovery_stop_transport = _transport(endpoint_url)
        observation_transport = _transport(endpoint_url)
        transports.extend(
            (
                recovery_motion_transport,
                recovery_stop_transport,
                observation_transport,
            )
        )
        recovery_sink = RosbridgeMobileBaseSink(
            recovery_motion_transport,
            daemon_owner_id="daemon_gazebo_phase3",
            command_topic=_COMMAND_TOPIC,
            pose_topic=_ODOM_TOPIC,
            pose_message_type=_ODOM_TYPE,
            observation_timeout_sec=3.0,
        )
        recovery_stop_driver = RosbridgeMobileBaseStopDriver(
            recovery_stop_transport,
            command_topic=_COMMAND_TOPIC,
            pose_topic=_ODOM_TOPIC,
            pose_message_type=_ODOM_TYPE,
            observation_timeout_sec=1.5,
            motion_sink=recovery_sink,
        )
        runtime.action_gateway.register_executor(
            GenericMobileBaseSimulationExecutor.capability_id,
            ExecutionMode.SHADOW,
            GenericMobileBaseSimulationExecutor(
                recovery_sink,
                daemon_instance_id="daemon_gazebo_phase3",
            ),
        )
        runtime.register_driver("gazebo_guarded_base_stop", recovery_stop_driver)

        before_recovery = read_mobile_base_pose(
            observation_transport,
            topic=_ODOM_TOPIC,
            message_type=_ODOM_TYPE,
            timeout_sec=2.0,
        )
        time.sleep(0.7)
        stable = read_mobile_base_pose(
            observation_transport,
            topic=_ODOM_TOPIC,
            message_type=_ODOM_TYPE,
            timeout_sec=2.0,
        )
        _require(before_recovery is not None and stable is not None, "recovery odometry missing")
        assert before_recovery is not None and stable is not None
        no_old_effect = abs(stable[0] - before_recovery[0]) <= 0.02 and abs(stable[1]) <= 0.02
        recovered = asyncio.run(
            mcp.request_action(
                capability_id="mobile_base.guarded_move",
                arguments={
                    "linear_x_mps": 0.2,
                    "angular_z_radps": 0.0,
                    "duration_sec": 0.4,
                },
                execution_mode="SHADOW",
                body_snapshot_hash=_BODY_HASH,
                body_id="sim_gazebo_guarded_base",
                action_id="gazebo-recovery-v3",
                required_evidence="TASK_VERIFIED",
                wait_timeout_sec=8.0,
            )
        )
        recovered_receipt = _receipt(recovered)
        recovery_ok = bool(
            no_old_effect
            and recovered_receipt.get("final_state") == "COMPLETED"
            and recovered_receipt.get("evidence_level") == EvidenceLevel.TASK_VERIFIED.value
            and rosbridge_loss["action_id"] != "gazebo-recovery-v3"
        )
        _require(recovery_ok, "new-session recovery or old-action replay check failed")
        partial["recovery"] = {
            "old_action_id": rosbridge_loss["action_id"],
            "new_action_id": "gazebo-recovery-v3",
            "new_session_required": True,
            "new_permit_required_for_real": True,
            "old_effect_during_quiet_window_m": stable[0] - before_recovery[0],
            "old_action_replayed": not no_old_effect,
            "receipt": recovered_receipt,
            "passed": True,
        }

        agent_kill = _run_agent_kill(
            daemon_socket=daemon.socket_path,
            client=client,
            observation_transport=observation_transport,
        )
        partial["agent_kill"] = agent_kill
        _require(
            agent_kill["bounded_stop"] is True
            and agent_kill["session_lost"] is True
            and agent_kill["old_action_completed"] is False,
            "canonical Agent-kill watchdog path did not stop safely",
        )

        direct_sink_blocked = _direct_sink_is_blocked(observation_transport)
        proof_bundle = _build_gazebo_proof_bundle(
            source=source,
            output=output,
            canonical_receipt=canonical_receipt,
            launch_result=launch_result,
            rosbridge_loss=rosbridge_loss,
            agent_kill=agent_kill,
            direct_sink_blocked=direct_sink_blocked,
        )
        proof_path = output / "gazebo-proof-bundle.json"
        _write_json(proof_path, proof_bundle.to_dict())
        partial["proof_bundle"] = {
            "path": str(proof_path),
            "bundle_hash": proof_bundle.bundle_hash,
            "levels": {proof.module: proof.level.value for proof in proof_bundle.proofs},
        }

        partial["finished_at_ns"] = time.time_ns()
        partial["acceptance"] = {
            "canonical_task_verified": True,
            "diff_drive": True,
            "odometry": True,
            "laser": True,
            "agent_kill_bounded_stop": True,
            "observation_loss_not_task_verified": True,
            "worker_crash_bounded_stop": bool(launch_result["worker_crash"]["passed"]),
            "safe_cancel": bool(launch_result["cancel_and_recover"]["passed"]),
            "restart_old_action_not_replayed": True,
            "launch_testing_real_processes": True,
            "runtime_e3_or_higher": _proof_at_least(
                proof_bundle, "runtime", ModuleEvidenceLevel.DECISION_IMPACT
            ),
            "rosclawd_e5": _proof_at_least(
                proof_bundle, "rosclawd", ModuleEvidenceLevel.REPLAY_VERIFIED
            ),
            "simulation_only": True,
        }
        partial["passed"] = all(partial["acceptance"].values())
        _require(partial["passed"] is True, "Gazebo GuardedBase acceptance matrix failed")
        report_path = output / "gazebo-guarded-base-report.json"
        _write_json(report_path, partial)
        report_hash = _hash_file(report_path)
        manifest = _artifact_manifest(output)
        _write_json(output / "hashes.json", manifest)
        return GazeboGuardedBaseResult(
            output_dir=output,
            report_path=report_path,
            report_hash=report_hash,
            canonical_task_verified=True,
            launch_testing_passed=True,
            agent_kill_bounded_stop=True,
            rosbridge_loss_fail_closed=True,
            recovery_no_replay=True,
        )
    except Exception as exc:
        partial["passed"] = False
        partial["error"] = f"{type(exc).__name__}: {exc}"
        partial["finished_at_ns"] = time.time_ns()
        _write_json(output / "gazebo-guarded-base-failed.json", partial)
        raise
    finally:
        if daemon is not None:
            daemon.stop()
        for transport in transports:
            transport.close()
        if started:
            logs = _docker(["logs", container], check=False)
            (output / "managed-stack.log").write_text(
                logs.stdout + logs.stderr,
                encoding="utf-8",
            )
            _docker(["stop", "--time", "5", container], check=False)
            _docker(["rm", container], check=False)
            # Process logs remain live until the container is fully stopped;
            # seal the manifest only after that mutation boundary.
            if (output / "gazebo-guarded-base-report.json").is_file():
                _write_json(output / "hashes.json", _artifact_manifest(output))


def _run_launch_testing(*, image: str, source: Path, output: Path) -> dict[str, Any]:
    command = [
        "run",
        "--rm",
        "--network",
        "host",
        "--ipc",
        "host",
        "-e",
        "ROS_DOMAIN_ID=187",
        "-e",
        "ROS_LOCALHOST_ONLY=1",
        "-e",
        "ROSCLAW_REPO_ROOT=/workspace",
        "-e",
        "ROSCLAW_GAZEBO_EVIDENCE_DIR=/evidence",
        "-e",
        "LIBGL_ALWAYS_SOFTWARE=1",
        "-v",
        f"{source}:/workspace:ro",
        "-v",
        f"{output}:/evidence:rw",
        image,
        "bash",
        "-lc",
        (
            "source /opt/ros/humble/setup.bash && "
            "launch_test --junit-xml=/evidence/launch-testing-junit.xml "
            "/workspace/tests/simforge/launch/test_gazebo_guarded_base_launch.py"
        ),
    ]
    result = _docker(command, timeout=90.0, check=False)
    (output / "launch-testing-console.log").write_text(
        result.stdout + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"launch_testing failed with exit code {result.returncode}")
    path = output / "launch-testing-result.json"
    if not path.is_file():
        raise RuntimeError("launch_testing produced no result artifact")
    value = json.loads(path.read_text(encoding="utf-8"))
    required = ("agent_kill", "cancel_and_recover", "worker_crash", "odom_loss", "laser")
    if not all(value.get(name, {}).get("passed") is True for name in required):
        raise RuntimeError("launch_testing process-fault acceptance did not pass")
    value["artifact_hash"] = _hash_file(path)
    return value


def _start_stack(
    *,
    image: str,
    source: Path,
    output: Path,
    container: str,
    host_port: int,
) -> None:
    result = _docker(
        [
            "run",
            "-d",
            "--name",
            container,
            "--ipc",
            "host",
            "-p",
            f"127.0.0.1:{host_port}:9090",
            "-e",
            "ROS_DOMAIN_ID=188",
            "-e",
            "ROS_LOCALHOST_ONLY=1",
            "-e",
            "ROSCLAW_REPO_ROOT=/workspace",
            "-e",
            "ROSCLAW_GAZEBO_EVIDENCE_DIR=/evidence",
            "-e",
            "ROSCLAW_DEADMAN_TIMEOUT_SEC=0.60",
            "-e",
            "LIBGL_ALWAYS_SOFTWARE=1",
            "-v",
            f"{source}:/workspace:ro",
            "-v",
            f"{output}:/evidence:rw",
            image,
            "bash",
            "-lc",
            (
                "source /opt/ros/humble/setup.bash && "
                "exec bash /workspace/scripts/simforge/"
                "start_gazebo_guarded_base_stack.sh"
            ),
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to start Gazebo stack: {result.stderr.strip()}")


def _run_rosbridge_loss(
    *,
    mcp: RuntimeClient,
    container: str,
    endpoint_url: str,
    output: Path,
) -> dict[str, Any]:
    action_id = "gazebo-rosbridge-loss-v2"
    before_events = _read_jsonl(output / "stack-deadman-events.jsonl")
    result_box: dict[str, Any] = {}
    error_box: list[BaseException] = []

    def request() -> None:
        try:
            result_box.update(
                asyncio.run(
                    mcp.request_action(
                        capability_id="mobile_base.guarded_move",
                        arguments={
                            "linear_x_mps": 0.2,
                            "angular_z_radps": 0.0,
                            "duration_sec": 2.0,
                        },
                        execution_mode="SHADOW",
                        body_snapshot_hash=_BODY_HASH,
                        body_id="sim_gazebo_guarded_base",
                        action_id=action_id,
                        required_evidence="TASK_VERIFIED",
                        wait_timeout_sec=7.0,
                    )
                )
            )
        except BaseException as exc:  # noqa: BLE001
            error_box.append(exc)

    thread = threading.Thread(target=request, name="rosbridge-loss-action")
    thread.start()
    time.sleep(0.45)
    killed_at_ns = time.time_ns()
    rosbridge_pid = _container_pid(output, "rosbridge")
    killed = _docker(["exec", container, "kill", "-KILL", str(rosbridge_pid)], check=False)
    if killed.returncode != 0:
        raise RuntimeError("could not inject rosbridge process loss")
    thread.join(timeout=7.0)
    if thread.is_alive():
        raise RuntimeError("rosbridge-loss action did not terminate")
    if error_box:
        raise RuntimeError(f"rosbridge-loss request failed: {error_box[0]}")
    receipt = _receipt(result_box)
    events = _wait_for_deadman_event(
        output / "stack-deadman-events.jsonl",
        after_count=len(before_events),
        timeout_sec=2.0,
    )
    stop_events = [
        event
        for event in events
        if event.get("event") == "deadman_stop" and event.get("reason") == "fresh_command_timeout"
    ]
    if not stop_events:
        raise RuntimeError("ROS deadman emitted no stop after rosbridge loss")
    bounded_stop_sec = (int(stop_events[0]["wall_time_ns"]) - killed_at_ns) / 1e9
    _restart_rosbridge(container)
    _wait_for_topics(endpoint_url, timeout_sec=15.0)
    evidence = str(receipt.get("evidence_level"))
    return {
        "action_id": action_id,
        "fault_process": "rosbridge",
        "fault_pid": rosbridge_pid,
        "actual_sigkill": True,
        "bounded_stop_sec": bounded_stop_sec,
        "deadman_stop_observed": 0 <= bounded_stop_sec <= 1.25,
        "maximum_evidence": evidence,
        "task_verified": evidence == EvidenceLevel.TASK_VERIFIED.value,
        "receipt": receipt,
    }


def _run_agent_kill(
    *,
    daemon_socket: Path,
    client: DaemonClient,
    observation_transport: RosbridgeTransport,
) -> dict[str, Any]:
    action_id = "gazebo-agent-kill-v4"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "rosclaw.simforge.gazebo_guarded_base",
            "_lease-client",
            "--socket",
            str(daemon_socket),
            "--action-id",
            action_id,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                status = client.get_action_status(action_id)
            except Exception:  # noqa: BLE001
                time.sleep(0.05)
                continue
            if status.get("state") == "RUNNING":
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("leased Agent action never reached RUNNING")
        moving = _wait_for_speed(
            observation_transport,
            lambda value: value >= 0.10,
            timeout_sec=2.0,
        )
        killed_at = time.monotonic()
        process.kill()
        process.wait(timeout=2.0)
        stopped = _wait_for_speed(
            observation_transport,
            lambda value: abs(value) <= 0.02,
            timeout_sec=2.0,
        )
        stopped_at = time.monotonic()
        terminal = client.wait_for_action(action_id, timeout_sec=6.0)
        receipt = _receipt(terminal)
        errors = receipt.get("errors", [])
        error_codes = {item.get("code") for item in errors if isinstance(item, dict)}
        safety_stop = receipt.get("safety_stop", {})
        return {
            "action_id": action_id,
            "agent_pid": process.pid,
            "actual_sigkill": process.returncode == -signal.SIGKILL,
            "moving_speed_mps": moving[1],
            "stopped_speed_mps": stopped[1],
            "bounded_stop_sec": stopped_at - killed_at,
            "bounded_stop": stopped_at - killed_at <= 1.25,
            "session_lost": "AGENT_SESSION_LOST" in error_codes,
            "old_action_completed": receipt.get("final_state") == "COMPLETED",
            "safety_stop": safety_stop,
            "receipt": receipt,
        }
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2.0)


def _lease_client(socket_path: Path, action_id: str) -> int:
    client = DaemonClient(socket_path=socket_path, timeout_sec=5.0)
    session_id = f"session-{action_id}"
    client.create_session(
        session_id=session_id,
        actor_id="phase3-agent",
        agent_framework="mcp",
        body_scope=["sim_gazebo_guarded_base"],
        capability_scope=["mobile_base.guarded_move"],
        ttl_ms=500,
    )
    action = ActionEnvelope(
        action_id=action_id,
        actor_id="phase3-agent",
        agent_framework="mcp",
        session_id=session_id,
        body_id="sim_gazebo_guarded_base",
        body_snapshot_hash=_BODY_HASH,
        capability_id="mobile_base.guarded_move",
        arguments={
            "linear_x_mps": 0.2,
            "angular_z_radps": 0.0,
            "duration_sec": 4.0,
        },
        execution_mode=ExecutionMode.SHADOW,
        lease_ttl_ms=500,
        renew_interval_ms=150,
        verification_policy=VerificationPolicy(
            required_evidence=EvidenceLevel.TASK_VERIFIED,
            timeout_sec=6.0,
            fail_closed=True,
        ),
    )
    client.request_action(action)
    print(f"ROSCLAW_AGENT_ACTION_RUNNING action_id={action_id}", flush=True)
    while True:
        status = client.get_action_status(action_id)
        if status.get("state") in {"FINISHED", "CANCELLED"}:
            return 0
        client.heartbeat_session(session_id)
        client.renew_action_lease(action_id, session_id)
        time.sleep(0.1)


def _direct_sink_is_blocked(transport: RosbridgeTransport) -> bool:
    try:
        RosbridgeMobileBaseSink(
            transport,
            daemon_owner_id="provider_worker_direct",
            command_topic=_COMMAND_TOPIC,
            pose_topic=_ODOM_TOPIC,
            pose_message_type=_ODOM_TYPE,
        )
    except ValueError as exc:
        return "daemon instance id" in str(exc)
    return False


def _build_gazebo_proof_bundle(
    *,
    source: Path,
    output: Path,
    canonical_receipt: dict[str, Any],
    launch_result: dict[str, Any],
    rosbridge_loss: dict[str, Any],
    agent_kill: dict[str, Any],
    direct_sink_blocked: bool,
) -> ProofBundle:
    body_hash = _hash_file(
        source / "benchmarks/simforge/suites/core_v1/guarded_base/gazebo_guarded_base.sdf"
    )
    control_runtime = _runtime()
    control_action = ActionEnvelope(
        action_id="gazebo-runtime-no-executor-control",
        actor_id="phase3-proof",
        agent_framework="proof",
        session_id="phase3-proof-session",
        body_id="sim_gazebo_guarded_base",
        body_snapshot_hash=body_hash,
        capability_id="mobile_base.guarded_move",
        arguments={
            "linear_x_mps": 0.2,
            "angular_z_radps": 0.0,
            "duration_sec": 0.5,
        },
        execution_mode=ExecutionMode.SHADOW,
    )
    control_receipt = control_runtime.action_gateway.submit(control_action).to_dict()
    control_path = output / "runtime-no-executor-control.json"
    _write_json(control_path, control_receipt)
    direct_control = {
        "schema_version": "rosclaw.ros_direct_route_control.v1",
        "attempted_owner": "provider_worker_direct",
        "sink_constructed": not direct_sink_blocked,
        "command_dispatched": False,
        "blocked": direct_sink_blocked,
        "required_route": "MCP->rosclawd->ActionGateway->daemon_owned_sink",
    }
    direct_path = output / "direct-route-control.json"
    _write_json(direct_path, direct_control)

    canonical_hash = _hash_json(canonical_receipt)
    launch_hash = str(launch_result["artifact_hash"])
    runtime_control_hash = _hash_file(control_path)
    direct_control_hash = _hash_file(direct_path)
    canonical_passed = bool(
        canonical_receipt.get("final_state") == "COMPLETED"
        and canonical_receipt.get("evidence_level") == EvidenceLevel.TASK_VERIFIED.value
    )
    runtime = ModuleProof(
        module="runtime",
        invoked=True,
        input_refs=("body://" + body_hash.removeprefix("sha256:"),),
        output_refs=("receipt://" + canonical_hash.removeprefix("sha256:"),),
        output_valid=canonical_passed,
        decision_impacts=("registered_executor_changed_block_to_execution",),
        counterfactual=CounterfactualRun(
            control_run_id="runtime_without_executor",
            treatment_run_id="runtime_with_daemon_executor",
            same_seed=True,
            same_scenario=True,
            same_body_hash=True,
            decision_changed=(
                control_receipt.get("final_state") != "COMPLETED" and canonical_passed
            ),
            outcome_changed=True,
            metrics=(
                CounterfactualMetric(
                    name="task_verified",
                    control=0.0,
                    treatment=1.0,
                    lower_is_better=False,
                ),
            ),
            control_ref="receipt://" + runtime_control_hash.removeprefix("sha256:"),
            treatment_ref="receipt://" + canonical_hash.removeprefix("sha256:"),
        ),
        fault_injections=(
            FaultInjectionResult(
                name="missing_executor_blocked",
                passed=(
                    control_receipt.get("final_state") == "FAILED"
                    and any(
                        isinstance(error, dict) and error.get("code") == "EXECUTOR_UNAVAILABLE"
                        for error in control_receipt.get("errors", [])
                    )
                ),
                evidence_ref="receipt://" + runtime_control_hash.removeprefix("sha256:"),
            ),
        ),
    )
    launch_signals = set(launch_result.get("launch_testing", {}).get("actual_signals_injected", []))
    rosclawd = ModuleProof(
        module="rosclawd",
        invoked=True,
        input_refs=("receipt://" + direct_control_hash.removeprefix("sha256:"),),
        output_refs=("receipt://" + canonical_hash.removeprefix("sha256:"),),
        output_valid=canonical_passed,
        decision_impacts=("direct_route_denied_canonical_route_executed",),
        counterfactual=CounterfactualRun(
            control_run_id="direct_provider_route",
            treatment_run_id="canonical_rosclawd_route",
            same_seed=True,
            same_scenario=True,
            same_body_hash=True,
            decision_changed=direct_sink_blocked and canonical_passed,
            outcome_changed=True,
            metrics=(
                CounterfactualMetric(
                    name="authorized_task_verified",
                    control=0.0,
                    treatment=1.0,
                    lower_is_better=False,
                ),
            ),
            control_ref="receipt://" + direct_control_hash.removeprefix("sha256:"),
            treatment_ref="receipt://" + canonical_hash.removeprefix("sha256:"),
        ),
        fault_injections=(
            FaultInjectionResult(
                name="agent_kill_bounded_stop",
                passed=(
                    agent_kill.get("actual_sigkill") is True
                    and agent_kill.get("bounded_stop") is True
                    and agent_kill.get("session_lost") is True
                ),
                evidence_ref="fault://agent_kill",
            ),
            FaultInjectionResult(
                name="rosbridge_loss_fail_closed",
                passed=(
                    rosbridge_loss.get("actual_sigkill") is True
                    and rosbridge_loss.get("maximum_evidence")
                    == EvidenceLevel.DISPATCH_CONFIRMED.value
                    and rosbridge_loss.get("task_verified") is False
                ),
                evidence_ref="fault://rosbridge_loss",
            ),
            FaultInjectionResult(
                name="odometry_loss_fail_closed",
                passed=(
                    launch_result.get("odom_loss", {}).get("task_verified") is False
                    and launch_result.get("odom_loss", {}).get("passed") is True
                ),
                evidence_ref="fault://odometry_loss",
            ),
            FaultInjectionResult(
                name="worker_crash_bounded_stop",
                passed=launch_result.get("worker_crash", {}).get("passed") is True,
                evidence_ref="fault://worker_crash",
            ),
        ),
        replay_verified=bool(
            launch_result.get("launch_testing", {}).get("passed") is True
            and {"SIGKILL", "SIGTERM"}.issubset(launch_signals)
        ),
        replay_ref="artifact://" + launch_hash.removeprefix("sha256:"),
    )
    return ProofBundle(
        run_id="proof_gazebo_guarded_base_v1",
        task_id="gazebo_guarded_base_v1",
        body_snapshot_hash=body_hash,
        proofs=(runtime, rosclawd),
        evidence_root_ref="arena://" + _hash_json({"output": str(output)}).removeprefix("sha256:"),
    )


def _proof_at_least(
    bundle: ProofBundle,
    module: str,
    minimum: ModuleEvidenceLevel,
) -> bool:
    return any(
        proof.module == module and proof.level.rank >= minimum.rank for proof in bundle.proofs
    )


def _runtime() -> Runtime:
    return Runtime(
        RuntimeConfig(
            robot_id="sim_gazebo_guarded_base",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_skill_manager=False,
            enable_knowledge=False,
            enable_how=False,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
            enable_event_persistence=False,
            enable_tracing=False,
        )
    )


def _transport(endpoint_url: str) -> RosbridgeTransport:
    endpoint = RosbridgeEndpoint.from_url(endpoint_url)
    endpoint.timeout_sec = 2.0
    return RosbridgeTransport(endpoint=endpoint, max_retries=1)


def _wait_for_topics(endpoint_url: str, *, timeout_sec: float) -> set[str]:
    deadline = time.monotonic() + timeout_sec
    required = {_COMMAND_TOPIC, _ODOM_TOPIC, _SCAN_TOPIC}
    last_error = ""
    while time.monotonic() < deadline:
        transport = _transport(endpoint_url)
        try:
            result = transport.call_service("/rosapi/topics", {}, timeout_sec=2.0)
            if result.ok and isinstance(result.data, dict):
                values = result.data.get("values", {})
                topics = set(values.get("topics", [])) if isinstance(values, dict) else set()
                if required.issubset(topics):
                    return required
            last_error = result.error or "required topics absent"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        finally:
            transport.close()
        time.sleep(0.2)
    raise RuntimeError(f"Gazebo rosbridge topics not ready: {last_error}")


def _read_laser(transport: RosbridgeTransport) -> dict[str, Any]:
    result = transport.subscribe_once(_SCAN_TOPIC, msg_type=_SCAN_TYPE, timeout_sec=3.0)
    if not result.ok or not isinstance(result.data, dict):
        raise RuntimeError("no Gazebo laser observation")
    message = result.data.get("msg", result.data)
    ranges = message.get("ranges", []) if isinstance(message, dict) else []
    finite = [
        float(value)
        for value in ranges
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float("-inf") < float(value) < float("inf")
    ]
    if not finite:
        raise RuntimeError("Gazebo laser observation contained no finite ranges")
    return {
        "topic": _SCAN_TOPIC,
        "samples": len(ranges),
        "finite_samples": len(finite),
        "nearest_obstacle_m": min(finite),
        "physically_observed": True,
    }


def _wait_for_speed(
    transport: RosbridgeTransport,
    predicate: Any,
    *,
    timeout_sec: float,
) -> tuple[float, float]:
    deadline = time.monotonic() + timeout_sec
    last: tuple[float, float] | None = None
    while time.monotonic() < deadline:
        last = read_mobile_base_pose(
            transport,
            topic=_ODOM_TOPIC,
            message_type=_ODOM_TYPE,
            timeout_sec=min(0.4, max(0.05, deadline - time.monotonic())),
        )
        if last is not None and predicate(last[1]):
            return last
    raise RuntimeError(f"odometry speed predicate failed; last={last!r}")


def _restart_rosbridge(container: str) -> None:
    result = _docker(
        [
            "exec",
            "-d",
            container,
            "bash",
            "-lc",
            (
                "source /opt/ros/humble/setup.bash && "
                "exec /opt/ros/humble/lib/rosbridge_server/rosbridge_websocket "
                "--ros-args -p port:=9090 "
                ">>/evidence/rosbridge-restart.log 2>&1"
            ),
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("failed to restart rosbridge")


def _container_pid(output: Path, name: str) -> int:
    path = output / "processes.txt"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[0] == name and parts[1].isdigit():
                    pid = int(parts[1])
                    if pid > 1:
                        return pid
        time.sleep(0.05)
    raise RuntimeError(f"managed stack PID not found: {name}")


def _wait_for_deadman_event(
    path: Path,
    *,
    after_count: int,
    timeout_sec: float,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        events = _read_jsonl(path)
        new = events[after_count:]
        if any(item.get("event") == "deadman_stop" for item in new):
            return new
        time.sleep(0.05)
    return _read_jsonl(path)[after_count:]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def _receipt(result: dict[str, Any]) -> dict[str, Any]:
    receipt = result.get("receipt")
    if not isinstance(receipt, dict):
        raise RuntimeError("action produced no ExecutionReceipt")
    return receipt


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _available_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _docker(
    arguments: list[str],
    *,
    timeout: float = 30.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["docker", *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"docker {' '.join(arguments[:3])} failed: {result.stderr.strip()}")
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _hash_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_json(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _artifact_manifest(output: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "hashes.json":
            continue
        entries.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": _hash_file(path),
            }
        )
    return {
        "schema_version": "rosclaw.raw_evidence_manifest.v1",
        "generated_at_ns": time.time_ns(),
        "artifacts": entries,
    }


def _main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    client = subparsers.add_parser("_lease-client")
    client.add_argument("--socket", type=Path, required=True)
    client.add_argument("--action-id", required=True)
    args = parser.parse_args()
    if args.command == "_lease-client":
        return _lease_client(args.socket, args.action_id)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["GazeboGuardedBaseResult", "run_gazebo_guarded_base"]
