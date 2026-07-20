#!/usr/bin/env python3
"""ROSClaw component integration acceptance.

This script exercises MuJoCo validation, a simulation-only guarded callback,
the independent rosclawd process boundary, and the canonical Agent MCP tool
catalog. It does not connect to ROS 2 or hardware and must not be cited as a
real-robot acceptance result.
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Allow this script to run directly from a source checkout.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rosclaw.agent.tool_catalog import P0_AGENT_MCP_TOOLS
from rosclaw.daemon.client import DaemonClient, DaemonUnavailableError
from rosclaw.firewall import DigitalTwinFirewall, SafetyViolationError
from rosclaw.kernel import ActionEnvelope, AuthorizationContext, ExecutionMode


def _heading(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_digital_twin() -> bool:
    """Validate both accepted and rejected UR5e trajectories in MuJoCo."""
    _heading("TEST 1: Digital Twin Firewall")
    model_path = ROOT / "src" / "rosclaw" / "specs" / "ur5e.xml"
    if not model_path.exists():
        print(f"FAIL: model not found at {model_path}")
        return False

    joint_limits = {
        "shoulder_pan_joint": (-6.2831853, 6.2831853),
        "shoulder_lift_joint": (-6.2831853, 6.2831853),
        "elbow_joint": (-3.1415926, 3.1415926),
        "wrist_1_joint": (-6.2831853, 6.2831853),
        "wrist_2_joint": (-6.2831853, 6.2831853),
        "wrist_3_joint": (-6.2831853, 6.2831853),
    }
    firewall = DigitalTwinFirewall(
        model_path=str(model_path),
        joint_limits=joint_limits,
        sim_steps_per_check=10,
    )

    valid = firewall.validate_trajectory(
        trajectory=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        time_step=0.001,
    )
    invalid = firewall.validate_trajectory(
        trajectory=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        time_step=0.001,
    )
    if not valid.is_safe or invalid.is_safe:
        print("FAIL: MuJoCo firewall did not preserve accept/reject semantics")
        return False

    print("PASS: safe trajectory accepted in simulation")
    print(f"PASS: unsafe trajectory rejected: {invalid.violation_details}")
    return True


def test_simulation_guard() -> bool:
    """Verify a validated callback without claiming physical execution."""
    _heading("TEST 2: Simulation-Only Firewall Decorator")
    from rosclaw.firewall import mujoco_firewall

    model_path = ROOT / "src" / "rosclaw" / "specs" / "ur5e.xml"
    joint_limits = {
        "shoulder_pan_joint": (-6.2831853, 6.2831853),
        "shoulder_lift_joint": (-6.2831853, 6.2831853),
        "elbow_joint": (-3.1415926, 3.1415926),
        "wrist_1_joint": (-6.2831853, 6.2831853),
        "wrist_2_joint": (-6.2831853, 6.2831853),
        "wrist_3_joint": (-6.2831853, 6.2831853),
    }

    @mujoco_firewall(model_path=str(model_path), joint_limits=joint_limits)
    def accept_simulated_plan(trajectory_points: list[list[float]]) -> dict[str, object]:
        return {
            "status": "validated_callback_completed",
            "execution_mode": "SIMULATION",
            "hardware_command_dispatched": False,
            "points": len(trajectory_points),
        }

    try:
        result = accept_simulated_plan(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    except SafetyViolationError as exc:
        print(f"FAIL: safe simulated plan was rejected: {exc}")
        return False

    if (
        result.get("execution_mode") != "SIMULATION"
        or result.get("hardware_command_dispatched") is not False
    ):
        print(f"FAIL: callback crossed its simulation boundary: {result}")
        return False
    print(f"PASS: validated simulation callback returned truthful evidence: {result}")
    return True


def test_rosclawd_boundary() -> bool:
    """Start rosclawd separately and prove a forged REAL permit is blocked."""
    _heading("TEST 3: rosclawd Authorization Boundary")
    with tempfile.TemporaryDirectory(prefix="rosclaw-integration-") as tmp:
        workspace = Path(tmp)
        home = workspace / "home"
        socket_path = home / "run" / "rosclawd.sock"
        env = os.environ.copy()
        env["ROSCLAW_HOME"] = str(home)
        env["ROSCLAW_DAEMON_SOCKET"] = str(socket_path)
        env["PYTHONPATH"] = str(ROOT / "src")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "rosclaw.daemon.cli",
                "--socket",
                str(socket_path),
                "--robot-id",
                "sim_ur5e",
                "--log-level",
                "ERROR",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        client = DaemonClient(socket_path=socket_path, timeout_sec=2.0)
        try:
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    print(f"FAIL: rosclawd exited early ({process.returncode}): {stdout}\n{stderr}")
                    return False
                try:
                    status = client.get_runtime_status()
                    break
                except DaemonUnavailableError:
                    time.sleep(0.05)
            else:
                print("FAIL: rosclawd did not become ready")
                return False

            action = ActionEnvelope(
                action_id="action-integration-forged",
                actor_id="integration-agent",
                agent_framework="integration-test",
                session_id="integration-session",
                body_id="sim_ur5e",
                body_snapshot_hash="sha256:integration-body",
                capability_id="robot.move_joints",
                arguments={"joint_positions": [0.0] * 6},
                execution_mode=ExecutionMode.REAL,
                authorization=AuthorizationContext(
                    principal_id="forged-operator",
                    approved=True,
                    approval_id="forged-permit",
                    scopes=["*"],
                ),
            )
            ticket = client.request_action(action)
            receipt = client.wait_for_action(
                str(ticket["action_id"]),
                timeout_sec=5.0,
            )["receipt"]
            final_status = client.get_runtime_status()
            checks = {
                "separate_process": status["daemon_pid"] == process.pid != os.getpid(),
                "southbound_owner": status["southbound_owner"] == "rosclawd",
                "blocked": receipt["final_state"] == "BLOCKED",
                "authorization_error": (receipt["errors"][0]["code"] == "AUTHORIZATION_REQUIRED"),
                "not_real_evidence": receipt["usable_for_real_execution"] is False,
                "no_hardware_action": final_status["hardware_actions_executed"] == 0,
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                print(f"FAIL: rosclawd boundary checks failed: {failed}")
                return False
            client.shutdown()
            process.wait(timeout=10.0)
            if socket_path.exists():
                print("FAIL: rosclawd left its socket behind")
                return False
            print(
                "PASS: separate rosclawd blocked forged REAL authorization; "
                "hardware_actions_executed=0"
            )
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: rosclawd boundary test raised: {exc}")
            return False
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)


def test_agent_tool_catalog() -> bool:
    """Verify the exact canonical Agent-facing tool catalog."""
    _heading("TEST 4: Agent MCP Tool Catalog")
    expected_control_tools = {
        "get_runtime_status",
        "request_action",
        "get_action_status",
        "cancel_action",
    }
    tools = tuple(P0_AGENT_MCP_TOOLS)
    if len(tools) != 22 or len(set(tools)) != 22:
        print(f"FAIL: expected 22 unique tools, found {len(tools)}")
        return False
    missing = sorted(expected_control_tools.difference(tools))
    if missing:
        print(f"FAIL: missing daemon-backed tools: {missing}")
        return False
    print("PASS: exact 22-tool catalog includes all daemon-backed controls")
    return True


def main() -> int:
    """Run component integration acceptance."""
    _heading("ROSCLAW COMPONENT INTEGRATION ACCEPTANCE")
    results = [
        ("Digital Twin", test_digital_twin()),
        ("Simulation Guard", test_simulation_guard()),
        ("rosclawd Boundary", test_rosclawd_boundary()),
        ("Agent MCP Catalog", test_agent_tool_catalog()),
    ]

    _heading("TEST SUMMARY")
    for name, passed in results:
        print(f"{name:.<40} {'PASS' if passed else 'FAIL'}")

    if not all(passed for _name, passed in results):
        print("\nComponent integration acceptance failed.")
        return 1
    print(
        "\nComponent integration checks passed. Real ROS 2, cross-UID, "
        "device-specific E-Stop, and robot-hardware acceptance remain required."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
