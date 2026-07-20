"""Subprocess black-box proof that agent clients do not own the Runtime."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rosclaw.daemon.client import DaemonClient, DaemonUnavailableError
from rosclaw.kernel import (
    ActionEnvelope,
    AuthorizationContext,
    ExecutionMode,
)


def test_rosclawd_subprocess_blocks_unauthorized_action_and_disappears_on_stop(
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "home" / "run" / "rosclawd.sock"
    env = os.environ.copy()
    env["ROSCLAW_HOME"] = str(tmp_path / "home")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "rosclaw.daemon.cli",
            "--socket",
            str(socket_path),
            "--robot-id",
            "universal_robots_ur5e",
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
                pytest.fail(f"rosclawd exited early ({process.returncode}): {stdout}\n{stderr}")
            if socket_path.exists():
                try:
                    status = client.get_runtime_status()
                    break
                except DaemonUnavailableError:
                    pass
            time.sleep(0.05)
        else:
            pytest.fail("rosclawd did not become ready")

        assert status["daemon_pid"] == process.pid
        assert status["daemon_pid"] != os.getpid()
        action = ActionEnvelope(
            action_id="action-subprocess-forged",
            actor_id="codex-agent",
            agent_framework="codex",
            session_id="session-blackbox",
            body_id="universal_robots_ur5e",
            body_snapshot_hash="sha256:body",
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
            ticket["action_id"],
            timeout_sec=3.0,
        )["receipt"]
        assert receipt["final_state"] == "BLOCKED"
        assert receipt["errors"][0]["code"] == "AUTHORIZATION_REQUIRED"
        assert client.get_runtime_status()["hardware_actions_executed"] == 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)

    with pytest.raises(DaemonUnavailableError):
        client.request_action(action)
