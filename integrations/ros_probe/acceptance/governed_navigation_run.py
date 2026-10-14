"""Canonical MCP Agent fixture and separate existing SDK governed rosclawd.

The Agent imports no ROS client, constructs no Runtime and registers no driver.
The isolated operator has already pinned original G1/L1, P03, Body and goals.
"""

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from rosclaw.daemon.client import DaemonClient
from rosclaw.mcp import tools
from rosclaw.mcp.adapters.runtime_client import RuntimeClient


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal-policy-sha256", required=True)
    args = parser.parse_args()
    evidence = Path("/evidence")
    raw = (evidence / "governed-goals.json").read_bytes()
    if not 0 < len(raw) <= 16_000 or hashlib.sha256(raw).hexdigest() != args.goal_policy_sha256:
        raise ValueError("independent bounded operator goal-policy SHA required")
    policy = json.loads(raw)
    if (
        set(policy) != {"schema_version", "run_id", "body_snapshot_hash", "goals"}
        or policy["schema_version"] != "rosclaw.governed_sim_navigation_goals.v1"
        or type(policy["goals"]) is not list
        or len(policy["goals"]) != 2
    ):
        raise ValueError("closed two-goal operator SIM policy required")
    root = evidence / "execution-control/daemon"
    root.mkdir(mode=0o700)
    (root / "home").mkdir()
    shutil.copytree(evidence / "generic-compiled/body", root / "home/body")
    base = json.loads((evidence / "generic-compiled/execution_config.json").read_text())
    facts_raw = (evidence / "execution-facts-enrollment.json").read_bytes()
    facts = json.loads(facts_raw)
    if (
        policy["body_snapshot_hash"] != base["body_snapshot_hash"]
        or policy["run_id"] != facts["request"]["sim_run_id"]
    ):
        raise ValueError("operator goals differ from actual compiled Body/run")
    p03_raw = (evidence / "P03-accepted-review.json").read_bytes()
    config = {
        "schema_version": "rosclaw.governed_navigation_fixture.v1",
        "executor_kind": "governed_navigation",
        "body_id": base["body_id"],
        "body_snapshot_hash": base["body_snapshot_hash"],
        "supervision": {
            "schema_version": "rosclaw.governed_navigation_supervision.v1",
            "facts_enrollment_sha256": hashlib.sha256(facts_raw).hexdigest(),
            "P03_review_sha256": hashlib.sha256(p03_raw).hexdigest(),
            "allowed_action_ids": [goal["action_id"] for goal in policy["goals"]],
        },
    }
    (root / "execution_config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in ("robot.urdf", "world.sdf", "nav2.yaml"):
        shutil.copy2(Path("/assets") / name, root / name)
    shutil.copy2(evidence / "run_id.txt", root / "run_id.txt")
    argv = [
        sys.executable,
        str(Path(__file__).with_name("daemon.py")),
        "--directory",
        str(root),
        "--endpoint",
        "ws://127.0.0.1:9090",
        "--persistent-ledger",
    ]
    (evidence / "governed-daemon-argv.json").write_text(json.dumps(argv) + "\n")
    os.environ["ROSCLAW_HOME"] = str(root / "home")
    os.environ["ROSCLAW_ROS_EXPERT"] = "1"
    result, error = [], None
    with (evidence / "governed-daemon-original.log").open("x") as log:
        daemon = subprocess.Popen(
            argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 30
            while not (root / "daemon_ready.json").exists():
                if daemon.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("existing governed SDK daemon did not become ready")
                await asyncio.sleep(0.05)
            ready = json.loads((root / "daemon_ready.json").read_text())
            (evidence / "governed-daemon-ready-original.json").write_text(
                json.dumps(ready, indent=2) + "\n"
            )
            client = DaemonClient(
                socket_path=root / "run/rosclawd.sock",
                expected_daemon_uid=os.getuid(),
                timeout_sec=0.5,
            )
            status = client.get_runtime_status()
            if (
                not status["running"]
                or status["daemon_peer"]["pid"] != daemon.pid
                or status["daemon_instance_id"] != ready["daemon_instance_id"]
                or status["supervision_state"] != "DISARMED"
            ):
                raise ValueError("actual independent SIM daemon socket/generation required")
            (evidence / "governed-daemon-enrolled-original-status.json").write_text(
                json.dumps(status, indent=2) + "\n"
            )
            runtime_client = RuntimeClient(
                project_root=Path("/repo"),
                robot_id=base["body_id"],
                runtime_profile={},
                daemon_client=client,
            )
            tools.set_client(runtime_client)
            for index, goal in enumerate(policy["goals"]):
                if set(goal) != {"action_id", "arguments", "timeout_sec"}:
                    raise ValueError("closed bounded SIM goal required")
                request = {
                    "capability_id": "navigation.navigate_to_pose",
                    "arguments": goal["arguments"],
                    "execution_mode": "SIMULATION",
                    "body_id": base["body_id"],
                    "body_snapshot_hash": base["body_snapshot_hash"],
                    "action_id": goal["action_id"],
                    "timeout_sec": goal["timeout_sec"],
                    "wait_timeout_sec": 0,
                }
                (evidence / f"governed-action-{index}-request-original.json").write_text(
                    json.dumps(request, indent=2) + "\n"
                )
                ticket = await tools._request_action(**request)
                (evidence / f"governed-action-{index}-ticket-original.json").write_text(
                    json.dumps(ticket, indent=2) + "\n"
                )
                deadline = time.monotonic() + goal["timeout_sec"] + 5
                while True:
                    action_status = await tools._get_action_status(goal["action_id"])
                    if action_status["state"] in {"FINISHED", "CANCELLED"}:
                        break
                    if daemon.poll() is not None or time.monotonic() >= deadline:
                        raise TimeoutError("canonical governed SIM action did not terminate")
                    await asyncio.sleep(0.1)
                receipt = client.get_execution_receipt(goal["action_id"])
                (evidence / f"governed-action-{index}-receipt-original.json").write_text(
                    json.dumps(receipt, indent=2) + "\n"
                )
                actual = receipt["receipt"]
                result.append(
                    {
                        "action_id": goal["action_id"],
                        "final_state": actual["final_state"],
                        "verification": actual["verification_result"],
                    }
                )
                if (
                    actual["final_state"] != "COMPLETED"
                    or actual["verification_result"].get("physical_stop_verified") is not True
                ):
                    raise RuntimeError(
                        "canonical governed action was not independently COMPLETED: "
                        + json.dumps(actual["errors"])
                    )
        except Exception as exc:
            error = str(exc)[:2500]
        finally:
            (evidence / "governed-owner-shutdown-request-original.json").write_text(
                json.dumps(
                    {
                        "monotonic_ns": time.monotonic_ns(),
                        "completed_goals": len(result),
                        "error": error,
                    }
                )
                + "\n"
            )
            if daemon.poll() is None:
                os.killpg(daemon.pid, signal.SIGTERM)
            try:
                code = daemon.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(daemon.pid, signal.SIGKILL)
                code = daemon.wait(timeout=5)
            (evidence / "governed-daemon-exit-original.json").write_text(
                json.dumps({"pid": daemon.pid, "returncode": code}) + "\n"
            )
    outcome = {
        "status": "FAILED_ORIGINAL_GOVERNED_SIM_JOURNEY"
        if error or code != 0
        else "PASS_TWO_CANONICAL_GOVERNED_SIM_GOALS_LOCAL_ONLY",
        "error": error,
        "daemon_returncode": code,
        "goals": result,
        "G3_done": False,
        "v1_done": False,
    }
    with (evidence / "governed-journey-original-outcome.json").open("x") as stream:
        stream.write(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome), flush=True)
    if error or code != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
