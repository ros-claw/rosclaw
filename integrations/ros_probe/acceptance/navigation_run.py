"""Known-model navigation-only SIM journey through MCP and a separate daemon.

The operator-owned stack prepares the compiled Body before starting physics.
This Agent-side process neither opens ROS nor constructs an execution Runtime.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from rosclaw.daemon.client import DaemonClient
from rosclaw.mcp import tools
from rosclaw.mcp.adapters.runtime_client import RuntimeClient


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    root = args.directory.resolve()
    if len(os.fsencode(root / "run/rosclawd.sock")) >= 104:
        raise ValueError("use a shorter isolated directory for the rosclawd Unix socket")
    config = json.loads((root / "execution_config.json").read_text())
    if config.get("executor_kind") != "navigation_only":
        raise ValueError("operator-prepared navigation-only Body required")
    os.environ["ROSCLAW_HOME"] = str(root / "home")
    os.environ["ROSCLAW_ROS_EXPERT"] = "1"
    ready = root / "daemon_ready.json"
    if ready.exists():
        raise ValueError("fresh isolated journey directory required")
    with (root / "daemon.log").open("x") as log:
        daemon = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("daemon.py")),
                "--directory",
                str(root),
                "--endpoint",
                args.endpoint,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 45
            while not ready.exists():
                if daemon.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("navigation daemon failed to start; inspect daemon.log")
                await asyncio.sleep(0.2)
            client = DaemonClient(socket_path=root / "run/rosclawd.sock")
            client.arm_runtime("isolated known-model navigation-only SIM acceptance")
            runtime_client = RuntimeClient(
                project_root=Path.cwd(),
                robot_id=config["body_id"],
                runtime_profile={},
                daemon_client=client,
            )
            tools._client = lambda: runtime_client
            ticket = await tools._request_action(
                capability_id="navigation.navigate_to_pose",
                arguments={
                    "pose": {
                        "header": {"frame_id": "map"},
                        "pose": {
                            "position": {"x": 0.35, "y": 0.0, "z": 0.0},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        },
                    }
                },
                execution_mode="SIMULATION",
                body_id=config["body_id"],
                body_snapshot_hash=config["body_snapshot_hash"],
                action_id="golden-navigation",
                timeout_sec=60,
                wait_timeout_sec=0,
            )
            (root / "golden-navigation.ticket.json").write_text(json.dumps(ticket, indent=2) + "\n")
            deadline = time.monotonic() + 80
            while time.monotonic() < deadline:
                status = client.get_action_status("golden-navigation")
                if status.get("state") in {"FINISHED", "CANCELLED"}:
                    receipt = client.get_execution_receipt("golden-navigation")
                    (root / "golden-navigation.receipt.json").write_text(
                        json.dumps(receipt, indent=2) + "\n"
                    )
                    if receipt.get("receipt", {}).get("final_state") != "COMPLETED":
                        raise RuntimeError("navigation was not independently verified COMPLETED")
                    print(
                        json.dumps({"status": "COMPLETED", "body_id": config["body_id"]}),
                        flush=True,
                    )
                    return
                await asyncio.sleep(0.2)
            raise TimeoutError("navigation action deadline")
        finally:
            daemon.terminate()
            try:
                daemon.wait(timeout=15)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait(timeout=5)


if __name__ == "__main__":
    asyncio.run(main())
