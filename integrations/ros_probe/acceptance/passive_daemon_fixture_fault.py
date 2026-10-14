"""One SDK-owned passive daemon generation replacement; no motion authority."""

import argparse
import json
import os
import time
from pathlib import Path

from rosclaw.daemon.client import DaemonClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    out = Path("/evidence")
    if not Path("/.dockerenv").exists() or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise ValueError("isolated owned SDK fixture required")
    config = json.loads((out / "supervision/execution_config.json").read_text())
    fact = json.loads((out / "passive-daemon-process.json").read_text())
    pid = fact["pid"]
    current_stat = Path(f"/proc/{pid}/stat").read_text()
    if (
        config["run_id"] != args.run_id
        or config["executor_kind"] != "passive_observation"
        or current_stat.rsplit(")", 1)[1].split()[19] != fact["stat"].rsplit(")", 1)[1].split()[19]
        or "/acceptance/daemon.py" not in Path(f"/proc/{pid}/cmdline").read_bytes().decode()
    ):
        raise ValueError("original owned passive daemon identity required")
    ready = json.loads((out / "supervision" / f"daemon-ready-{pid}.json").read_text())
    request = {
        "run_id": args.run_id,
        "expected_pid": pid,
        "expected_generation": ready["daemon_instance_id"],
    }
    before = time.monotonic_ns()
    with (out / "passive-daemon-restart-request.json").open("x") as stream:
        stream.write(json.dumps(request) + "\n")
    deadline = time.monotonic() + 15
    client = DaemonClient(
        socket_path=out / "supervision/run/rosclawd.sock",
        expected_daemon_uid=ready["uid"],
        timeout_sec=0.15,
    )
    while True:
        if time.monotonic() > deadline:
            raise RuntimeError("owned daemon replacement did not complete")
        try:
            status = client.get_runtime_status()
            if status["daemon_instance_id"] != ready["daemon_instance_id"]:
                assert status["daemon_pid"] != pid
                break
        except RuntimeError:
            pass
        time.sleep(0.05)
    with (out / "passive-daemon-replacement-original.json").open("x") as stream:
        stream.write(
            json.dumps(
                {
                    "before_monotonic_ns": before,
                    "after_monotonic_ns": time.monotonic_ns(),
                    "run_id": args.run_id,
                    "old_generation": ready["daemon_instance_id"],
                    "old_pid": pid,
                    "actual_new_status": status,
                    "goals_sent": 0,
                    "velocity_publishers_created": 0,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
