"""Live SIM fault acceptance; all motion requests use the existing daemon MCP path.

Run against a fresh owned golden container for each case. Fixture perturbations
only pause the world or stop owned processes; no motor command is sent here.
Independent Gazebo poses, not an action acknowledgement, establish standstill.
"""

import argparse
import asyncio
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from rosclaw.daemon.client import DaemonClient
from rosclaw.mcp import tools
from rosclaw.mcp.adapters.runtime_client import RuntimeClient

ROOT = Path(__file__).resolve().parents[3]


def observation(fixture):
    with (fixture / "witness.jsonl").open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 32768))
        sample = json.loads(stream.read().splitlines()[-1])
    age = (datetime.now(UTC) - datetime.fromisoformat(sample["captured_at"])).total_seconds()
    if not 0 <= age < 1 or not sample["observation_complete"]:
        raise RuntimeError("fresh complete independent observation required")
    return sample


def fixture_command(container, command):
    return subprocess.run(
        ["docker", "exec", container, "bash", "-c", command],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout


def clock_paused(container, paused):
    response = fixture_command(
        container,
        "source /opt/ros/jazzy/setup.bash && gz service -s /world/ros_expert/control "
        "--reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean --timeout 3000 "
        f"--req 'pause: {'true' if paused else 'false'}'",
    )
    if "data: true" not in response:
        raise RuntimeError("world mutation was not acknowledged")
    return response.strip()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=["daemon_kill", "clock_pause", "bridge_kill"])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--container", default="ros-expert-golden")
    args = parser.parse_args()
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ("robot.urdf", "measured_map.json"):
        (root / name).write_bytes((args.fixture / name).read_bytes())
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "integrations/ros_probe/acceptance/run.py"),
            "--directory",
            str(root),
            "--prepare-only",
        ],
        check=True,
    )
    os.environ["ROSCLAW_HOME"] = str(root / "home")
    os.environ["ROSCLAW_ROS_EXPERT"] = "1"
    (root / "daemon_ready.json").unlink(missing_ok=True)
    log = (root / "daemon.log").open("w")
    daemon = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "integrations/ros_probe/acceptance/daemon.py"),
            "--directory",
            str(root),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    record = {"case": args.case, "evidence_domain": "SIMULATION", "status": "FAIL"}
    paused = False
    try:
        deadline = time.monotonic() + 30
        while not (root / "daemon_ready.json").exists():
            if daemon.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("daemon not ready")
            await asyncio.sleep(0.2)
        body = json.loads((root / "body.json").read_text())
        client = DaemonClient(socket_path=root / "run/rosclawd.sock")
        client.arm_runtime("isolated SIM live fault test")
        runtime_client = RuntimeClient(
            project_root=ROOT,
            robot_id=body["body_id"],
            runtime_profile={},
            daemon_client=client,
        )
        tools._client = lambda: runtime_client

        async def request(capability, arguments, action_id, timeout):
            return await tools._request_action(
                capability_id=capability,
                arguments=arguments,
                execution_mode="SIMULATION",
                body_snapshot_hash=body["effective_body_hash"],
                body_id=body["body_id"],
                action_id=action_id,
                timeout_sec=timeout,
                wait_timeout_sec=0,
            )

        await request("localization.set_initial_pose", {}, "fault-localize", 10)
        deadline = time.monotonic() + 15
        while client.get_action_status("fault-localize").get("state") != "FINISHED":
            if time.monotonic() > deadline:
                raise TimeoutError("localization")
            await asyncio.sleep(0.1)
        localize = client.get_execution_receipt("fault-localize")
        if localize.get("receipt", {}).get("final_state") != "COMPLETED":
            raise RuntimeError("localization failed")
        record["localization_receipt"] = localize
        record["ticket"] = await request(
            "coverage.execute",
            {
                "mission_id": "fault-test",
                "polygons": [
                    {"points": [{"x": x, "y": y, "z": 0.0} for x, y in body["coverage_polygon"]]}
                ],
            },
            "fault-coverage",
            900,
        )
        deadline = time.monotonic() + 45
        initial = observation(args.fixture)
        while True:
            current = observation(args.fixture)
            if (
                current["cleaning_enabled"]
                and math.hypot(current["x"] - initial["x"], current["y"] - initial["y"]) > 0.05
            ):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("actual motion was not observed before fault")
            await asyncio.sleep(0.1)
        record["before"] = current
        record["injected_at"] = datetime.now(UTC).isoformat()
        if args.case == "daemon_kill":
            daemon.send_signal(signal.SIGKILL)
            daemon.wait(timeout=5)
        elif args.case == "clock_pause":
            record["pause_ack"] = clock_paused(args.container, True)
            paused = True
            await asyncio.sleep(2)
            record["resume_ack"] = clock_paused(args.container, False)
            paused = False
        else:
            listing = fixture_command(args.container, "ps -eo pid,args")
            pids = [
                int(line.split()[0])
                for line in listing.splitlines()
                if "/rosbridge_server/rosbridge_websocket" in line
            ]
            if len(pids) != 1:
                raise RuntimeError("exact owned rosbridge process not found")
            fixture_command(args.container, f"kill -KILL {pids[0]}")
        await asyncio.sleep(5)
        samples = []
        until = time.monotonic() + 3
        while time.monotonic() < until:
            samples.append(observation(args.fixture))
            await asyncio.sleep(0.1)
        record["post_fault_observations"] = samples
        displacement = max(
            math.hypot(s["x"] - samples[0]["x"], s["y"] - samples[0]["y"]) for s in samples
        )
        yaw_change = max(
            abs(
                math.atan2(
                    math.sin(s["yaw"] - samples[0]["yaw"]), math.cos(s["yaw"] - samples[0]["yaw"])
                )
            )
            for s in samples
        )
        record["standstill_displacement_m"] = displacement
        record["standstill_yaw_change_rad"] = yaw_change
        if args.case != "daemon_kill":
            deadline = time.monotonic() + 15
            while client.get_action_status("fault-coverage").get("state") != "FINISHED":
                if time.monotonic() > deadline:
                    raise TimeoutError("failed canonical action did not settle")
                await asyncio.sleep(0.1)
            record["canonical_receipt"] = client.get_execution_receipt("fault-coverage")
            if record["canonical_receipt"].get("receipt", {}).get("final_state") != "FAILED":
                raise RuntimeError("fault was not reflected in canonical failure")
        else:
            record["canonical_receipt"] = None
            record["receipt_absence_reason"] = "daemon killed before terminal persistence"
        if (
            displacement > 0.01
            or yaw_change > 0.03
            or any(
                s["cleaning_enabled"] or s["lease_remaining_sec"] > 0 or s["collision_count"] > 0
                for s in samples
            )
        ):
            raise RuntimeError("independently measured stop/cleaning/collision gate failed")
        record["status"] = "PASS"
    except Exception as exc:
        record["error"] = str(exc)
        raise
    finally:
        if paused:
            clock_paused(args.container, False)
        if daemon.poll() is None:
            daemon.terminate()
            try:
                daemon.wait(timeout=10)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait()
        log.close()
        (root / "safety.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"case": args.case, "status": record["status"]}))


if __name__ == "__main__":
    asyncio.run(main())
