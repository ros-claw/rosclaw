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

from rosclaw.daemon.client import DaemonClient, DaemonRequestError
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
    parser.add_argument(
        "case",
        choices=["daemon_kill", "daemon_restart", "clock_pause", "bridge_kill", "observer_stop"],
    )
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
    daemon_argv = [
        sys.executable,
        str(ROOT / "integrations/ros_probe/acceptance/daemon.py"),
        "--directory",
        str(root),
    ]
    if args.case == "daemon_restart":
        daemon_argv.append("--persistent-ledger")
    daemon = subprocess.Popen(
        daemon_argv,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    record = {"case": args.case, "evidence_domain": "SIMULATION", "status": "FAIL"}
    paused = False
    stopped_witness = None
    pose_observer = None
    pose_log = None
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
        if args.case == "daemon_restart":
            record["runtime_before"] = client.get_runtime_status()
            record["session_before"] = client.get_session(record["ticket"]["session_id"])
        record["injected_at"] = datetime.now(UTC).isoformat()
        if args.case in {"daemon_kill", "daemon_restart"}:
            daemon.send_signal(signal.SIGKILL)
            daemon.wait(timeout=5)
        elif args.case == "clock_pause":
            record["pause_ack"] = clock_paused(args.container, True)
            paused = True
            await asyncio.sleep(2)
            record["resume_ack"] = clock_paused(args.container, False)
            paused = False
        elif args.case == "bridge_kill":
            listing = fixture_command(args.container, "ps -eo pid,args")
            pids = [
                int(line.split()[0])
                for line in listing.splitlines()
                if "/rosbridge_server/rosbridge_websocket" in line
            ]
            if len(pids) != 1:
                raise RuntimeError("exact owned rosbridge process not found")
            fixture_command(args.container, f"kill -KILL {pids[0]}")
        else:
            pose_log = (root / "pose-observer.log").open("w")
            pose_observer = subprocess.Popen(
                [
                    "docker",
                    "exec",
                    args.container,
                    "bash",
                    "-c",
                    "source /opt/ros/jazzy/setup.bash && python3 /workspace/integrations/ros_probe/acceptance/pose_observer.py --output /evidence/independent-stop.jsonl",
                ],
                stdout=pose_log,
                stderr=subprocess.STDOUT,
            )
            await asyncio.sleep(2)
            listing = fixture_command(args.container, "ps -eo pid,args")
            pids = [
                int(line.split()[0])
                for line in listing.splitlines()
                if "/acceptance/witness.py" in line
            ]
            if len(pids) != 1:
                raise RuntimeError("exact owned witness process not found")
            stopped_witness = pids[0]
            record["observer_stopped_at"] = datetime.now(UTC).isoformat()
            fixture_command(args.container, f"kill -STOP {stopped_witness}")
            await asyncio.sleep(4)
            independent = [
                json.loads(line)
                for line in (args.fixture / "independent-stop.jsonl").read_text().splitlines()
            ]
            settled = [
                s
                for s in independent
                if (
                    datetime.fromisoformat(s["captured_at"])
                    - datetime.fromisoformat(record["observer_stopped_at"])
                ).total_seconds()
                >= 1
            ]
            record["independent_observer_loss_poses"] = independent
            if (
                len(settled) < 30
                or (
                    datetime.now(UTC) - datetime.fromisoformat(settled[-1]["captured_at"])
                ).total_seconds()
                > 0.3
            ):
                raise RuntimeError("independent stop measurement is incomplete")
            record["observer_loss_displacement_m"] = max(
                math.hypot(s["x"] - settled[0]["x"], s["y"] - settled[0]["y"]) for s in settled
            )
            record["observer_loss_yaw_change_rad"] = max(
                abs(
                    math.atan2(
                        math.sin(s["yaw"] - settled[0]["yaw"]),
                        math.cos(s["yaw"] - settled[0]["yaw"]),
                    )
                )
                for s in settled
            )
            fixture_command(args.container, f"kill -CONT {stopped_witness}")
            stopped_witness = None
            if (
                record["observer_loss_displacement_m"] > 0.01
                or record["observer_loss_yaw_change_rad"] > 0.03
            ):
                raise RuntimeError("controller did not independently stop on observer-process loss")
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
        if args.case not in {"daemon_kill", "daemon_restart"}:
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
        if args.case == "daemon_restart":
            (root / "daemon_ready.json").unlink(missing_ok=True)
            daemon = subprocess.Popen(daemon_argv, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 30
            while not (root / "daemon_ready.json").exists():
                if daemon.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("restarted daemon not ready")
                await asyncio.sleep(0.1)
            restarted = DaemonClient(socket_path=root / "run/rosclawd.sock")
            status = restarted.get_runtime_status()
            record["runtime_after"] = status
            if (
                status["daemon_instance_id"] == record["runtime_before"]["daemon_instance_id"]
                or status["supervision_state"] != "DISARMED"
                or status["ledger"]["integrity_verified"] is not True
                or status["ledger"]["write_failed"]
            ):
                raise RuntimeError(
                    "restart generation, disarmed state or durable ledger gate failed"
                )
            receipt = restarted.get_execution_receipt("fault-coverage")
            record["recovered_canonical_receipt"] = receipt
            canonical = receipt["receipt"]
            if canonical["final_state"] != "FAILED" or not any(
                e.get("code") == "DAEMON_RESTART_INTERRUPTED" for e in canonical["errors"]
            ):
                raise RuntimeError("interrupted SIM action was not durably recovered as failed")
            for name, call in {
                "old_session_heartbeat": lambda: restarted.heartbeat_session(
                    record["ticket"]["session_id"]
                ),
                "old_action_lease": lambda: restarted.renew_action_lease(
                    "fault-coverage", record["ticket"]["session_id"]
                ),
            }.items():
                try:
                    call()
                except DaemonRequestError as exc:
                    expected = (
                        "SESSION_NOT_FOUND"
                        if name == "old_session_heartbeat"
                        else "ACTION_NOT_ACTIVE"
                    )
                    if exc.code != expected:
                        raise RuntimeError(
                            f"unexpected stale-authority rejection: {exc.code}"
                        ) from exc
                    record[name] = {"rejected": True, "code": exc.code, "error": str(exc)}
                else:
                    raise RuntimeError(f"restart accepted stale authority: {name}")
            after = []
            until = time.monotonic() + 3
            while time.monotonic() < until:
                after.append(observation(args.fixture))
                await asyncio.sleep(0.1)
            record["post_restart_observations"] = after
            movement = max(
                math.hypot(s["x"] - after[0]["x"], s["y"] - after[0]["y"]) for s in after
            )
            record["post_restart_displacement_m"] = movement
            rotation = max(
                abs(
                    math.atan2(
                        math.sin(s["yaw"] - after[0]["yaw"]), math.cos(s["yaw"] - after[0]["yaw"])
                    )
                )
                for s in after
            )
            record["post_restart_yaw_change_rad"] = rotation
            if (
                len(after) < 20
                or movement > 0.01
                or rotation > 0.03
                or any(
                    s["cleaning_enabled"] or s["lease_remaining_sec"] > 0 or s["collision_count"]
                    for s in after
                )
            ):
                raise RuntimeError("restart resurrected motion or the cleaning lease")
        record["status"] = "PASS"
    except Exception as exc:
        record["error"] = str(exc)
        raise
    finally:
        if stopped_witness:
            fixture_command(args.container, f"kill -CONT {stopped_witness}")
        if pose_observer:
            listing = fixture_command(args.container, "ps -eo pid,args")
            for line in listing.splitlines():
                if "/acceptance/pose_observer.py" in line:
                    fixture_command(args.container, f"kill -INT {int(line.split()[0])}")
            try:
                pose_observer.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pose_observer.terminate()
            pose_log.close()
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
