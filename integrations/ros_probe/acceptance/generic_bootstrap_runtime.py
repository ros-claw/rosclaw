"""Deadline-bounded operator SIM bootstrap with inactive controllers only.

This owns an isolated launch process, never admits a Body or dispatches a task.
Actual Graph/TF/sensor/physics admission is still required before activation.
"""

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

from fixture_network import validate_ros_domain
from generic_bootstrap_launch import bootstrap_launch_plan, build_bootstrap_launch_description


def validate_deadline_seconds(seconds):
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 1 <= seconds <= 1800:
        raise ValueError("finite immutable bootstrap deadline within 1..1800 seconds required")
    return seconds


def require_read_only_workspace(directory):
    directory = Path(directory)
    if directory != Path("/evidence") or directory.is_symlink():
        raise ValueError("exact non-symlink /evidence source mount required")
    if not os.statvfs(directory).f_flag & os.ST_RDONLY:
        raise ValueError("actual read-only source mount required")


def require_isolated_environment():
    partition = os.environ.get("GZ_PARTITION", "")
    if not re.fullmatch(r"rosclaw_generic_[a-f0-9]{32}", partition):
        raise ValueError("fresh explicit generic fixture GZ partition required")
    # Validate against this process's actual kernel range before SDK import.
    validate_ros_domain(int(os.environ.get("ROS_DOMAIN_ID", "0")))


def supervise_owned_launch(argv, output, *, seconds, check_source):
    """Own one new process group, refuse any early exit, always reap it.

    The absolute monotonic deadline is created before source checks or Popen;
    neither dependency failures nor cleanup extend the permitted runtime.
    Cleanup has a separate bounded grace period, with no task authorization.
    """
    deadline = time.monotonic() + validate_deadline_seconds(seconds)
    output = Path(output)
    output.mkdir(mode=0o700, exist_ok=False)
    child = None
    outcome = "SOURCE_PREFLIGHT_FAILED"
    try:
        check_source()
        if time.monotonic() >= deadline:
            raise TimeoutError("bootstrap deadline exhausted before launch")
        with (output / "launch.log").open("x") as log:
            child = subprocess.Popen(
                argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            while True:
                if time.monotonic() >= deadline:
                    outcome = "IMMUTABLE_DEADLINE_REACHED"
                    return outcome
                if child.poll() is not None:
                    outcome = "UNEXPECTED_LAUNCH_EXIT"
                    raise RuntimeError(f"owned bootstrap launch exited: {child.returncode}")
                check_source()
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    except BaseException:
        if outcome == "SOURCE_PREFLIGHT_FAILED" and child is not None:
            outcome = "SOURCE_OR_SUPERVISOR_FAILURE"
        raise
    finally:
        if child is not None:
            # Kill the owned group even if the immediate launch parent exited.
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=2)
            # A reaped parent can leave a descendant in the original group.
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
        result = {
            "schema_version": "rosclaw.generic_inactive_bootstrap_runtime.v1",
            "outcome": outcome,
            "owned_launch_pid": child.pid if child is not None else None,
            "owned_launch_reaped": child is not None and child.poll() is not None,
            "returncode": child.returncode if child is not None else None,
            "controller_activation": False,
            "live_body_admitted": False,
            "authorization": False,
            "physical_acceptance": "NOT_EVALUATED",
        }
        (output / "runtime-result.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--sdk-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    require_isolated_environment()
    require_read_only_workspace("/evidence")
    original = args.declaration.read_bytes()
    if not 0 < len(original) <= 65536 or args.declaration.is_symlink():
        raise ValueError("bounded regular bootstrap declaration required")
    declaration = json.loads(original)
    plan = bootstrap_launch_plan("/evidence", declaration)
    frozen_plan = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).digest()
    if args.sdk_child:
        from launch import LaunchService
        from launch.actions import EmitEvent, TimerAction
        from launch.events import Shutdown

        service = LaunchService()
        description = build_bootstrap_launch_description("/evidence", declaration)
        description.add_action(
            TimerAction(
                period=validate_deadline_seconds(args.seconds),
                actions=[EmitEvent(event=Shutdown(reason="owned bootstrap deadline"))],
            )
        )
        service.include_launch_description(description)
        return service.run()
    if args.output is None or args.output.resolve().is_relative_to(Path("/evidence")):
        raise ValueError("fresh writable runtime output outside source mount required")

    def check_source():
        require_read_only_workspace("/evidence")
        if args.declaration.read_bytes() != original:
            raise ValueError("frozen bootstrap declaration changed")
        current = bootstrap_launch_plan("/evidence", declaration)
        if hashlib.sha256(json.dumps(current, sort_keys=True).encode()).digest() != frozen_plan:
            raise ValueError("frozen bootstrap source plan changed")

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    old_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        supervise_owned_launch(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--declaration",
                str(args.declaration),
                "--seconds",
                str(args.seconds),
                "--sdk-child",
            ],
            args.output,
            seconds=args.seconds,
            check_source=check_source,
        )
    finally:
        signal.signal(signal.SIGTERM, old_handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
