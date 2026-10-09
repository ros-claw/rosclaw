"""Bounded cleanup of one exact, independently owned SIM container.

Container ownership does not admit a Body or authorize any robot action.
Unlike process-group cleanup, container teardown includes descendants that
created their own sessions. Never select a container by a name prefix.
"""

import json
import re
import subprocess
import time
from pathlib import Path

OWNER_LABEL = "org.rosclaw.generic-bootstrap-owner"
KIND_LABEL = "org.rosclaw.fixture-kind"
KIND = "generic-inactive-bootstrap"


def inspect_owned_container(container_id, owner, *, timeout=5):
    if (
        type(container_id) is not str
        or not re.fullmatch(r"[a-f0-9]{64}", container_id)
        or type(owner) is not str
        or not re.fullmatch(r"[a-f0-9]{32}", owner)
    ):
        raise ValueError("exact full container ID and private generated ownership token required")
    reply = subprocess.run(
        ["docker", "inspect", "--type", "container", container_id],
        capture_output=True,
        check=True,
        timeout=timeout,
    )
    values = json.loads(reply.stdout)
    if type(values) is not list or len(values) != 1:
        raise ValueError("one exact original Docker inspection required")
    row = values[0]
    labels = row.get("Config", {}).get("Labels") or {}
    if (
        row.get("Id") != container_id
        or labels.get(OWNER_LABEL) != owner
        or labels.get(KIND_LABEL) != KIND
    ):
        raise ValueError("foreign or unknown container ownership; no mutation permitted")
    return row


def stop_owned_container(container_id, owner, output):
    """Verify before every mutation; unknown cleanup never becomes STOPPED."""
    output = Path(output)
    attempts = []
    report = {
        "schema_version": "rosclaw.owned_inactive_container_cleanup.v1",
        "container_id": container_id,
        "ownership_verified": False,
        "container_stopped": False,
        "attempts": attempts,
        "live_body_admitted": False,
        "authorization": False,
        "physical_stop_proof": "NOT_MEASURED",
    }
    try:
        row = inspect_owned_container(container_id, owner)
        report["ownership_verified"] = True
        for operation in (["stop", "--time", "1"], ["kill"]):
            if row["State"]["Running"] is False and row["State"]["Pid"] == 0:
                break
            # Recheck the exact immutable ID and labels immediately before mutation.
            inspect_owned_container(container_id, owner)
            try:
                reply = subprocess.run(
                    ["docker", *operation, container_id], capture_output=True, timeout=5
                )
                attempts.append(
                    {
                        "operation": operation[0],
                        "returncode": reply.returncode,
                        "stdout": reply.stdout.decode(errors="replace"),
                        "stderr": reply.stderr.decode(errors="replace"),
                    }
                )
            except subprocess.TimeoutExpired:
                attempts.append({"operation": operation[0], "error": "Docker request timeout"})
            row = inspect_owned_container(container_id, owner)
        report["final_original_inspection"] = row
        report["container_stopped"] = row["State"]["Running"] is False and row["State"]["Pid"] == 0
        if not report["container_stopped"]:
            raise RuntimeError("owned container teardown is not verified")
        return report
    except BaseException as error:
        report["error"] = type(error).__name__ + ":" + str(error)
        raise
    finally:
        with output.open("x") as stream:
            json.dump(report, stream, indent=2)


def wait_owned_container(container_id, owner, *, deadline, check_source):
    """Observe a caller's absolute deadline; it is never renewed by polling."""
    import math

    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("finite absolute monotonic deadline required")
    while True:
        if time.monotonic() >= deadline:
            return "IMMUTABLE_DEADLINE_REACHED"
        check_source()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "IMMUTABLE_DEADLINE_REACHED"
        try:
            row = inspect_owned_container(container_id, owner, timeout=min(5, remaining))
        except subprocess.TimeoutExpired:
            if time.monotonic() >= deadline:
                return "IMMUTABLE_DEADLINE_REACHED"
            raise
        if row["State"]["Running"] is not True:
            raise RuntimeError("required owned bootstrap container exited before deadline")
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
