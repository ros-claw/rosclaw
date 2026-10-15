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

from simulation_execution_fault_policy import (
    BEFORE_FAULT_JOURNEY_STATUS,
    MOTION_FAULT_JOURNEY_STATUS,
    parse_governed_before_dispatch_fault_policy,
    parse_governed_motion_fault_policy,
)
from simulation_fault_status_poll import permits_unavailable_status_retry
from simulation_source_capture import BoundedSourceJsonlCapture

from rosclaw.daemon.client import DaemonClient
from rosclaw.mcp import tools
from rosclaw.mcp.adapters.runtime_client import RuntimeClient
from rosclaw.mcp.schemas.common import MCPError


def read_nonrenewing_action_status(client, action_id):
    """The owner observes lease expiry without DaemonClient's automatic renewal."""
    return client.call("action.status", {"action_id": action_id})


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal-policy-sha256", required=True)
    parser.add_argument("--fault-policy-sha256")
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
    fault_policy = None
    if args.fault_policy_sha256:
        fault_raw = (evidence / "governed-fault.json").read_bytes()
        parse_fault = (
            parse_governed_before_dispatch_fault_policy
            if json.loads(fault_raw).get("phase") == "before_dispatch"
            else parse_governed_motion_fault_policy
        )
        fault_policy = parse_fault(
            fault_raw,
            expected_sha256=args.fault_policy_sha256,
            run_id=policy["run_id"],
            body_snapshot_hash=base["body_snapshot_hash"],
            world_sha256=facts["world_sha256"],
            action_id=policy["goals"][0]["action_id"],
        )
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
    if fault_policy and fault_policy["case"] == "rosbridge_disconnect":
        argv += ["--operator-rosbridge-disconnect-sha256", args.fault_policy_sha256]
    if fault_policy and fault_policy["case"] == "wrong_run_id":
        argv += ["--operator-wrong-run-context-sha256", args.fault_policy_sha256]
    (evidence / "governed-daemon-argv.json").write_text(json.dumps(argv) + "\n")
    os.environ["ROSCLAW_HOME"] = str(root / "home")
    os.environ["ROSCLAW_ROS_EXPERT"] = "1"
    result, error = [], None
    actor = None
    lease_probe = None
    lease_probe_log = None
    fault_checks = None
    last_request = None
    emergency_stop_result = None
    fault_receipt = None
    lease_renewal_stopped = False
    nonrenewing_status_capture = None

    def export_public_action_artifacts():
        target = evidence / "public-action-artifacts"
        target.mkdir(exist_ok=True)
        index = []
        source = root / "actions"
        if source.is_dir():
            for file in sorted(source.iterdir()):
                if (
                    not file.is_file()
                    or file.is_symlink()
                    or not file.name.startswith(("rosevidence_", "coverage-audit-"))
                    or file.suffix not in {".json", ".jsonl"}
                ):
                    continue
                raw = file.read_bytes()
                destination = target / file.name
                if destination.exists() and destination.read_bytes() != raw:
                    raise ValueError("already exported original action artifact changed")
                if not destination.exists():
                    with destination.open("xb") as output:
                        output.write(raw)
                index.append(
                    {
                        "source": str(file),
                        "public_file": file.name,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw),
                    }
                )
        (evidence / "public-action-artifacts-index.json").write_text(
            json.dumps(index, indent=2) + "\n"
        )

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
            if fault_policy:
                probe_argv = [
                    sys.executable,
                    str(Path(__file__).with_name("governed_old_lease_probe.py")),
                    "--policy-sha256",
                    args.fault_policy_sha256,
                ]
                lease_probe_log = (evidence / "governed-old-lease-probe-original.log").open("x")
                lease_probe = subprocess.Popen(
                    probe_argv, stdout=lease_probe_log, stderr=subprocess.STDOUT
                )
                setup_deadline = time.monotonic_ns() + 10_000_000_000
                while not (evidence / "governed-old-lease-probe-ready-original.json").exists():
                    if lease_probe.poll() is not None or time.monotonic_ns() >= setup_deadline:
                        raise ValueError(
                            "passive SDK lease probe did not initialize before canonical action"
                        )
                    await asyncio.sleep(0.05)
                start_time = (
                    Path(f"/proc/{daemon.pid}/stat").read_text().split(")", 1)[1].split()[19]
                )
                actor_argv = [
                    sys.executable,
                    str(
                        Path(__file__).with_name(
                            "governed_before_dispatch_fault_fixture.py"
                            if fault_policy["phase"] == "before_dispatch"
                            else "governed_daemon_sigkill_fixture.py"
                        )
                    ),
                    "--policy-sha256",
                    args.fault_policy_sha256,
                    "--daemon-pid",
                    str(daemon.pid),
                    "--daemon-start-time",
                    start_time,
                ]
                (evidence / "governed-fault-actor-argv-original.json").write_text(
                    json.dumps(actor_argv) + "\n"
                )
                actor = subprocess.Popen(actor_argv)
                setup_deadline = time.monotonic_ns() + 10_000_000_000
                while not (evidence / "governed-fault-actor-ready-original.json").exists():
                    if actor.poll() is not None or time.monotonic_ns() >= setup_deadline:
                        raise ValueError(
                            "SDK operator actor did not initialize before canonical action"
                        )
                    await asyncio.sleep(0.05)

            for index, goal in enumerate(policy["goals"]):
                if fault_policy and index > 0:
                    break
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
                if fault_policy and fault_policy["phase"] == "before_dispatch":
                    original = json.loads(json.dumps(request))
                    if fault_policy["case"] == "wrong_body_hash":
                        request["body_snapshot_hash"] = "0" * 64
                    elif fault_policy["case"] == "out_of_bounds_goal":
                        body = json.loads(
                            (
                                evidence / "generic-compiled/body/refs/effective_body.json"
                            ).read_text()
                        )
                        request["arguments"] = json.loads(json.dumps(request["arguments"]))
                        request["arguments"]["pose"]["pose"]["position"]["x"] += (
                            2 * body["safety"]["navigation_simulation"]["max_goal_distance_m"]
                        )
                    with (evidence / "governed-fault-injection-original.json").open("x") as stream:
                        json.dump(
                            {
                                "case": fault_policy["case"],
                                "phase": "before_dispatch",
                                "run_id": policy["run_id"],
                                "body_snapshot_hash": base["body_snapshot_hash"],
                                "before_monotonic_ns": time.monotonic_ns(),
                                "original_valid_request": original,
                                "actual_invalid_request": request,
                                "canonical_request_modified": request != original,
                                "fault_location": "SDK_operator_admission.run_id"
                                if fault_policy["case"] == "wrong_run_id"
                                else "canonical_action_request",
                            },
                            stream,
                            indent=2,
                        )
                (evidence / f"governed-action-{index}-request-original.json").write_text(
                    json.dumps(request, indent=2) + "\n"
                )
                last_request = request
                try:
                    request_started_ns = time.monotonic_ns()
                    ticket = await tools._request_action(**request)
                except Exception as request_error:
                    (
                        evidence / f"governed-action-{index}-request-rejected-original.json"
                    ).write_text(
                        json.dumps(
                            {
                                "original_error": str(request_error),
                                "before_monotonic_ns": request_started_ns,
                                "after_monotonic_ns": time.monotonic_ns(),
                            }
                        )
                        + "\n"
                    )
                    raise
                (evidence / f"governed-action-{index}-ticket-original.json").write_text(
                    json.dumps(ticket, indent=2) + "\n"
                )
                if fault_policy and fault_policy["case"] == "lease_loss":
                    if (
                        ticket["action_lease"]["ttl_ms"] != 10_000
                        or ticket["action_lease"]["renew_interval_ms"] != 3_000
                    ):
                        raise ValueError(
                            "original canonical lease defaults changed; do not shorten them"
                        )
                    nonrenewing_status_capture = BoundedSourceJsonlCapture(
                        evidence / "governed-nonrenewing-action-status-original.jsonl",
                        max_segments=1,
                        max_bytes=10_000_000,
                        max_rows=500,
                        max_line_bytes=64_000,
                    )
                deadline = time.monotonic() + goal["timeout_sec"] + 5
                unavailable_status_reads = 0
                while True:
                    if daemon.poll() is not None or time.monotonic() >= deadline:
                        raise TimeoutError("canonical governed SIM action did not terminate")
                    lease_trigger = evidence / "governed-lease-loss-trigger-original.json"
                    if (
                        fault_policy
                        and fault_policy["case"] == "lease_loss"
                        and not lease_renewal_stopped
                        and lease_trigger.exists()
                    ):
                        trigger = json.loads(lease_trigger.read_text())
                        if trigger["policy_sha256"] != args.fault_policy_sha256:
                            raise ValueError("original motion-triggered lease-loss context changed")
                        observed = await asyncio.to_thread(
                            read_nonrenewing_action_status, client, goal["action_id"]
                        )
                        if (
                            observed["action_lease"]["last_renewed_at"]
                            != ticket["action_lease"]["last_renewed_at"]
                        ):
                            raise ValueError(
                                "first physical motion followed an automatic renewal; do not alter the default lease"
                            )
                        lease_renewal_stopped = True
                        (evidence / "governed-lease-renewal-stopped-original.json").write_text(
                            json.dumps(
                                {
                                    "monotonic_ns": time.monotonic_ns(),
                                    "original_status_at_stop": observed,
                                    "action_lease_ttl_changed": False,
                                    "policy_sha256": args.fault_policy_sha256,
                                },
                                indent=2,
                            )
                            + "\n"
                        )
                    trigger_path = evidence / "governed-emergency-stop-trigger-original.json"
                    if (
                        fault_policy
                        and fault_policy["case"] == "emergency_stop"
                        and emergency_stop_result is None
                        and trigger_path.exists()
                    ):
                        trigger = json.loads(trigger_path.read_text())
                        if trigger["policy_sha256"] != args.fault_policy_sha256:
                            raise ValueError("original motion-triggered E-stop context changed")
                        started_ns = time.monotonic_ns()
                        emergency_stop_result = await tools._emergency_stop(
                            "owned disposable SIM during-motion acceptance " + policy["run_id"]
                        )
                        if (
                            emergency_stop_result.get("requested_source") != "mcp.emergency_stop"
                            or emergency_stop_result.get("authenticated_peer", {}).get("pid")
                            != os.getpid()
                        ):
                            raise ValueError(
                                "actual canonical MCP daemon-authenticated E-stop required"
                            )
                        (evidence / "governed-emergency-stop-MCP-original.json").write_text(
                            json.dumps(
                                {
                                    "before_monotonic_ns": started_ns,
                                    "after_monotonic_ns": time.monotonic_ns(),
                                    "original_response": emergency_stop_result,
                                },
                                indent=2,
                            )
                            + "\n"
                        )
                    if lease_renewal_stopped:
                        action_status = await asyncio.to_thread(
                            read_nonrenewing_action_status, client, goal["action_id"]
                        )
                        nonrenewing_status_capture.write(
                            json.dumps(
                                {
                                    "monotonic_ns": time.monotonic_ns(),
                                    "method": "action.status",
                                    "renews_lease": False,
                                    "original_status": action_status,
                                }
                            )
                            + "\n"
                        )
                    else:
                        try:
                            action_status = await tools._get_action_status(goal["action_id"])
                        except MCPError as unavailable:
                            effect_path = (
                                evidence / "governed-rosbridge-disconnect-effect-original.json"
                            )
                            if (
                                fault_policy is None
                                or fault_policy["case"] != "rosbridge_disconnect"
                                or not effect_path.is_file()
                                or effect_path.stat().st_size > 32000
                                or daemon.poll() is not None
                                or time.monotonic() >= deadline
                                or unavailable_status_reads >= 32
                            ):
                                raise
                            effect = json.loads(effect_path.read_text())
                            if not permits_unavailable_status_retry(
                                effect,
                                fault_policy,
                                ready,
                                policy_sha256=args.fault_policy_sha256,
                                error=unavailable,
                                now_ns=time.monotonic_ns(),
                            ):
                                raise
                            unavailable_status_reads += 1
                            path = (
                                evidence
                                / f"governed-action-{index}-status-unavailable-{unavailable_status_reads:03}-original.json"
                            )
                            with path.open("x") as stream:
                                json.dump(
                                    {
                                        "status": "UNKNOWN_ORIGINAL_READ_UNAVAILABLE",
                                        "original_error": str(unavailable),
                                        "monotonic_ns": time.monotonic_ns(),
                                        "unchanged_goal_observation_deadline": deadline,
                                        "receipt_fabricated": False,
                                    },
                                    stream,
                                )
                            await asyncio.sleep(min(0.1, max(0, deadline - time.monotonic())))
                            continue
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "canonical governed SIM action observation deadline expired"
                        )
                    if action_status["state"] in {"FINISHED", "CANCELLED"}:
                        if lease_renewal_stopped:
                            await tools._get_action_status(goal["action_id"])
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
            if fault_policy:
                try:
                    actor_code = None
                    if actor is not None:
                        try:
                            actor_code = actor.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            actor.kill()
                            actor_code = actor.wait(timeout=2)
                    missing = not (evidence / "governed-action-0-receipt-original.json").exists()
                    if (
                        fault_policy["case"]
                        in {
                            "emergency_stop",
                            "lease_loss",
                            "lidar_stream_lost",
                            "rosbridge_disconnect",
                            "collision_monitor_disabled",
                            "out_of_bounds_goal",
                            "wrong_body_hash",
                            "wrong_run_id",
                            "entity_replaced",
                        }
                        and actor_code == 0
                    ):
                        try:
                            drained = client.get_execution_receipt(policy["goals"][0]["action_id"])
                            (
                                evidence
                                / "governed-action-0-receipt-after-fault-drain-original.json"
                            ).write_text(json.dumps(drained, indent=2) + "\n")
                            fault_receipt = drained.get("receipt")
                            missing = not isinstance(fault_receipt, dict)
                            if not missing and not result:
                                result.append(
                                    {
                                        "action_id": fault_receipt["action_id"],
                                        "final_state": fault_receipt["final_state"],
                                        "verification": fault_receipt["verification_result"],
                                    }
                                )
                        except Exception as drain_error:
                            missing = True
                            (
                                evidence / "governed-terminal-receipt-drain-error-original.json"
                            ).write_text(
                                json.dumps(
                                    {
                                        "original_error": str(drain_error),
                                        "monotonic_ns": time.monotonic_ns(),
                                    }
                                )
                                + "\n"
                            )
                    (evidence / "governed-missing-terminal-receipt-original.json").write_text(
                        json.dumps(
                            {
                                "action_id": policy["goals"][0]["action_id"],
                                "terminal_receipt_missing": missing,
                                "case": fault_policy["case"],
                                "reason": "original observation; no terminal receipt fabricated",
                                "original_error": error,
                                "monotonic_ns": time.monotonic_ns(),
                            }
                        )
                        + "\n"
                    )
                    old_action_refused = False
                    if actor_code == 0 and last_request is not None:
                        probe_code = lease_probe.wait(timeout=8)
                        try:
                            old_action_result = await tools._request_action(**last_request)
                            refusal = {"refused": False, "unexpected_result": old_action_result}
                        except Exception as exc:
                            old_action_refused = True
                            refusal = {"refused": True, "original_error": str(exc)[:2500]}
                        refusal.update(
                            action_id=last_request["action_id"],
                            monotonic_ns=time.monotonic_ns(),
                            purpose="explicit old-action rejection test after confirmed "
                            + fault_policy["case"]
                            + " and independent stop",
                        )
                        (evidence / "governed-old-action-refusal-original.json").write_text(
                            json.dumps(refusal, indent=2) + "\n"
                        )
                        fault_checks = {
                            "actor_returncode": actor_code,
                            "old_lease_probe_returncode": probe_code,
                            "old_action_refused": old_action_refused,
                            "terminal_receipt_missing": missing,
                        }
                        if fault_policy["phase"] == "before_dispatch":
                            fault_checks["actual_before_dispatch_unadmitted_refusal"] = True
                        if fault_policy["case"] == "emergency_stop":
                            status = await tools._get_runtime_status()
                            (
                                evidence / "governed-emergency-stop-latched-status-original.json"
                            ).write_text(json.dumps(status, indent=2) + "\n")
                            fault_checks["canonical_emergency_stop_latched"] = (
                                status["emergency_stop_latched"] is True
                                and status["supervision_state"] == "ESTOPPED"
                                and status["daemon_instance_id"] == ready["daemon_instance_id"]
                                and emergency_stop_result is not None
                            )
                        elif fault_policy["case"] == "lease_loss":
                            fault_checks["canonical_action_lease_expired"] = (
                                lease_renewal_stopped
                                and fault_receipt is not None
                                and "ACTION_LEASE_EXPIRED" in json.dumps(fault_receipt)
                                and emergency_stop_result is None
                            )
                        elif fault_policy["case"] == "lidar_stream_lost":
                            actor_result = json.loads(
                                (
                                    evidence / "governed-fault-actor-outcome-original.json"
                                ).read_text()
                            )
                            effect = actor_result["fault_effect"]
                            fault_checks["original_lidar_stream_loss_observed"] = (
                                effect["original_lidar_signal_before_resume"]["last_message_age_ms"]
                                > 500
                                and effect["actual_owned_scan_bridge_pause"]["label"]
                                == "scan_bridge"
                                and emergency_stop_result is None
                                and not lease_renewal_stopped
                            )
                        elif fault_policy["case"] == "rosbridge_disconnect":
                            effect = json.loads(
                                (
                                    evidence / "governed-rosbridge-disconnect-effect-original.json"
                                ).read_text()
                            )
                            fault_checks["original_navigation_sockets_disconnected"] = (
                                len(effect["original_navigation_sockets_after"]) == 4
                                and all(
                                    x["closed"] and x["websocket_released"]
                                    for x in effect["original_navigation_sockets_after"]
                                )
                                and effect["new_connections_opened"] == 0
                                and effect["physics_server_signals_sent"] == 0
                                and emergency_stop_result is None
                                and not lease_renewal_stopped
                            )
                        elif fault_policy["case"] == "collision_monitor_disabled":
                            effect = json.loads(
                                (
                                    evidence / "governed-fault-actor-outcome-original.json"
                                ).read_text()
                            )["fault_effect"]
                            fault_checks["original_collision_monitor_disabled_and_restored"] = (
                                effect["actual_collision_monitor_deactivation"]["actual_state_id"]
                                == 2
                                and effect["restoration_after_independent_terminal_stop"][
                                    "actual_state_id"
                                ]
                                == 3
                                and emergency_stop_result is None
                                and not lease_renewal_stopped
                            )
                        elif fault_policy["case"] == "entity_replaced":
                            effect = json.loads(
                                (
                                    evidence / "governed-fault-actor-outcome-original.json"
                                ).read_text()
                            )["fault_effect"]["actual_original_model_replacement"]
                            fault_checks["actual_entity_identity_changed_without_reenrollment"] = (
                                effect["original_body_removed"] is True
                                and effect["replacement_body_reenrolled"] is False
                                and effect["original_entity_id"] != effect["replacement_entity_id"]
                                and emergency_stop_result is None
                                and not lease_renewal_stopped
                            )
                    else:
                        fault_checks = {
                            "actor_returncode": actor_code,
                            "terminal_receipt_missing": missing,
                            "old_action_refused": False,
                            "old_lease_probe_returncode": None,
                        }
                except Exception as fault_error:
                    fault_checks = {"acceptance_error": str(fault_error)[:2500]}
            if actor is not None and actor.poll() is None:
                actor.terminate()
                try:
                    actor.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    actor.kill()
                    actor.wait(timeout=2)
            if lease_probe is not None and lease_probe.poll() is None:
                lease_probe.terminate()
                try:
                    lease_probe.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    lease_probe.kill()
                    lease_probe.wait(timeout=2)
            if lease_probe_log is not None:
                lease_probe_log.close()
            if nonrenewing_status_capture is not None:
                nonrenewing_status_capture.close()
            (evidence / "governed-owner-shutdown-request-original.json").write_text(
                json.dumps(
                    {
                        "monotonic_ns": time.monotonic_ns(),
                        "completed_goals": sum(row["final_state"] == "COMPLETED" for row in result),
                        "terminal_goals": len(result),
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
            export_public_action_artifacts()
    before_pass = bool(
        fault_policy
        and fault_policy["phase"] == "before_dispatch"
        and code == 0
        and fault_checks
        and fault_checks.get("actor_returncode") == 0
        and fault_checks.get("old_lease_probe_returncode") == 0
        and fault_checks.get("old_action_refused")
        and fault_checks.get("actual_before_dispatch_unadmitted_refusal")
        and not fault_checks.get("terminal_receipt_missing")
        and isinstance(fault_receipt, dict)
        and fault_receipt.get("final_state")
        == ("FAILED" if fault_policy["case"] == "wrong_run_id" else "BLOCKED")
    )
    fault_pass = before_pass or bool(
        fault_policy
        and fault_policy["phase"] == "during_motion"
        and code == (-signal.SIGKILL if fault_policy["case"] == "daemon_sigkill" else 0)
        and fault_checks
        and fault_checks.get("actor_returncode") == 0
        and fault_checks.get("old_lease_probe_returncode") == 0
        and fault_checks.get("old_action_refused")
        and (
            fault_checks.get("terminal_receipt_missing")
            if fault_policy["case"] == "daemon_sigkill"
            else (
                fault_checks.get("canonical_emergency_stop_latched")
                if fault_policy["case"] == "emergency_stop"
                else fault_checks.get("canonical_action_lease_expired")
                if fault_policy["case"] == "lease_loss"
                else fault_checks.get("original_lidar_stream_loss_observed")
                if fault_policy["case"] == "lidar_stream_lost"
                else fault_checks.get("original_collision_monitor_disabled_and_restored")
                if fault_policy["case"] == "collision_monitor_disabled"
                else fault_checks.get("actual_entity_identity_changed_without_reenrollment")
                if fault_policy["case"] == "entity_replaced"
                else fault_checks.get("original_navigation_sockets_disconnected")
            )
            and not fault_checks.get("terminal_receipt_missing")
            and isinstance(fault_receipt, dict)
            and fault_receipt.get("final_state")
            in (
                {"FAILED", "CANCELLED", "TIMED_OUT"}
                if fault_policy["case"] == "lease_loss"
                else {"FAILED", "CANCELLED"}
            )
        )
    )
    outcome = {
        "status": "FAILED_ORIGINAL_GOVERNED_SIM_JOURNEY"
        if error or code != 0
        else "PASS_TWO_CANONICAL_GOVERNED_SIM_GOALS_LOCAL_ONLY",
        "error": error,
        "daemon_returncode": code,
        "fault_checks": fault_checks,
        "goals": result,
        "G3_done": False,
        "v1_done": False,
    }
    if fault_policy:
        outcome["status"] = (
            {**MOTION_FAULT_JOURNEY_STATUS, **BEFORE_FAULT_JOURNEY_STATUS}[fault_policy["case"]]
            if fault_pass
            else "FAILED_ORIGINAL_EXECUTION_FAULT_ACCEPTANCE"
        )
    with (evidence / "governed-journey-original-outcome.json").open("x") as stream:
        stream.write(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome), flush=True)
    if (fault_policy and not fault_pass) or (not fault_policy and (error or code != 0)):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
