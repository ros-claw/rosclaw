"""Native model black-box SIM acceptance using the existing installed PTY harness.

Requires preconfigured isolated model state and SIM-only fixture config. The
runner sends one task input; y answers are the explicit test operator. Physical
execution remains in the separately spawned rosclawd and independent operatord.
No credentials or raw model reasoning are copied into report artifacts.
"""

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]


def validate_fixture_config(config, body_id, endpoint):
    if config.get("agent", {}).get("default_mode") != "SIMULATION" or any(
        s.get("supported_modes") != ["SIMULATION"] for s in config.get("mcp_servers", [])
    ):
        raise ValueError("automated test approval requires the isolated SIM-only fixture")
    if config.get("agent", {}).get("body_id") != body_id or any(
        s.get("action_tools") and body_id not in s.get("required_body_types", [])
        for s in config.get("mcp_servers", [])
    ):
        raise ValueError("Native action declarations do not bind the prepared fixture Body")
    if config.get("agent", {}).get("ros_expert", {}).get("endpoint") != endpoint:
        raise ValueError("Native observer and daemon fixture endpoints differ")


def capture_terminal_counters(root):
    """Read public model counters/task state after owned Native process shutdown."""
    columns = (
        "provider",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "finish_reason",
    )
    with closing(
        sqlite3.connect(f"file:{root / 'home/agentd/missions.db'}?mode=ro", uri=True)
    ) as db:
        rows = db.execute(
            "select provider,model,prompt_tokens,completion_tokens,total_tokens,finish_reason from model_usage limit 257"
        ).fetchall()
        states = db.execute("select state from tasks limit 65").fetchall()
        if len(rows) > 256 or len(states) > 64:
            raise ValueError("isolated Native usage/task evidence exceeds bound")
        (root / "usage.json").write_text(
            json.dumps([dict(zip(columns, row, strict=True)) for row in rows], indent=2) + "\n"
        )
        (root / "task-kernel-final-states.json").write_text(
            json.dumps({"states": [row[0] for row in states]}, indent=2) + "\n"
        )


def negative_native_progress(root, case, receipt_client):
    """Read genuine failed root/transaction/receipt; never declare physics safe."""
    if type(case) is not str or case not in {"D4", "D5"}:
        raise ValueError("explicit D4/D5 negative Native case required")
    root = Path(root).resolve()
    with (root / "execution_config.json").open("rb") as stream:
        config_raw = stream.read(2_000_001)
    if len(config_raw) > 2_000_000:
        raise ValueError("bounded frozen execution configuration required")
    config = json.loads(config_raw)
    if type(config) is not dict or any(
        type(config.get(k)) is not str or not 1 <= len(config[k]) <= 256
        for k in ("body_id", "body_snapshot_hash")
    ):
        raise ValueError("typed bound negative Native configuration required")
    with closing(
        sqlite3.connect(f"file:{root / 'home/agentd/missions.db'}?mode=ro", uri=True)
    ) as db:
        db.row_factory = sqlite3.Row
        tasks = db.execute(
            "select task_id,mission_id,root_goal,body_id,mode,state,terminal_reason from tasks limit 2"
        ).fetchall()
        if len(tasks) != 1:
            raise ValueError("one actual Native root task required")
        task = dict(tasks[0])
        if (
            task["root_goal"] != "完成整个房间清扫。"
            or task["body_id"] != config["body_id"]
            or task["mode"] != "SIMULATION"
            or not task["mission_id"]
        ):
            raise ValueError("negative Native root does not match the isolated SIM task")
        if task["state"] == "SUCCEEDED":
            raise RuntimeError("negative Native task incorrectly reports success")
        if task["state"] not in {"FAILED", "BLOCKED"}:
            return {
                "status": "PENDING_REAL_TERMINAL_STATE",
                "task_state": task["state"],
                "physical_acceptance": "NOT_VERIFIED",
            }
        rows = db.execute(
            "select action_id,capability_id,state,mission_id,body_hash,mode from action_txns "
            "where capability_id='coverage.execute' limit 65"
        ).fetchall()
        if len(rows) > 64:
            raise ValueError("bounded negative Native coverage transactions required")
    retained = []
    for row in rows:
        if (
            row["state"] != "FAILED"
            or row["mission_id"] != task["mission_id"]
            or row["body_hash"] != config["body_snapshot_hash"]
            or row["mode"] != "SIMULATION"
            or type(row["action_id"]) is not str
            or not row["action_id"]
        ):
            continue
        result = receipt_client.get_execution_receipt(row["action_id"])
        if type(result) is not dict:
            raise ValueError("typed negative Native canonical receipt bundle required")
        receipt = result.get("receipt")
        if (
            type(receipt) is not dict
            or result.get("action_id") != row["action_id"]
            or receipt.get("action_id") != row["action_id"]
            or receipt.get("body_id") != config["body_id"]
            or receipt.get("body_snapshot_hash") != config["body_snapshot_hash"]
            or receipt.get("capability_id") != "coverage.execute"
            or receipt.get("execution_mode") != "SIMULATION"
            or receipt.get("final_state") not in {"FAILED", "BLOCKED", "TIMED_OUT"}
        ):
            raise ValueError("negative Native canonical receipt identity or terminal state differs")
        verification = receipt.get("verification_result")
        blocked = receipt["final_state"] == "BLOCKED"
        artifact = verification.get("failure_artifact") if type(verification) is dict else None
        temporal_artifact = blocked and artifact is None
        if temporal_artifact:
            artifact = verification.get("evidence_artifact") if type(verification) is dict else None
        if type(artifact) is not dict or type(artifact.get("path")) is not str:
            raise ValueError("canonical negative receipt requires retained failure artifact")
        path = Path(artifact["path"]).resolve()
        if (
            path.parent != (root / "actions").resolve()
            or (not path.name.endswith(".failed.json") and not temporal_artifact)
            or (
                temporal_artifact
                and (path.suffix != ".json" or path.name.endswith(".verification.json"))
            )
            or not path.is_file()
            or not 0 < path.stat().st_size <= 1_000_000_000
        ):
            raise ValueError("bounded owned failed artifact required")
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if sha != artifact.get("sha256"):
            raise ValueError("canonical retained failure artifact changed")
        if temporal_artifact:
            from rosclaw.connectors.ros.verification.mission import replay_coverage

            evidence = json.loads(path.read_bytes())
            admission = config.get("dynamic_fixture_admission")
            if (
                case != "D4"
                or type(admission) is not dict
                or type(evidence) is not dict
                or evidence.get("schema_version") != "rosclaw.time_paired_mission_evidence.v1"
                or evidence.get("body_id") != config["body_id"]
                or evidence.get("body_snapshot_hash") != config["body_snapshot_hash"]
                or evidence.get("mission_id") != admission.get("mission_id")
                or evidence.get("mission_id") != verification.get("mission_id")
                or evidence.get("action_ids") != [row["action_id"]]
                or evidence.get("occupancy_binding", {}).get("run_id") != admission.get("run_id")
            ):
                raise ValueError(
                    "canonical blocked artifact requires exact dynamic mission/Body/run"
                )
            coverage, temporal = replay_coverage(evidence)
            if (
                temporal is None
                or not temporal["complete"]
                or coverage["coverage_ratio"] >= 0.98
                or temporal != verification.get("time_paired_accounting")
                or coverage["coverage_ratio"] != verification.get("coverage_ratio")
            ):
                raise ValueError(
                    "canonical blocked artifact differs from actual partial accounting"
                )
        retained.append({"capability_id": row["capability_id"], **result})
    if not retained:
        return {
            "status": "PENDING_REAL_FAILED_COVERAGE_RECEIPT",
            "task_state": task["state"],
            "physical_acceptance": "NOT_VERIFIED",
        }
    return {
        "status": "CANONICAL_NEGATIVE_TERMINAL_OBSERVED_NOT_PHYSICS_VERIFIED",
        "case": case,
        "task": task,
        "canonical_receipts": retained,
        "physical_acceptance": "NOT_VERIFIED",
        "requires_independent_stop_and_closed_source_and_scenario_validation": True,
        "task_state_modified": False,
    }


def required_scenario_progress(root, scenario_bytes):
    """Fixture progress only; canonical component/credit/stop gates remain required."""
    from observations import latest_completed_observation

    if not 0 < len(scenario_bytes) <= 65536:
        raise ValueError("bounded frozen scenario required")
    spec = json.loads(scenario_bytes)
    config = json.loads((root / "execution_config.json").read_bytes())
    if not isinstance(spec, dict) or not isinstance(config, dict):
        raise ValueError("scenario and fixture config must be objects")
    binding = config.get("occupancy_binding")
    admission = config.get("dynamic_fixture_admission")
    if not isinstance(binding, dict) or not isinstance(admission, dict):
        raise ValueError("source-admitted dynamic fixture required")
    if (
        spec.get("schema_version") != "rosclaw.dynamic_fixture_scenario.v1"
        or spec.get("case") not in {"D2", "D3", "D4"}
        or not isinstance(spec.get("run_id"), str)
        or not spec["run_id"]
        or not isinstance(spec.get("mission_id"), str)
        or not spec["mission_id"]
        or spec["run_id"] != binding.get("run_id")
        or spec["mission_id"] != admission.get("mission_id")
    ):
        raise ValueError("scenario must bind the source-admitted Native fixture")
    event = latest_completed_observation(root / "dynamic-scenario-events.jsonl")
    if (
        event.get("scenario_sha256") != hashlib.sha256(scenario_bytes).hexdigest()
        or event.get("run_id") != spec["run_id"]
        or event.get("mission_id") != spec["mission_id"]
        or event.get("physical_acceptance") != "NOT_VERIFIED"
    ):
        raise ValueError("required scenario progress identity differs from frozen fixture")
    if event.get("kind") in {"SCENARIO_FAILED", "TASK_RUNNER_STOP_REQUESTED"}:
        raise RuntimeError("required scenario failed or stopped before Native completion")
    return {
        "case": spec["case"],
        "perturbation_complete": event.get("kind")
        == "PERTURBATION_COMPLETE_REQUIRES_NATIVE_AND_CREDIT_VALIDATION",
        "physical_acceptance": "NOT_VERIFIED",
    }


def main():
    sys.path.insert(0, str(REPO))
    from tests.agentd.test_product_journey import PtySession

    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--endpoint", default="ws://127.0.0.1:19090")
    parser.add_argument("--required-scenario", type=Path)
    parser.add_argument("--expected-safe-failure", choices=["D4", "D5"])
    args = parser.parse_args()
    root = args.directory.resolve()
    home = root / "home"
    config = yaml.safe_load((home / "config.yaml").read_text())
    body_id = json.loads((root / "body.json").read_text())["body_id"]
    validate_fixture_config(config, body_id, args.endpoint)
    scenario_bytes = args.required_scenario.read_bytes() if args.required_scenario else None
    if scenario_bytes is not None:
        progress = required_scenario_progress(root, scenario_bytes)
        if args.expected_safe_failure and progress["case"] != args.expected_safe_failure:
            raise ValueError("negative Native expectation differs from required scenario")
    elif args.expected_safe_failure == "D4":
        raise ValueError("D4 negative Native requires its frozen obstruction scenario")
    env = os.environ.copy()
    env["ROSCLAW_HOME"] = str(home)
    env["ROSCLAW_ROS_EXPERT"] = "1"
    env["ROSCLAW_DAEMON_SOCKET"] = str(root / "run/rosclawd.sock")
    (root / "daemon_ready.json").unlink(missing_ok=True)
    log = (root / "daemon.log").open("w")
    daemon = subprocess.Popen(
        [
            sys.executable,
            str(REPO / "integrations/ros_probe/acceptance/daemon.py"),
            "--directory",
            str(root),
            "--endpoint",
            args.endpoint,
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    session = None
    operator = None
    operator_log = None
    try:
        deadline = time.monotonic() + 30
        while not (root / "daemon_ready.json").exists():
            if daemon.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("daemon not ready")
            time.sleep(0.2)
        if not (home / "operatord/operator-identity.json").exists():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rosclaw.entrypoint",
                    "operatord",
                    "enroll",
                    "--home",
                    str(home),
                ],
                env=env,
                check=True,
            )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "rosclaw.entrypoint",
                "operatord",
                "register-daemon",
                "--home",
                str(home),
            ],
            env=env,
            check=True,
        )
        operator_log = (root / "operator.log").open("w")
        operator = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "rosclaw.entrypoint",
                "operatord",
                "start",
                "--home",
                str(home),
                "--no-human-presence-check",
            ],
            env=env,
            stdout=operator_log,
            stderr=subprocess.STDOUT,
        )
        session = PtySession(
            [sys.executable, "-m", "rosclaw.entrypoint", "chat"],
            env,
            log_path=root / "native-pty.log",
            cwd=root,
        )
        session.expect(b"ROSClaw Native Agent", timeout=60)
        session.expect(b"Operator Ready", timeout=60)
        print(
            json.dumps(
                {
                    "stage": "operator_ready",
                    "model_source": "isolated_home_configuration",
                    "input": "完成整个房间清扫。",
                }
            ),
            flush=True,
        )
        session.send("完成整个房间清扫。\r")
        deadline = time.monotonic() + 1800
        cursor = len(session.clean)
        last = time.monotonic()
        approvals = 0
        failed_at = None
        while time.monotonic() < deadline:
            scenario_progress = None
            if scenario_bytes is not None:
                if args.required_scenario.read_bytes() != scenario_bytes:
                    raise ValueError("required frozen scenario changed during Native task")
                scenario_progress = required_scenario_progress(root, scenario_bytes)
            failed = list((root / "actions").glob("*.failed.json"))
            blocked_artifacts = (
                [
                    p
                    for p in (root / "actions").glob("rosevidence_*.json")
                    if not p.name.endswith((".verification.json", ".failed.json"))
                ]
                if args.expected_safe_failure
                else []
            )
            if len(blocked_artifacts) > 64:
                raise ValueError("bounded negative Native artifact collection required")
            terminal_artifact_seen = bool(failed or blocked_artifacts)
            output = session.clean[cursor:]
            if (
                not terminal_artifact_seen
                and failed_at is None
                and "ROSCLAW 授权请求".encode() in output
            ):
                time.sleep(0.5)
                session.send("y")
                cursor = len(session.clean)
                approvals += 1
                print(
                    json.dumps({"stage": "operator_approved_simulation_card", "count": approvals}),
                    flush=True,
                )
            if terminal_artifact_seen:
                if not args.expected_safe_failure:
                    raise RuntimeError(
                        "canonical mission failed: " + json.loads(failed[0].read_text())["error"]
                    )
                # Let the actual failed response reach TaskKernel. This grace
                # does not renew a physical mission deadline or change state.
                failed_at = time.monotonic() if failed_at is None else failed_at
                from rosclaw.daemon.client import DaemonClient

                progress = negative_native_progress(
                    root,
                    args.expected_safe_failure,
                    DaemonClient(socket_path=root / "run/rosclawd.sock", timeout_sec=1),
                )
                if (
                    progress["status"]
                    == "CANONICAL_NEGATIVE_TERMINAL_OBSERVED_NOT_PHYSICS_VERIFIED"
                ):
                    (root / "native-negative-terminal.json").write_text(
                        json.dumps(progress, indent=2) + "\n"
                    )
                    print(
                        json.dumps(
                            {
                                "stage": "negative_native_terminal_observed",
                                "case": args.expected_safe_failure,
                                "task_state": progress["task"]["state"],
                                "physical_acceptance": "NOT_VERIFIED",
                            }
                        ),
                        flush=True,
                    )
                    session.send("/quit\r")
                    return
                if time.monotonic() - failed_at >= 60:
                    raise TimeoutError(
                        "actual negative Native terminal/receipt did not close within60s"
                    )
            artifacts = list((root / "actions").glob("*.verification.json"))
            if artifacts:
                if args.expected_safe_failure:
                    raise RuntimeError("negative Native unexpectedly produced mission verification")
                result = json.loads(artifacts[0].read_text())
                if result["verification_status"] != "PASS":
                    raise RuntimeError("mission not verified")
                if scenario_progress is not None:
                    if (
                        scenario_progress["case"] != "D2"
                        or not scenario_progress["perturbation_complete"]
                    ):
                        raise RuntimeError("Native success requires the completed D2 perturbation")
                    from rosclaw.connectors.ros.verification.dynamic_diagnostics import (
                        analyze_dynamic_credit,
                    )

                    source = artifacts[0].with_name(
                        artifacts[0].name.removesuffix(".verification.json") + ".json"
                    )
                    diagnosis = analyze_dynamic_credit(json.loads(source.read_bytes()))
                    (root / "dynamic-credit-diagnostics.json").write_text(
                        json.dumps(diagnosis, indent=2) + "\n"
                    )
                    if not diagnosis["d2_calculation_withdrawal_and_actual_revisit_present"]:
                        raise RuntimeError(
                            "D2 requires actual enabled revisit of withdrawn unclean cells"
                        )
                print(
                    json.dumps(
                        {
                            "stage": "mission_verified",
                            "coverage": result["coverage"]["coverage_ratio"],
                            "safety": result["safety"],
                        }
                    ),
                    flush=True,
                )
                time.sleep(20)
                from rosclaw.daemon.client import DaemonClient

                client = DaemonClient(socket_path=root / "run/rosclawd.sock")
                receipt_db = sqlite3.connect(home / "agentd/missions.db")
                canonical = []
                for capability, action_id in receipt_db.execute(
                    "select capability_id,action_id from action_txns where state='COMPLETED'"
                ):
                    canonical.append(
                        {"capability_id": capability, **client.get_execution_receipt(action_id)}
                    )
                (root / "canonical-receipts.json").write_text(
                    json.dumps(canonical, indent=2) + "\n"
                )
                if not any(
                    x["capability_id"] == "ros.expert.remember"
                    and x.get("receipt", {}).get("final_state") == "COMPLETED"
                    for x in canonical
                ):
                    raise RuntimeError("verified Memory receipt missing")
                task = receipt_db.execute(
                    "select state from tasks order by created_at desc limit 1"
                ).fetchone()
                if not task or task[0] != "SUCCEEDED":
                    raise RuntimeError("existing TaskKernel is not SUCCEEDED")
                receipt_db.close()
                session.send("/quit\r")
                time.sleep(3)
                db = sqlite3.connect(home / "agentd/missions.db")
                usage = [
                    dict(
                        zip(
                            (
                                "provider",
                                "model",
                                "prompt_tokens",
                                "completion_tokens",
                                "total_tokens",
                                "finish_reason",
                            ),
                            row,
                            strict=True,
                        )
                    )
                    for row in db.execute(
                        "select provider,model,prompt_tokens,completion_tokens,total_tokens,finish_reason from model_usage"
                    )
                ]
                (root / "usage.json").write_text(json.dumps(usage, indent=2) + "\n")
                db.close()
                native_session = max(
                    (home / "agent/sessions").glob("*.jsonl"), key=lambda p: p.stat().st_mtime
                )
                measured = []
                for line in native_session.read_text().splitlines():
                    message = json.loads(line).get("message", {})
                    if message.get("role") == "assistant":
                        measured.append(
                            {
                                k: message.get(k)
                                for k in ("model", "provider", "usage", "stopReason")
                            }
                        )
                (root / "sdk-usage.json").write_text(json.dumps(measured, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "status": "PASS",
                            "model_turns": len(measured),
                            "core_metered_turns": len(usage),
                            "approvals": approvals,
                        }
                    ),
                    flush=True,
                )
                break
            if time.monotonic() - last > 30:
                print(
                    json.dumps(
                        {
                            "stage": "running",
                            "approvals": approvals,
                            "action_artifacts": len(list((root / "actions").glob("*"))),
                        }
                    ),
                    flush=True,
                )
                last = time.monotonic()
            if session.proc.poll() is not None:
                raise RuntimeError("Native Agent exited")
            sessions = list((home / "agent/sessions").glob("*.jsonl"))
            if sessions and failed_at is None:
                latest = max(sessions, key=lambda p: p.stat().st_mtime)
                if time.time() - latest.stat().st_mtime > 3:
                    rows = latest.read_text().splitlines()
                    message = json.loads(rows[-1]).get("message", {}) if rows else {}
                    if message.get("role") == "assistant" and message.get("stopReason") == "stop":
                        raise RuntimeError(
                            "model finished without a verified Memory artifact and TaskKernel success"
                        )
            time.sleep(0.3)
        else:
            raise TimeoutError("native mission")
    finally:
        try:
            sessions = list((home / "agent/sessions").glob("*.jsonl"))
            if sessions:
                latest = max(sessions, key=lambda p: p.stat().st_mtime)
                measured = []
                for line in latest.read_text().splitlines():
                    message = json.loads(line).get("message", {})
                    if message.get("role") == "assistant":
                        measured.append(
                            {
                                k: message.get(k)
                                for k in ("model", "provider", "usage", "stopReason")
                            }
                        )
                (root / "sdk-usage.json").write_text(json.dumps(measured, indent=2) + "\n")
        except (OSError, ValueError, sqlite3.Error):
            # Evidence failures must never bypass process/motion cleanup.
            print("SDK usage capture failed; acceptance evidence is incomplete", file=sys.stderr)
        if session:
            session.stop()
        try:
            capture_terminal_counters(root)
        except (OSError, ValueError, sqlite3.Error):
            print(
                "Core usage/task capture failed; acceptance evidence is incomplete", file=sys.stderr
            )
        try:
            from rosclaw.daemon.client import DaemonClient

            # Also retain failed canonical receipts before stopping the owned
            # daemon. Read-only calls have a one-second bound and cannot hold
            # cleanup indefinitely. Absence stays explicit, never successful.
            receipt_client = DaemonClient(socket_path=root / "run/rosclawd.sock", timeout_sec=1)
            connection = sqlite3.connect(home / "agentd/missions.db")
            try:
                action_rows = connection.execute(
                    "select capability_id, action_id from action_txns where action_id is not null"
                ).fetchall()
            finally:
                connection.close()
            captured = []
            for capability, action_id in action_rows:
                try:
                    receipt = receipt_client.get_execution_receipt(action_id)
                    captured.append({"capability_id": capability, **receipt})
                except Exception:
                    captured.append(
                        {
                            "capability_id": capability,
                            "action_id": action_id,
                            "receipt": None,
                            "capture_status": "UNAVAILABLE",
                        }
                    )
            (root / "canonical-receipts-final.json").write_text(
                json.dumps(captured, indent=2) + "\n"
            )
        except Exception:
            print(
                "Canonical receipt capture incomplete; process cleanup continues", file=sys.stderr
            )
        if operator:
            operator.terminate()
            try:
                operator.wait(timeout=5)
            except subprocess.TimeoutExpired:
                operator.kill()
                operator.wait()
        daemon.terminate()
        try:
            daemon.wait(timeout=10)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait()
        log.close()
        if operator_log:
            operator_log.close()


if __name__ == "__main__":
    main()
