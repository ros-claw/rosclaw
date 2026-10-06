"""Native model black-box SIM acceptance using the existing installed PTY harness.

Requires preconfigured isolated model state and SIM-only fixture config. The
runner sends one task input; y answers are the explicit test operator. Physical
execution remains in the separately spawned rosclawd and independent operatord.
No credentials or raw model reasoning are copied into report artifacts.
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
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


def main():
    sys.path.insert(0, str(REPO))
    from tests.agentd.test_product_journey import PtySession

    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--endpoint", default="ws://127.0.0.1:19090")
    args = parser.parse_args()
    root = args.directory.resolve()
    home = root / "home"
    config = yaml.safe_load((home / "config.yaml").read_text())
    body_id = json.loads((root / "body.json").read_text())["body_id"]
    validate_fixture_config(config, body_id, args.endpoint)
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
        while time.monotonic() < deadline:
            output = session.clean[cursor:]
            if "ROSCLAW 授权请求".encode() in output:
                time.sleep(0.5)
                session.send("y")
                cursor = len(session.clean)
                approvals += 1
                print(
                    json.dumps({"stage": "operator_approved_simulation_card", "count": approvals}),
                    flush=True,
                )
            failed = list((root / "actions").glob("*.failed.json"))
            if failed:
                raise RuntimeError(
                    "canonical mission failed: " + json.loads(failed[0].read_text())["error"]
                )
            artifacts = list((root / "actions").glob("*.verification.json"))
            if artifacts:
                result = json.loads(artifacts[0].read_text())
                if result["verification_status"] != "PASS":
                    raise RuntimeError("mission not verified")
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
            if sessions:
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
        except (OSError, ValueError):
            # Evidence failures must never bypass process/motion cleanup.
            print("SDK usage capture failed; acceptance evidence is incomplete", file=sys.stderr)
        if session:
            session.stop()
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
