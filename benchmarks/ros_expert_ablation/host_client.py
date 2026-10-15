"""Fresh tool-disabled CLI turns for externally mediated structured tools.

No repo, SDK, Runtime, ROS transport or MCP server is loaded into this client.
The actual external evaluator applies validated proposals in disposable staging.
"""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

from benchmarks.ros_expert_ablation.isolated_client import DISABLED_FEATURES, EFFORT, MODEL
from benchmarks.ros_expert_ablation.protocol import usage_from_events


def invoke(prompt, schema, output, *, timeout_sec=120):
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    (output / "request.txt").write_text(prompt)
    schema_path = output / "output-schema.json"
    schema_path.write_text(json.dumps(schema, indent=2) + "\n")
    binary = Path(
        "/home/nvidia/.codex/packages/standalone/releases/0.162.1-aarch64-unknown-linux-musl/bin/codex"
    )
    auth = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "auth.json"
    with tempfile.TemporaryDirectory(prefix="reh-g5-private-host-client-") as directory:
        private = Path(directory)
        private.chmod(0o700)
        shutil.copyfile(auth, private / "auth.json")
        (private / "auth.json").chmod(0o600)
        argv = [str(binary)]
        for feature in DISABLED_FEATURES:
            argv.extend(["-c", f"features.{feature}=false"])
        argv.extend(
            [
                "-c",
                'web_search="disabled"',
                "-c",
                f'model_reasoning_effort="{EFFORT}"',
                "exec",
                "--ignore-user-config",
                "--ignore-rules",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--json",
                "--color",
                "never",
                "--model",
                MODEL,
                "--output-schema",
                str(schema_path),
                "-",
            ]
        )
        (output / "argv.json").write_text(json.dumps(argv, indent=2) + "\n")
        with (
            (output / "original-events.jsonl").open("xb") as stdout,
            (output / "original-stderr.txt").open("xb") as stderr,
        ):
            proc = subprocess.Popen(
                argv,
                cwd=private,
                env=dict(os.environ, CODEX_HOME=directory),
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            timed_out = False
            try:
                proc.communicate(prompt.encode(), timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
        events = [
            json.loads(line)
            for line in (output / "original-events.jsonl").read_text().splitlines()
            if line.strip()
        ]
        try:
            usage = usage_from_events(events)
        except ValueError:
            usage = None
        native_items = [
            e.get("item", {}).get("type")
            for e in events
            if e.get("type") == "item.completed"
            and e.get("item", {}).get("type") not in {"context", "agent_message", "reasoning"}
        ]
        outcome = {
            "status": "ACTUAL_MODEL_TURN_COMPLETED"
            if proc.returncode == 0 and usage is not None
            else "MODEL_CLIENT_TIMEOUT"
            if timed_out
            else "MODEL_CLIENT_FAILED_OR_USAGE_MISSING",
            "returncode": proc.returncode,
            "requested_model": MODEL,
            "effort": EFFORT,
            "client_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "client_version": "0.162.1",
            "model_revision": None,
            "temperature": None,
            "sampling_policy": "CLIENT_PROVIDER_DEFAULT; no numeric temperature asserted",
            "usage": usage,
            "cost": None,
            "cost_reason": "not reported by this CLI",
            "client_scope": "HOST_PROCESS_READ_ONLY_NO_NATIVE_EXECUTION_TOOLS_NO_MCP; externally mediated tools only",
            "container_isolation": False,
            "raw_event_types": [e.get("type") for e in events],
            "native_tool_items": native_items,
        }
        if native_items:
            outcome["status"] = "NATIVE_TOOL_PROTOCOL_VIOLATION"
        (output / "outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")
        return outcome
