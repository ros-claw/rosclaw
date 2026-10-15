"""Actual Codex JSONL client in a scratch container without ROS or host tools.

The caller supplies structured task/tool transcripts. Physical validation stays
outside this container with independent SDK fixtures and canonical rosclawd.
"""

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from benchmarks.ros_expert_ablation.protocol import usage_from_events

IMAGE = "sha256:4f085522bb37681888ab817c4dabadae74a15e59207a58071dbd6bfe68e0bc02"
MODEL = "gpt-6.1-sol"
EFFORT = "medium"
DISABLED_FEATURES = (
    "shell_tool",
    "apps",
    "multi_agent",
    "multi_agent_v2",
    "browser_use",
    "computer_use",
    "code_mode_host",
    "external_agent_memory_import",
)


def invoke(prompt, schema, output, *, timeout_sec=120):
    """One actual original model turn; preserve failure/usage without retries."""
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    (output / "request.txt").write_text(prompt)
    (output / "output-schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    auth = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "auth.json"
    if not auth.is_file():
        raise RuntimeError("actual Codex client authentication unavailable")
    with tempfile.TemporaryDirectory(prefix="reh-g5-private-client-") as private:
        home = Path(private)
        home.chmod(0o700)
        shutil.copyfile(auth, home / "auth.json")
        (home / "auth.json").chmod(0o600)
        invocation_id = uuid.uuid4().hex
        container_name = "reh-g5-model-client-" + invocation_id
        argv = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--name",
            container_name,
            "--label",
            "rosclaw.g5.invocation=" + invocation_id,
            "--network",
            "bridge",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--memory",
            "2g",
            "--cpus",
            "2",
            "--pids-limit",
            "64",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=128m",
            "--env",
            "HOME=/tmp",
            "--env",
            "CODEX_HOME=/client",
            "--mount",
            f"type=bind,src={home},dst=/client",
            "--mount",
            f"type=bind,src={output / 'output-schema.json'},dst=/schema.json,readonly",
            IMAGE,
        ]
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
                "/schema.json",
                "-",
            ]
        )
        # Do not persist the private credential directory in public provenance.
        public_argv = [v.replace(str(home), "<EPHEMERAL_PRIVATE_AUTH_DIRECTORY>") for v in argv]
        (output / "argv.json").write_text(json.dumps(public_argv, indent=2) + "\n")
        with (
            (output / "original-events.jsonl").open("xb") as stdout,
            (output / "original-stderr.txt").open("xb") as stderr,
        ):
            try:
                result = subprocess.run(
                    argv,
                    input=prompt.encode(),
                    stdout=stdout,
                    stderr=stderr,
                    timeout=timeout_sec,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                (output / "outcome.json").write_text(
                    json.dumps({"status": "MODEL_CLIENT_TIMEOUT", "usage": None}) + "\n"
                )
                raise
            finally:
                # Killing a docker CLI on timeout does not kill its container.
                # Remove only the exact invocation-owned container before the
                # ephemeral credential directory leaves this context.
                inspected = subprocess.run(
                    ["docker", "inspect", container_name], capture_output=True, timeout=15
                )
                if inspected.returncode == 0:
                    record = json.loads(inspected.stdout)[0]
                    if record["Config"]["Labels"].get("rosclaw.g5.invocation") != invocation_id:
                        raise RuntimeError("model client container ownership changed")
                    subprocess.run(
                        ["docker", "rm", "-f", record["Id"]],
                        capture_output=True,
                        timeout=15,
                        check=True,
                    )
        events = [
            json.loads(line)
            for line in (output / "original-events.jsonl").read_text().splitlines()
            if line.strip()
        ]
        try:
            usage = usage_from_events(events)
        except ValueError:
            usage = None
        outcome = {
            "status": "ACTUAL_MODEL_TURN_COMPLETED"
            if result.returncode == 0 and usage is not None
            else "MODEL_CLIENT_FAILED_OR_USAGE_MISSING",
            "returncode": result.returncode,
            "requested_model": MODEL,
            "effort": EFFORT,
            "model_revision": None,
            "temperature": None,
            "sampling_policy": "CLIENT_PROVIDER_DEFAULT; no unsupported numeric temperature asserted",
            "usage": usage,
            "cost": None,
            "cost_reason": "not reported by this CLI",
            "raw_event_types": [e.get("type") for e in events],
        }
        (output / "outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")
        return outcome
