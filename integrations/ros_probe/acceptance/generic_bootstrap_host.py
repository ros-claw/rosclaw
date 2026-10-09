"""Own one isolated inactive SIM bootstrap container, never a robot task.

This operator fixture workflow preserves full source capture and bounded cleanup.
It grants no Body admission; all controllers remain inactive and Nav2 autostart
is disabled by the sealed launch source.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

from fixture_network import validate_ros_domain
from generic_bootstrap_launch import bootstrap_launch_plan
from generic_bootstrap_runtime import validate_deadline_seconds
from generic_container_ownership import (
    KIND,
    KIND_LABEL,
    OWNER_LABEL,
    inspect_owned_container,
    stop_owned_container,
    wait_owned_container,
)
from generic_stack_source import read_prepared_generic_stack

IMAGE_ID = "sha256:d16320799584a60035548fb298243bdd8b0d2cb5760675a9cb46b608ea211e56"


def bootstrap_container_command(source, workspace, output, declaration, *, owner, domain, seconds):
    """Constant entrypoint; user data is passed only as distinct argv entries."""
    source, workspace, output, declaration = (
        Path(p).resolve() for p in (source, workspace, output, declaration)
    )
    if any(
        tree.is_relative_to(output) or output.is_relative_to(tree) for tree in (source, workspace)
    ) or declaration.is_relative_to(output):
        raise ValueError("writable output must not alias any immutable source mount")
    return [
        "docker",
        "create",
        "--pull",
        "never",
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        "1000:1000",
        "--name",
        "rosclaw-generic-bootstrap-" + owner,
        "--label",
        OWNER_LABEL + "=" + owner,
        "--label",
        KIND_LABEL + "=" + KIND,
        "--env",
        "ROS_DOMAIN_ID=" + str(domain),
        "--env",
        "GZ_PARTITION=rosclaw_generic_" + owner,
        "--env",
        "PYTHONPATH=/workspace/src:/workspace/integrations/ros_probe/acceptance",
        "--env",
        "TMPDIR=/runtime",
        "--env",
        "ROS_LOG_DIR=/runtime/ros-logs",
        "--mount",
        "type=bind,src=" + str(source) + ",dst=/workspace,readonly",
        "--mount",
        "type=bind,src=" + str(workspace) + ",dst=/evidence,readonly",
        "--mount",
        "type=bind,src=" + str(output) + ",dst=/runtime",
        "--mount",
        "type=bind,src=" + str(declaration) + ",dst=/bootstrap-declaration.json,readonly",
        "--entrypoint",
        "bash",
        IMAGE_ID,
        "-c",
        'source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash && source /opt/reh-control-metadata/setup.bash && exec python3 "$@"',
        "owned-generic-bootstrap",
        "/workspace/integrations/ros_probe/acceptance/generic_bootstrap_runtime.py",
        "--declaration",
        "/bootstrap-declaration.json",
        "--output",
        "/runtime/owned-launch",
        "--seconds",
        str(seconds),
    ]


def run_owned_bootstrap(source, workspace, declaration, output, *, seconds, domain):
    seconds = validate_deadline_seconds(seconds)
    validate_ros_domain(domain)
    deadline = time.monotonic() + seconds
    paths = [Path(p) for p in (source, workspace, declaration, output)]
    if any(p.is_symlink() or not p.is_absolute() or "," in str(p) for p in paths):
        raise ValueError("absolute non-symlink unambiguous bind paths required")
    source, workspace, declaration, output = [p.resolve() for p in paths]
    if (
        Path(__file__).resolve()
        != source / "integrations/ros_probe/acceptance/generic_bootstrap_host.py"
    ):
        raise ValueError("host launcher must belong to the exact mounted source tree")
    if output.is_relative_to(source) or output.is_relative_to(workspace):
        raise ValueError("fresh output must be outside immutable source trees")
    raw = declaration.read_bytes()
    if not 0 < len(raw) <= 65536:
        raise ValueError("bounded original bootstrap declaration required")
    frozen_plan = bootstrap_launch_plan(workspace, json.loads(raw))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=source):
        raise ValueError("exact clean source commit required before container creation")
    names = subprocess.check_output(
        [
            "git",
            "ls-files",
            "-z",
            "src",
            "integrations/ros_probe/acceptance",
            "integrations/ros_probe/ros2",
        ],
        cwd=source,
    ).split(b"\0")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    captured = {}
    for name in names:
        if not name:
            continue
        relative = Path(os.fsdecode(name))
        original = source / relative
        if original.is_symlink() or not original.is_file():
            raise ValueError("regular tracked runtime source required")
        data = original.read_bytes()
        captured[str(relative)] = hashlib.sha256(data).hexdigest()
        copy = output / "before-call-source" / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(data)
    # Writable runtime output must not expose an alias of any readonly source.
    frozen_source = output / "before-call-source"
    frozen_workspace = output / "sealed-workspace"
    frozen_workspace.mkdir()
    sealed = read_prepared_generic_stack(workspace)
    for name, data in sealed["captured_files"].items():
        target = frozen_workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    manifest = (workspace / "generic-stack-source-manifest.json").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != sealed["manifest_sha256"]:
        raise ValueError("workspace manifest changed during source capture")
    (frozen_workspace / "generic-stack-source-manifest.json").write_bytes(manifest)
    frozen_declaration = output / "bootstrap-declaration.json"
    frozen_declaration.write_bytes(raw)
    runtime_output = output / "runtime-files"
    runtime_output.mkdir(mode=0o700)
    (output / "source-freeze.json").write_text(
        json.dumps(
            {
                "commit": commit,
                "source_hashes": captured,
                "plan": frozen_plan,
                "declaration_sha256": hashlib.sha256(raw).hexdigest(),
                "image_id": IMAGE_ID,
            },
            indent=2,
        )
        + "\n"
    )
    owner, container_id = uuid.uuid4().hex, None
    outcome = "PREFLIGHT_FAILED"
    cleanup_verified, log_capture = False, "NOT_ATTEMPTED"

    def check_source():
        if (
            declaration.read_bytes() != raw
            or bootstrap_launch_plan(workspace, json.loads(raw)) != frozen_plan
        ):
            raise ValueError("frozen declaration or workspace changed")
        if any(
            hashlib.sha256((source / name).read_bytes()).hexdigest() != sha
            for name, sha in captured.items()
        ):
            raise ValueError("captured runtime source changed")
        if (
            frozen_declaration.read_bytes() != raw
            or bootstrap_launch_plan(frozen_workspace, json.loads(raw)) != frozen_plan
            or any(
                hashlib.sha256((frozen_source / name).read_bytes()).hexdigest() != sha
                for name, sha in captured.items()
            )
        ):
            raise ValueError("actual readonly mounted source capture changed")

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    prior = signal.signal(signal.SIGTERM, interrupted)
    try:
        check_source()
        if time.monotonic() >= deadline:
            raise TimeoutError("immutable host deadline exhausted during source capture")
        argv = bootstrap_container_command(
            frozen_source,
            frozen_workspace,
            runtime_output,
            frozen_declaration,
            owner=owner,
            domain=domain,
            seconds=seconds,
        )
        (output / "original-create-argv.json").write_text(json.dumps(argv, indent=2) + "\n")
        created = subprocess.run(
            argv, capture_output=True, timeout=min(10, deadline - time.monotonic()), check=True
        )
        container_id = created.stdout.decode().strip()
        inspect_owned_container(container_id, owner)
        check_source()
        if time.monotonic() >= deadline:
            raise TimeoutError("immutable host deadline exhausted before container start")
        subprocess.run(
            ["docker", "start", container_id],
            capture_output=True,
            check=True,
            timeout=min(5, deadline - time.monotonic()),
        )
        outcome = wait_owned_container(
            container_id, owner, deadline=deadline, check_source=check_source
        )
    except BaseException:
        outcome = "SOURCE_OR_CONTAINER_FAILURE"
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            if container_id is not None:
                cleanup = stop_owned_container(
                    container_id, owner, output / "container-cleanup.json"
                )
                cleanup_verified = cleanup["container_stopped"]
                with (
                    (output / "container.original-stdout").open("xb") as stdout,
                    (output / "container.original-stderr").open("xb") as stderr,
                ):
                    try:
                        reply = subprocess.run(
                            ["docker", "logs", container_id],
                            stdout=stdout,
                            stderr=stderr,
                            timeout=5,
                        )
                        log_capture = "CAPTURED" if reply.returncode == 0 else "DOCKER_LOGS_FAILED"
                    except subprocess.TimeoutExpired:
                        log_capture = "DOCKER_LOGS_TIMEOUT"
        finally:
            signal.signal(signal.SIGTERM, prior)
            (output / "host-result.json").write_text(
                json.dumps(
                    {
                        "source_commit": commit,
                        "container_id": container_id,
                        "outcome": outcome,
                        "cleanup_verified": cleanup_verified,
                        "log_capture": log_capture,
                        "controller_activation": False,
                        "live_body_admitted": False,
                        "authorization": False,
                        "physical_acceptance": "NOT_EVALUATED",
                    },
                    indent=2,
                )
                + "\n"
            )
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "workspace", "declaration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--domain", type=int, required=True)
    args = parser.parse_args()
    run_owned_bootstrap(
        args.source,
        args.workspace,
        args.declaration,
        args.output,
        seconds=args.seconds,
        domain=args.domain,
    )


if __name__ == "__main__":
    main()
