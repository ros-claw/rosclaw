"""One fixed-image sensor-only observation container, never a robot operator.

The SDK fixture creates only frozen sensor test endpoints. Host code opens no
ROS transport, Runtime or driver; model arguments cannot select paths/commands.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from benchmarks.ros_expert_ablation.rsk_episode_runner import Closed, OriginalRef, _read


class SensorObservationConfigV1(Closed):
    schema_version: Literal["rosclaw.rsk_sensor_observation_config.v1"] = (
        "rosclaw.rsk_sensor_observation_config.v1"
    )
    namespace: str = Field(pattern=r"^/rsk_fixture_[a-z0-9]{8,32}$")
    topic: str = Field(pattern=r"^/rsk_fixture_[a-z0-9]{8,32}/sensor/[a-z0-9_]{1,32}$")
    offered_reliability: Literal["RELIABLE", "BEST_EFFORT"]
    requested_reliability: Literal["RELIABLE", "BEST_EFFORT"]
    producer_schedule: Literal["CONTINUOUS", "STOP_BEFORE_WINDOW"]

    @model_validator(mode="after")
    def within_namespace(self):
        if not self.topic.startswith(self.namespace + "/sensor/"):
            raise ValueError("fixed sensor topic must belong to the fixture namespace")
        return self


class SensorOnlyDiagnosticOperator:
    def __init__(
        self, policy, *, worktree, evidence_root, execution_config: OriginalRef, output_parent
    ):
        from benchmarks.ros_expert_ablation.rsk_diagnostic_provider import DiagnosticProbePolicyV1

        self.policy = DiagnosticProbePolicyV1.model_validate(policy)
        self.worktree = Path(worktree).resolve(strict=True)
        self.root = Path(evidence_root).resolve(strict=True)
        self.execution_config = OriginalRef.model_validate(execution_config)
        self.output_parent = Path(output_parent).resolve(strict=True)
        self.output_parent.relative_to(self.root)
        raw = _read(self.root, self.execution_config, limit=64_000)
        SensorObservationConfigV1.model_validate_json(raw)
        if self.execution_config.sha256 != self.policy.execution_config_sha256:
            raise ValueError("operator config differs from frozen probe policy")
        self.calls = 0

    def collect(self, plan, selected, *, remaining_sec):
        from benchmarks.ros_expert_ablation.rsk_diagnostic_provider import (
            DiagnosticProbeOriginalsV1,
        )

        if Path(__file__).resolve().parents[2] != self.worktree:
            raise ValueError(
                "SDK operator implementation differs from the declared source worktree"
            )
        if self.calls or selected.request.operation not in self.policy.operations:
            raise ValueError("single supported sensor-only diagnostic capture required")
        if not self.policy.minimum_wall_sec <= remaining_sec <= 300:
            raise ValueError("sufficient host budget required before SDK creation")
        raw = _read(self.root, self.execution_config, limit=64_000)
        SensorObservationConfigV1.model_validate_json(raw)
        if self.execution_config.sha256 != self.policy.execution_config_sha256:
            raise ValueError("original SDK execution config changed")

        def git(*args):
            return subprocess.check_output(["git", *args], cwd=self.worktree, text=True).strip()

        if git("rev-parse", "HEAD") != self.policy.source_revision or git("status", "--porcelain"):
            raise ValueError("exact clean source required before SDK creation")
        self.calls += 1
        begun = datetime.now(UTC)
        started = time.monotonic()
        run = uuid.uuid4().hex
        out = self.output_parent / ("RSK-diagnostic-" + run)
        out.mkdir(exist_ok=False)
        execution = out / "execution"
        execution.mkdir()
        config = (self.root / self.execution_config.path).resolve(strict=True)
        argv = [
            "docker",
            "create",
            "--pull",
            "never",
            "--name",
            "reh-rsk-diagnostic-" + run[:16],
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--memory",
            "1g",
            "--cpus",
            "2",
            "--pids-limit",
            "128",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=128m",
            "--env",
            "HOME=/tmp",
            "--env",
            "ROS_LOG_DIR=/tmp/ros-log",
            "--env",
            "ROS_DOMAIN_ID=179",
            "--env",
            "ROS_LOCALHOST_ONLY=1",
            "--env",
            "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp",
            "--env",
            "ROSCLAW_RSK_FIXTURE=REGISTERED_SENSOR_ONLY_READONLY_V1",
            "--env",
            "RSK_RUN_ID=" + run,
            "--env",
            "RSK_SOURCE_COMMIT=" + self.policy.source_revision,
            "--mount",
            f"type=bind,src={self.worktree},dst=/repo,readonly",
            "--mount",
            f"type=bind,src={config},dst=/fixture-config.json,readonly",
            "--mount",
            f"type=bind,src={execution},dst=/evidence",
            "--entrypoint",
            "/bin/bash",
            self.policy.sdk_image,
            "-c",
            "source /opt/ros/jazzy/setup.bash && python3 /repo/benchmarks/ros_expert_ablation/rsk_native_diagnostic_fixture.py",
        ]
        self._write(out / "operator-argv.json", argv)
        cid = None
        try:
            cid = subprocess.check_output(argv, text=True, timeout=min(15, remaining_sec)).strip()
            (out / "container-id.txt").write_text(cid + "\n")
            created = json.loads(
                subprocess.check_output(["docker", "inspect", cid], text=True, timeout=15)
            )[0]
            self._write(out / "container-before.json", created)
            with (out / "SDK-original.log").open("xb") as log:
                process = subprocess.run(
                    ["docker", "start", "-a", cid],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=max(0.001, remaining_sec - (time.monotonic() - started) - 2),
                )
            if process.returncode:
                raise ValueError("original sensor-only SDK fixture failed; preserve logs, no retry")
            stopped = json.loads(
                subprocess.check_output(["docker", "inspect", cid], text=True, timeout=15)
            )[0]
            self._write(out / "container-after.json", stopped)
            if stopped["State"]["Running"] or stopped["State"]["ExitCode"] != 0:
                raise ValueError("sensor-only SDK did not stop successfully")
            if git("rev-parse", "HEAD") != self.policy.source_revision or git(
                "status", "--porcelain"
            ):
                raise ValueError("frozen source changed during original SDK capture")
            if time.monotonic() - started > remaining_sec:
                raise ValueError("sensor-only capture exceeded the original host budget")

            def ref(path):
                return OriginalRef(
                    path=str(path.relative_to(self.root)),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )

            manifest = DiagnosticProbeOriginalsV1(
                source_revision=self.policy.source_revision,
                original_run_id=self.policy.original_run_id,
                probe_run_id=run,
                requested_operation=selected.request.operation,
                started_at=begun.isoformat(),
                finished_at=datetime.now(UTC).isoformat(),
                execution_config_sha256=self.policy.execution_config_sha256,
                native_before=ref(execution / "native-before.json"),
                native_after=ref(execution / "native-after.json"),
                owner_observations=ref(execution / "owner-observations.json"),
                container_before=ref(out / "container-before.json"),
                container_after=ref(out / "container-after.json"),
            )
            path = out / "original-probe-manifest.json"
            self._write(path, manifest.model_dump(mode="json"))
            return ref(path).model_dump(mode="json")
        except Exception as error:
            self._write(
                out / "original-failure.json",
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "source": self.policy.source_revision,
                    "run_id": run,
                    "automatic_retry": False,
                },
            )
            raise
        finally:
            if cid is not None:
                actual = json.loads(
                    subprocess.check_output(["docker", "inspect", cid], text=True, timeout=15)
                )[0]
                if (
                    actual["Id"] != cid
                    or actual["Image"] != self.policy.sdk_image
                    or actual["Name"] != "/reh-rsk-diagnostic-" + run[:16]
                ):
                    raise ValueError("owned SDK identity changed; no unrelated container cleanup")
                if actual["State"]["Running"]:
                    subprocess.run(
                        ["docker", "stop", "-t", "3", cid],
                        check=True,
                        capture_output=True,
                        timeout=15,
                    )
                    actual = json.loads(
                        subprocess.check_output(["docker", "inspect", cid], text=True, timeout=15)
                    )[0]
                self._write(out / "container-finally-stopped.json", actual)

    @staticmethod
    def _write(path, value):
        with path.open("x") as stream:
            json.dump(value, stream, indent=2)
