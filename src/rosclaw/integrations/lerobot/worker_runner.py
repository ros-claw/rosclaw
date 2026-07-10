"""ROSClaw-side runner for the LeRobot subprocess worker.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It writes a JSON request, spawns the LeRobot runtime Python to run
``worker_main.py``, reads the JSON response, and translates errors into
structured ``LeRobotWorkerErrorCode`` values.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from rosclaw.integrations.lerobot.config import get_configured_lerobot_runtime
from rosclaw.integrations.lerobot.runtime import inspect_lerobot_runtime
from rosclaw.integrations.lerobot.schemas import LeRobotWorkerErrorCode
from rosclaw.integrations.lerobot.worker_schema import WorkerError, WorkerRequest, WorkerResponse


class LeRobotWorkerRunner:
    """One-shot subprocess worker runner.

    The runner resolves the configured LeRobot runtime (or falls back to the
    current interpreter if LeRobot is importable in-process), writes a temp
    request JSON, runs ``worker_main.py``, and returns a ``WorkerResponse``.
    """

    def __init__(self, timeout_sec: int = 120, *, debug: bool | None = None):
        self.timeout_sec = timeout_sec
        self.debug = debug if debug is not None else bool(
            os.environ.get("ROSCLAW_LEROBOT_DEBUG_WORKER", "")
        )
        self._temp_dir: Path | None = None
        self.worker_script = Path(__file__).with_name("worker_main.py")

    def _resolve_runtime(self) -> tuple[str, str | None]:
        """Return the (python_executable, hf_endpoint) to use.

        1. Use configured LeRobot runtime if present and subprocess_available.
        2. Else fall back to the current interpreter if LeRobot is importable.
        3. Else raise ``RuntimeNotConfiguredError``.
        """
        configured = get_configured_lerobot_runtime()
        if configured and configured.get("subprocess_available"):
            python = configured.get("python_executable")
            if python and Path(python).exists():
                return str(python), configured.get("hf_endpoint")

        # Try current interpreter.
        import sys

        info = inspect_lerobot_runtime(sys.executable)
        if info.state in ("ready", "degraded") and info.lerobot_version is not None:
            return str(info.python_executable), None

        raise RuntimeNotConfiguredError(
            "No LeRobot runtime configured and current interpreter cannot import LeRobot."
        )

    def run(self, request: WorkerRequest) -> WorkerResponse:
        """Execute the worker for ``request`` and return the parsed response."""
        try:
            python_executable, hf_endpoint = self._resolve_runtime()
        except RuntimeNotConfiguredError as exc:
            return _runtime_error_response(request, LeRobotWorkerErrorCode.RUNTIME_NOT_CONFIGURED, str(exc))

        if not self.worker_script.exists():
            return _runtime_error_response(
                request,
                LeRobotWorkerErrorCode.WORKER_SCRIPT_MISSING,
                f"Worker script not found: {self.worker_script}",
            )

        self._temp_dir = Path(tempfile.mkdtemp(prefix="rosclaw_lerobot_worker_"))
        request_path = self._temp_dir / "request.json"
        response_path = self._temp_dir / "response.json"

        try:
            request_path.write_text(json.dumps(request.to_dict(), indent=2), encoding="utf-8")
            env = self._build_env(request, hf_endpoint)
            cmd = [
                python_executable,
                str(self.worker_script),
                "--request-json",
                str(request_path),
                "--output-json",
                str(response_path),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                env=env,
                check=False,
            )

            if result.returncode != 0:
                details = result.stderr.strip() or result.stdout.strip()
                return _runtime_error_response(
                    request,
                    LeRobotWorkerErrorCode.WORKER_PROCESS_FAILED,
                    f"Worker process exited with code {result.returncode}.",
                    details,
                )

            if not response_path.exists():
                return _runtime_error_response(
                    request,
                    LeRobotWorkerErrorCode.WORKER_INVALID_JSON,
                    "Worker did not write a response JSON file.",
                    result.stderr.strip(),
                )

            try:
                raw = json.loads(response_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                return _runtime_error_response(
                    request,
                    LeRobotWorkerErrorCode.WORKER_INVALID_JSON,
                    f"Worker response is not valid JSON: {exc}",
                    response_path.read_text(encoding="utf-8")[:2000],
                )

            if not isinstance(raw, dict):
                return _runtime_error_response(
                    request,
                    LeRobotWorkerErrorCode.WORKER_INVALID_JSON,
                    "Worker response is not a JSON object.",
                    str(raw)[:2000],
                )

            return WorkerResponse.from_dict(raw)
        except subprocess.TimeoutExpired:
            return _runtime_error_response(
                request,
                LeRobotWorkerErrorCode.WORKER_TIMEOUT,
                f"Worker timed out after {self.timeout_sec}s.",
            )
        except Exception as exc:  # noqa: BLE001
            return _runtime_error_response(
                request,
                LeRobotWorkerErrorCode.WORKER_PROCESS_FAILED,
                f"Failed to run worker: {exc}",
            )
        finally:
            self._cleanup()

    def _build_env(self, request: WorkerRequest, hf_endpoint: str | None) -> dict[str, str]:
        env = os.environ.copy()
        if not request.allow_network:
            env["HF_HUB_OFFLINE"] = "1"
        if hf_endpoint:
            env["HF_ENDPOINT"] = hf_endpoint
        if request.device.startswith("cuda"):
            env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        # Keep PYTHONPATH minimal to avoid ROS pytest plugins leaking in.
        env.pop("PYTHONPATH", None)
        return env

    def _cleanup(self) -> None:
        if self._temp_dir and not self.debug:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None


class RuntimeNotConfiguredError(Exception):
    """Raised when no LeRobot runtime can be resolved."""


def _runtime_error_response(
    request: WorkerRequest,
    code: LeRobotWorkerErrorCode,
    message: str,
    details: str = "",
) -> WorkerResponse:
    return WorkerResponse(
        status="error",
        op=request.op,
        policy_path=request.policy_path,
        error=WorkerError(
            code=code.value,
            message=message,
            details=details,
        ),
    )


def run_worker_op(
    op: str,
    policy_path: str,
    *,
    device: str = "cpu",
    allow_network: bool = False,
    timeout_sec: int = 120,
    observation: dict[str, Any] | None = None,
) -> WorkerResponse:
    """Convenience helper to build a request and run it in one call."""
    request = WorkerRequest(
        op=op,  # type: ignore[arg-type]
        policy_path=policy_path,
        device=device,
        allow_network=allow_network,
        timeout_sec=timeout_sec,
        observation=observation or {},
    )
    runner = LeRobotWorkerRunner(timeout_sec=timeout_sec)
    return runner.run(request)
