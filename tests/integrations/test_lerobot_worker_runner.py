"""Tests for the LeRobot subprocess worker runner."""

from __future__ import annotations

import sys
from pathlib import Path

from rosclaw.integrations.lerobot.worker_runner import LeRobotWorkerRunner
from rosclaw.integrations.lerobot.worker_schema import WorkerRequest


def _patch_runner(monkeypatch, runner: LeRobotWorkerRunner, script: Path) -> None:
    """Force the runner to use the current Python and the provided fake script."""
    monkeypatch.setattr(runner, "worker_script", script)
    monkeypatch.setattr(
        runner,
        "_resolve_runtime",
        lambda: (sys.executable, None),
    )


def test_worker_runner_builds_request(tmp_path: Path, monkeypatch, fake_worker_script: Path) -> None:
    runner = LeRobotWorkerRunner(timeout_sec=10)
    _patch_runner(monkeypatch, runner, fake_worker_script)

    request = WorkerRequest(op="inspect", policy_path="/tmp/policy")
    response = runner.run(request)

    assert response.ok
    assert response.op == "inspect"
    assert response.policy_path == "/tmp/policy"
    assert response.policy_metadata["policy_type"] == "act"


def test_worker_runner_runs_infer(tmp_path: Path, monkeypatch, fake_worker_script: Path) -> None:
    runner = LeRobotWorkerRunner(timeout_sec=10)
    _patch_runner(monkeypatch, runner, fake_worker_script)

    request = WorkerRequest(
        op="infer",
        policy_path="/tmp/policy",
        observation={"observation.state": [0.0] * 6},
    )
    response = runner.run(request)

    assert response.ok
    assert response.real_inference is True
    assert response.real_model_loaded is True
    assert response.action is not None
    assert response.action.values == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def test_worker_runner_handles_missing_runtime(monkeypatch, tmp_path: Path) -> None:
    runner = LeRobotWorkerRunner(timeout_sec=10)
    # Simulate no configured runtime and no in-process LeRobot.
    monkeypatch.setattr(
        "rosclaw.integrations.lerobot.worker_runner.get_configured_lerobot_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        "rosclaw.integrations.lerobot.worker_runner.inspect_lerobot_runtime",
        lambda *_args, **_kwargs: _FakeRuntime(state="error"),
    )

    request = WorkerRequest(op="inspect", policy_path="/tmp/policy")
    response = runner.run(request)

    assert not response.ok
    assert response.error_code() == "runtime_not_configured"


def test_worker_runner_handles_invalid_json(
    tmp_path: Path, monkeypatch, fake_worker_script_invalid_json: Path
) -> None:
    runner = LeRobotWorkerRunner(timeout_sec=10)
    _patch_runner(monkeypatch, runner, fake_worker_script_invalid_json)

    request = WorkerRequest(op="inspect", policy_path="/tmp/policy")
    response = runner.run(request)

    assert not response.ok
    assert response.error_code() == "worker_invalid_json"


def test_worker_runner_handles_process_failed(
    tmp_path: Path, monkeypatch, fake_worker_script_nonzero: Path
) -> None:
    runner = LeRobotWorkerRunner(timeout_sec=10)
    _patch_runner(monkeypatch, runner, fake_worker_script_nonzero)

    request = WorkerRequest(op="inspect", policy_path="/tmp/policy")
    response = runner.run(request)

    assert not response.ok
    assert response.error_code() == "worker_process_failed"


def test_worker_runner_handles_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    """A slow worker should trigger a timeout response."""
    slow_script = tmp_path / "slow_worker.py"
    slow_script.write_text(
        "#!/usr/bin/env python3\n"
        "import time\n"
        "time.sleep(10)\n"
    )
    slow_script.chmod(0o755)

    runner = LeRobotWorkerRunner(timeout_sec=1)
    _patch_runner(monkeypatch, runner, slow_script)

    request = WorkerRequest(op="inspect", policy_path="/tmp/policy")
    response = runner.run(request)

    assert not response.ok
    assert response.error_code() == "worker_timeout"


class _FakeRuntime:
    def __init__(self, state: str) -> None:
        self.state = state
        self.lerobot_version = None
