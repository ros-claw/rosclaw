"""Provider-level P1 contract tests using a fake LeRobot worker."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rosclaw.core.async_utils import run_sync
from rosclaw.integrations.lerobot.provider import LeRobotPolicyProvider
from rosclaw.integrations.lerobot.worker_runner import LeRobotWorkerRunner
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.request import ProviderRequest


@pytest.fixture
def sample_manifest() -> Path:
    return Path(__file__).parent.parent.parent / "examples" / "lerobot" / "sample_policy_manifest.yaml"


@pytest.fixture
def provider(sample_manifest: Path) -> LeRobotPolicyProvider:
    manifest = ProviderManifest.from_yaml(sample_manifest)
    return LeRobotPolicyProvider(manifest)


def _patch_worker(monkeypatch, runner: LeRobotWorkerRunner, script: Path) -> None:
    monkeypatch.setattr(runner, "worker_script", script)
    monkeypatch.setattr(runner, "_resolve_runtime", lambda: (sys.executable, None))


def _build_request(
    capability: str,
    inputs: dict,
    request_id: str = "test-req",
) -> ProviderRequest:
    return ProviderRequest(
        request_id=request_id,
        capability=capability,
        inputs=inputs,
        context={"manifest": "sample.yaml"},
    )


def _patch_runner_init(monkeypatch, fake_worker_script: Path):
    original_init = LeRobotWorkerRunner.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _patch_worker(monkeypatch, self, fake_worker_script)

    monkeypatch.setattr(LeRobotWorkerRunner, "__init__", patched_init)


def test_provider_inspect_success(
    provider: LeRobotPolicyProvider,
    monkeypatch,
    fake_worker_script: Path,
) -> None:
    _patch_runner_init(monkeypatch, fake_worker_script)

    request = _build_request(
        "lerobot.policy.inspect",
        {"policy.path": "/tmp/policy"},
    )
    response = run_sync(provider.infer(request))

    assert response.status == "ok"
    assert response.result["mode"] == "policy_inspect"
    assert response.result["real_inference"] is False
    assert response.result["not_executed"] is True
    assert response.result["requires_sandbox"] is True
    assert response.result["action_proposal"] is None


def test_provider_load_test_success(
    provider: LeRobotPolicyProvider,
    monkeypatch,
    fake_worker_script: Path,
) -> None:
    _patch_runner_init(monkeypatch, fake_worker_script)

    request = _build_request(
        "lerobot.policy.load_test",
        {"policy.path": "/tmp/policy", "device": "cpu"},
    )
    response = run_sync(provider.infer(request))

    assert response.status == "ok"
    assert response.result["mode"] == "policy_load_test"
    assert response.result["real_model_loaded"] is True
    assert response.result["real_inference"] is False
    assert response.result["action_proposal"] is None


def test_provider_infer_marks_not_executed(
    provider: LeRobotPolicyProvider,
    monkeypatch,
    fake_worker_script: Path,
) -> None:
    _patch_runner_init(monkeypatch, fake_worker_script)

    request = _build_request(
        "lerobot.policy.infer",
        {
            "policy.path": "/tmp/policy",
            "device": "cpu",
            "observation": {"observation.state": [0.0] * 6},
        },
    )
    response = run_sync(provider.infer(request))

    assert response.status == "ok"
    assert response.result["mode"] == "real_policy_infer"
    assert response.result["real_inference"] is True
    assert response.result["not_executed"] is True
    assert response.result["requires_sandbox"] is True
    proposal = response.result["action_proposal"]
    assert proposal is not None
    assert proposal["executable"] is False
    assert proposal["requires_sandbox"] is True
    assert proposal["values"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def test_provider_infer_rejects_execute(provider: LeRobotPolicyProvider) -> None:
    request = _build_request(
        "lerobot.policy.infer",
        {"policy.path": "/tmp/policy", "execute": True},
    )
    response = run_sync(provider.infer(request))

    assert response.status == "blocked"
    assert response.result["not_executed"] is True
    assert response.result["requires_sandbox"] is True
    assert response.result["action_proposal"] is None


def test_provider_infer_dry_run_returns_sample(provider: LeRobotPolicyProvider) -> None:
    request = _build_request(
        "lerobot.policy.infer",
        {"policy.path": "/tmp/policy", "dry_run": True},
    )
    response = run_sync(provider.infer(request))

    assert response.status == "ok"
    assert response.result["mode"] == "dry_run"
    assert response.result["real_inference"] is False
    assert response.result["action_proposal"]["executable"] is False
    assert response.result["action_proposal"]["values"] == [0.0] * 7


def test_provider_infer_requires_policy_path(provider: LeRobotPolicyProvider) -> None:
    request = _build_request("lerobot.policy.infer", {})
    response = run_sync(provider.infer(request))

    assert response.status == "failed"
    assert response.result["error_code"] == "policy_config_not_found"
