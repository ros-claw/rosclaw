"""Tests for the real-policy smoke workflow.

These tests do not require a real LeRobot runtime or network access.
"""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path

import pytest

from rosclaw.integrations.lerobot.action_adapter import adapt_action_to_proposal
from rosclaw.integrations.lerobot.policy_cache import (
    PolicyMaterializationError,
    get_policy_cache_dir,
    materialize_policy_path,
)
from rosclaw.integrations.lerobot.smoke_policy import (
    DEFAULT_SMOKE_POLICY,
    SmokePolicyOptions,
    run_smoke_policy_sync,
)
from rosclaw.integrations.lerobot.smoke_report import (
    SmokeReport,
    get_validation_status,
    read_latest_smoke_report,
    write_smoke_report,
)
from rosclaw.integrations.lerobot.worker_runner import LeRobotWorkerRunner


@pytest.fixture
def fake_lerobot_worker(monkeypatch, tmp_path: Path):
    """Patch ``LeRobotWorkerRunner`` to use a fake worker script."""
    script = tmp_path / "fake_worker_main.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "req_path = Path(sys.argv[sys.argv.index('--request-json') + 1])\n"
        "out_path = Path(sys.argv[sys.argv.index('--output-json') + 1])\n"
        "req = json.loads(req_path.read_text())\n"
        "op = req.get('op', 'inspect')\n"
        "response = {\n"
        "    'schema_version': 'rosclaw.lerobot.worker.v1',\n"
        "    'status': 'ok',\n"
        "    'op': op,\n"
        "    'policy_path': req.get('policy_path', ''),\n"
        "    'real_model_loaded': op in ('load_test', 'infer'),\n"
        "    'real_inference': op == 'infer',\n"
        "    'policy_metadata': {\n"
        "        'policy_type': 'act',\n"
        "        'config_found': True,\n"
        "        'input_features': {\n"
        "            'observation.images.top': {'type': 'VISUAL', 'shape': [3, 480, 640]},\n"
        "            'observation.state': {'type': 'STATE', 'shape': [14]}\n"
        "        },\n"
        "        'output_features': {\n"
        "            'action': {'type': 'ACTION', 'shape': [14]}\n"
        "        }\n"
        "    },\n"
        "    'timing': {'load_time_sec': 0.1} if op in ('load_test', 'infer') else {},\n"
        "    'runtime': {'python': sys.executable, 'python_version': '3.12.0'},\n"
        "}\n"
        "if op == 'infer':\n"
        "    response['action'] = {\n"
        "        'type': 'lerobot_action_chunk',\n"
        "        'values': [[0.0]*14 for _ in range(100)],\n"
        "        'shape': [100, 14],\n"
        "        'dtype': 'float32',\n"
        "    }\n"
        "out_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "out_path.write_text(json.dumps(response, indent=2))\n"
    )

    original_init = LeRobotWorkerRunner.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.worker_script = script
        self._resolve_runtime = lambda: (sys.executable, None)

    monkeypatch.setattr(LeRobotWorkerRunner, "__init__", patched_init)


@pytest.fixture
def local_policy_dir(tmp_path: Path) -> Path:
    """Create a minimal local policy directory that passes materialization."""
    policy_dir = tmp_path / "act_aloha"
    policy_dir.mkdir()
    (policy_dir / "config.json").write_text(
        json.dumps(
            {
                "policy_type": "act",
                "input_features": {
                    "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 640]},
                    "observation.state": {"type": "STATE", "shape": [14]},
                },
                "output_features": {
                    "action": {"type": "ACTION", "shape": [14]},
                },
            }
        ),
        encoding="utf-8",
    )
    (policy_dir / "model.safetensors").write_bytes(b"")
    return policy_dir


@pytest.fixture
def fake_runtime_check(monkeypatch):
    """Patch the runtime check so tests pass without a real LeRobot runtime."""
    from rosclaw.integrations.lerobot import smoke_policy

    def fake_check(options):
        from rosclaw.integrations.lerobot.smoke_policy import SmokeStageResult

        return SmokeStageResult(
            status="ok",
            data={
                "runtime": {
                    "mode": "current-env",
                    "python_executable": sys.executable,
                    "python_version": "3.12.0",
                    "lerobot_version": "0.6.1",
                    "torch_version": "2.11.0",
                    "cuda_available": False,
                }
            },
        )

    monkeypatch.setattr(smoke_policy, "_stage_runtime_check", fake_check)


def test_smoke_policy_rejects_uncached_hf_repo_without_allow_network(
    fake_runtime_check, monkeypatch, tmp_path: Path
):
    """A HF repo id without --allow-network must fail at materialization."""

    def fake_materialize(*args, **kwargs):
        raise PolicyMaterializationError(
            "network_disabled",
            "Policy is not cached locally and allow_network=false.",
        )

    monkeypatch.setattr(
        "rosclaw.integrations.lerobot.smoke_policy.materialize_policy_path",
        fake_materialize,
    )

    options = SmokePolicyOptions(
        policy_path=DEFAULT_SMOKE_POLICY,
        allow_network=False,
    )
    report = run_smoke_policy_sync(options)

    assert report.status == "error"
    assert report.stages.get("materialize", {}).get("status") == "error"
    assert report.error is not None
    assert report.error["code"] == "network_disabled"


def test_materialize_hf_repo_uses_complete_rosclaw_cache_without_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete ROSClaw policy cache should not require huggingface_hub."""
    repo_id = "lerobot/act_cached"
    cached = get_policy_cache_dir() / "lerobot_act_cached"
    cached.mkdir(parents=True)
    (cached / "config.json").write_text('{"type": "act"}', encoding="utf-8")
    (cached / "model.safetensors").write_bytes(b"weights")

    original_import = builtins.__import__

    def fail_hub_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("blocked in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_hub_import)

    result = materialize_policy_path(repo_id, allow_network=False)

    assert result.local_path == cached
    assert result.cache_hit is True
    assert result.network_used is False


def test_materialize_hf_repo_rejects_config_only_rosclaw_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config-only cache is not enough for load-test/infer smoke."""
    repo_id = "lerobot/config_only"
    cached = get_policy_cache_dir() / "lerobot_config_only"
    cached.mkdir(parents=True)
    (cached / "config.json").write_text('{"type": "act"}', encoding="utf-8")

    original_import = builtins.__import__

    def fail_hub_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("blocked in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_hub_import)

    with pytest.raises(PolicyMaterializationError) as exc:
        materialize_policy_path(repo_id, allow_network=False)

    assert exc.value.code == "network_disabled"


def test_smoke_policy_uses_local_policy_path(
    fake_runtime_check, fake_lerobot_worker, local_policy_dir: Path, tmp_path: Path
):
    """A local policy directory should complete all smoke stages."""
    options = SmokePolicyOptions(
        policy_path=str(local_policy_dir),
        device="cpu",
        observation_file=tmp_path / "obs.json",
    )
    obs_file = options.observation_file
    img = tmp_path / "top.png"
    img.write_bytes(b"")
    obs_file.write_text(
        json.dumps(
            {
                "observation": {
                    "state": [0.0] * 14,
                    "images": {"top": str(img)},
                }
            }
        ),
        encoding="utf-8",
    )

    report = run_smoke_policy_sync(options)

    assert report.status == "ok"
    assert report.stages["runtime_check"]["status"] == "ok"
    assert report.stages["materialize"]["status"] == "ok"
    assert report.stages["inspect"]["status"] == "ok"
    assert report.stages["load_test"]["status"] == "ok"
    assert report.stages["infer"]["status"] == "ok"
    assert report.policy["local_path"] == str(local_policy_dir.resolve())
    assert report.policy["repo_id"] is None
    assert report.features["input_features"]["observation.images.top"] == [3, 480, 640]
    assert report.features["input_features"]["observation.state"] == [14]
    assert report.features["output_features"]["action"] == [14]
    assert report.sample_observation["state_shape"] == [14]
    assert report.sample_observation["image_keys"] == ["top"]
    assert report.action_proposal is not None
    assert report.action_proposal["shape"] == [100, 14]
    assert report.action_proposal["type"] == "lerobot_action_chunk"
    assert report.action_proposal["not_executed"] is True
    assert report.action_proposal["requires_sandbox"] is True
    assert report.action_proposal["executable"] is False


def test_smoke_policy_skip_infer(fake_runtime_check, fake_lerobot_worker, local_policy_dir: Path):
    """--skip-infer should skip inference but still inspect/load-test."""
    options = SmokePolicyOptions(
        policy_path=str(local_policy_dir),
        skip_infer=True,
    )
    report = run_smoke_policy_sync(options)

    assert report.status == "ok"
    assert report.stages["infer"].get("status") == "skipped"
    assert report.action_proposal is None


def test_smoke_report_write_and_read(tmp_path: Path, monkeypatch):
    """Smoke reports should round-trip through the report directory."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))

    report = SmokeReport(
        status="ok",
        policy={"repo_id": "lerobot/act_test", "local_path": "/tmp/policy"},
        stages={"runtime_check": "ok", "infer": "ok"},
        action_proposal={"shape": [14], "values": [0.0] * 14},
    )
    path = write_smoke_report(report)

    assert path.exists()
    latest = read_latest_smoke_report()
    assert latest is not None
    assert latest.status == "ok"
    assert latest.policy["repo_id"] == "lerobot/act_test"

    validation = get_validation_status()
    assert validation["validated"] is True
    assert validation["last_policy"] == "lerobot/act_test"


def test_doctor_shows_not_validated_without_report(tmp_path: Path, monkeypatch):
    """Doctor validation should report 'not validated' when no report exists."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    validation = get_validation_status()
    assert validation["validated"] is False
    assert validation["last_policy"] is None


def test_doctor_shows_validated_with_success_report(tmp_path: Path, monkeypatch):
    """Doctor validation should report 'validated' after a successful smoke."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    report = SmokeReport(
        status="ok",
        policy={"repo_id": "lerobot/act_test"},
        action_proposal={"shape": [100, 14]},
    )
    write_smoke_report(report)
    validation = get_validation_status()
    assert validation["validated"] is True
    assert validation["action_shape"] == [100, 14]


def test_action_chunk_adapter_shape_100_14():
    """Action adapter must preserve a [100, 14] chunk shape."""
    from rosclaw.integrations.lerobot.worker_schema import WorkerAction

    values = [[float(j) for j in range(14)] for _ in range(100)]
    action = WorkerAction(
        type="lerobot_action_chunk",
        values=values,
        shape=[100, 14],
        dtype="float32",
    )
    proposal = adapt_action_to_proposal(action)
    assert proposal["shape"] == [100, 14]
    assert proposal["type"] == "lerobot_action_chunk"
    assert proposal["chunk_size"] == 100
    assert proposal["action_dim"] == 14
    assert proposal["executable"] is False
    assert proposal["requires_sandbox"] is True
    assert len(proposal["values"]) == 100
    assert len(proposal["values"][0]) == 14
