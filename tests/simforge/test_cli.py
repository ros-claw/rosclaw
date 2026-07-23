from __future__ import annotations

import importlib.util
import json
import stat
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from rosclaw.simforge.attestation import (
    create_simforge_signing_key_pair,
    sign_scale_curve,
)
from rosclaw.simforge.cli import _verify_scale_curve, dispatch_simforge_argv

_CANDIDATE_HASH = "sha256:" + "a" * 64


def test_main_cli_dispatches_simforge(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rosclaw.cli import main

    monkeypatch.setattr(
        sys,
        "argv",
        ["rosclaw", "simforge", "suite", "validate", "--json"],
    )

    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is True
    assert result["failures"] == []


def _scale_curve() -> dict[str, object]:
    scales = []
    evaluated_worlds = 0
    for count in (1, 2, 4):
        shards = []
        for gpu in range(count):
            worlds = 256
            shards.append(
                {
                    "schema_version": "rosclaw.mjwarp_shard.v1",
                    "backend": "mujoco_warp",
                    "physical_gpu": str(gpu),
                    "visible_devices": str(gpu),
                    "worlds": worlds,
                    "steps": 100,
                    "world_steps": worlds * 100,
                    "pose": "mixed",
                    "world_offset": gpu * worlds,
                    "wall_time_sec": 1.0,
                    "compile_time_sec": 1.0,
                    "warmup_time_sec": 1.0,
                    "world_steps_per_sec": 100.0,
                    "cpu_baseline_time_sec": 1.0,
                    "device": "cuda:0",
                    "device_name": "test-gpu",
                    "model_hash": "sha256:" + "b" * 64,
                    "qpos_checksum": "sha256:" + "c" * 64,
                    "gpu_memory_used_bytes": 1024,
                    "randomization": {
                        "parameter_hash": "sha256:" + "d" * 64,
                        "joint_control_offset_rad": [[0.0] * 6 for _ in range(worlds)],
                    },
                    "finite_state": True,
                    "expected_collision_label": True,
                    "scenario_label_valid": True,
                    "candidate_threshold": 0.5,
                    "risk_values": [0.8] * worlds,
                    "collision_worlds": list(range(worlds)),
                    "cpu_collision_worlds": list(range(worlds)),
                    "scenario_collision_worlds": list(range(worlds)),
                    "collision_world_count": worlds,
                    "differential": {
                        "baseline_backend": "mujoco_cpu",
                        "comparison_backend": "mujoco_warp",
                        "critical_label": "collision",
                        "critical_disagreement_count": 0,
                    },
                    "shield_metrics": {"candidate_unsafe_allow_count": 0},
                }
            )
        worlds = count * 256
        evaluated_worlds += worlds
        scales.append(
            {
                "gpu_count": count,
                "requested_gpus": [str(gpu) for gpu in range(count)],
                "complete": True,
                "failures": [],
                "worlds": worlds,
                "steps": 100,
                "world_steps": worlds * 100,
                "process_wall_time_sec": 1.0,
                "aggregate_world_steps_per_sec": count * 100.0,
                "speedup_vs_one_gpu": float(count),
                "shards": shards,
            }
        )
    return {
        "schema_version": "rosclaw.simforge.scale_curve.v1",
        "task_id": "shield_reach_v1",
        "minimum_worlds_required": 1000,
        "minimum_speedup_required": 2.5,
        "complete": True,
        "target_met": True,
        "fault_injected": False,
        "fault_type": None,
        "scales": scales,
        "differential": {
            "baseline_backend": "mujoco_cpu",
            "comparison_backend": "mujoco_warp",
            "critical_label": "collision",
            "critical_disagreements": 0,
            "evaluated_worlds": evaluated_worlds,
        },
    }


def _signing_material(tmp_path: Path) -> tuple[Path, Path]:
    private_key = tmp_path / "simforge-private.pem"
    public_key = tmp_path / "simforge-public.pem"
    create_simforge_signing_key_pair(
        private_key_path=private_key,
        public_key_path=public_key,
        source_checkout=Path(__file__).resolve().parents[2],
    )
    return private_key, public_key


def _sign(value: dict[str, object], private_key: Path) -> None:
    value.pop("attestation", None)
    value["attestation"] = sign_scale_curve(value, private_key_path=private_key)


def _scale_runner_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "simforge" / "run_scale_curve.py"
    spec = importlib.util.spec_from_file_location("rosclaw_run_scale_curve", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_suite_validate_cli_checks_all_core_tasks(capsys: pytest.CaptureFixture[str]) -> None:
    result = dispatch_simforge_argv(["simforge", "suite", "validate", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["valid"] is True
    assert payload["failures"] == []
    assert set(payload["validated_tasks"]) == {
        "body_mutation",
        "contact_push",
        "guarded_base",
        "ros2_chaos",
        "shield_reach",
    }


def test_scenario_cli_keeps_holdout_private_and_external(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "shield-reach"
    result = dispatch_simforge_argv(
        [
            "simforge",
            "scenarios",
            "generate",
            "--task",
            "shield-reach",
            "--output-dir",
            str(output),
        ]
    )
    capsys.readouterr()

    assert result == 0
    manifest = json.loads((output / "scenario_manifest.json").read_text())
    assert manifest["counts"]["total"] == 1000
    assert all(
        "risk" not in item and "pose" not in item for item in manifest["partitions"]["holdout"]
    )
    for private_name in (
        "private-seed-ledger.key",
        "holdout-private.json",
    ):
        mode = stat.S_IMODE((output / private_name).stat().st_mode)
        assert mode == 0o600


def test_simforge_rejects_raw_output_inside_source_checkout() -> None:
    checkout_output = Path(__file__).resolve().parents[2] / "raw-evidence-must-not-live-here"
    with pytest.raises(SystemExit, match="outside the source checkout"):
        dispatch_simforge_argv(
            [
                "simforge",
                "scenarios",
                "generate",
                "--task",
                "shield-reach",
                "--output-dir",
                str(checkout_output),
            ]
        )
    assert not checkout_output.exists()


def test_evolution_requires_signed_scale_shards_and_recomputes_completeness(
    tmp_path: Path,
) -> None:
    private_key, public_key = _signing_material(tmp_path)
    scale = _scale_curve()
    first_shard = scale["scales"][0]["shards"][0]  # type: ignore[index]
    assert (
        _scale_runner_module()._validate_shard(
            first_shard,
            gpu="0",
            worlds=256,
            steps=100,
            offset=0,
        )
        is None
    )
    _sign(scale, private_key)
    four_gpu, differential, stress = _verify_scale_curve(
        scale,
        expected_public_key_path=public_key,
        expected_candidate_hash=_CANDIDATE_HASH,
        expected_candidate_threshold=0.5,
    )
    assert four_gpu["worlds"] == 1024
    assert differential["critical_disagreements"] == 0
    assert stress.candidate_hash == _CANDIDATE_HASH

    forged = deepcopy(scale)
    forged["scales"][2]["shards"] = []  # type: ignore[index]
    _sign(forged, private_key)
    with pytest.raises(SystemExit, match="incomplete 4-GPU shards"):
        _verify_scale_curve(
            forged,
            expected_public_key_path=public_key,
            expected_candidate_hash=_CANDIDATE_HASH,
            expected_candidate_threshold=0.5,
        )

    tampered = deepcopy(scale)
    tampered["minimum_worlds_required"] = 999
    with pytest.raises(SystemExit, match="signature verification failed"):
        _verify_scale_curve(
            tampered,
            expected_public_key_path=public_key,
            expected_candidate_hash=_CANDIDATE_HASH,
            expected_candidate_threshold=0.5,
        )

    incomparable = deepcopy(scale)
    incomparable["scales"][1]["shards"][0]["steps"] = 99  # type: ignore[index]
    incomparable["scales"][1]["shards"][0]["world_steps"] = 256 * 99  # type: ignore[index]
    incomparable["scales"][1]["world_steps"] -= 256  # type: ignore[index,operator]
    _sign(incomparable, private_key)
    with pytest.raises(SystemExit, match="workloads are not comparable"):
        _verify_scale_curve(
            incomparable,
            expected_public_key_path=public_key,
            expected_candidate_hash=_CANDIDATE_HASH,
            expected_candidate_threshold=0.5,
        )


def test_simforge_key_cli_creates_external_non_overwriting_keys(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_key = tmp_path / "keys" / "private.pem"
    public_key = tmp_path / "keys" / "public.pem"
    private_key.parent.mkdir()
    result = dispatch_simforge_argv(
        [
            "simforge",
            "key",
            "create",
            "--private-key",
            str(private_key),
            "--public-key",
            str(public_key),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["public_key_fingerprint"].startswith("sha256:")
    assert stat.S_IMODE(private_key.stat().st_mode) == 0o600
    assert stat.S_IMODE(public_key.stat().st_mode) == 0o644
    with pytest.raises(FileExistsError):
        dispatch_simforge_argv(
            [
                "simforge",
                "key",
                "create",
                "--private-key",
                str(private_key),
                "--public-key",
                str(public_key),
            ]
        )
