from __future__ import annotations

import json
import stat
from copy import deepcopy
from pathlib import Path

import pytest

from rosclaw.simforge.cli import _verify_scale_curve, dispatch_simforge_argv


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
                    "world_steps_per_sec": 100.0,
                    "finite_state": True,
                    "expected_collision_label": True,
                    "candidate_threshold": 0.5,
                    "differential": {
                        "baseline_backend": "mujoco_cpu",
                        "comparison_backend": "mujoco_warp",
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
                "complete": True,
                "worlds": worlds,
                "world_steps": worlds * 100,
                "aggregate_world_steps_per_sec": count * 100.0,
                "speedup_vs_one_gpu": float(count),
                "shards": shards,
            }
        )
    return {
        "schema_version": "rosclaw.simforge.scale_curve.v1",
        "candidate_threshold": 0.5,
        "complete": True,
        "target_met": True,
        "scales": scales,
        "differential": {
            "baseline_backend": "mujoco_cpu",
            "comparison_backend": "mujoco_warp",
            "critical_disagreements": 0,
            "evaluated_worlds": evaluated_worlds,
        },
    }


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


def test_evolution_recomputes_scale_shard_completeness() -> None:
    scale = _scale_curve()
    four_gpu, differential = _verify_scale_curve(scale)
    assert four_gpu["worlds"] == 1024
    assert differential["critical_disagreements"] == 0

    forged = deepcopy(scale)
    forged["scales"][2]["shards"] = []  # type: ignore[index]
    with pytest.raises(SystemExit, match="incomplete 4-GPU shards"):
        _verify_scale_curve(forged)
