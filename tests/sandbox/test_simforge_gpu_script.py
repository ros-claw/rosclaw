"""Contract tests for the four-GPU SimForge evidence aggregator."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[2] / "scripts" / "simforge" / "run_four_gpu.py"
    spec = importlib.util.spec_from_file_location("rosclaw_run_four_gpu", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shard() -> dict:
    return {
        "schema_version": "rosclaw.mjwarp_shard.v1",
        "physical_gpu": "2",
        "visible_devices": "2",
        "worlds": 4,
        "steps": 350,
        "world_steps": 1400,
        "pose": "collision",
        "wall_time_sec": 1.0,
        "world_steps_per_sec": 1400.0,
        "finite_state": True,
        "expected_collision_label": True,
        "differential": {
            "baseline_backend": "mujoco_cpu",
            "comparison_backend": "mujoco_warp",
            "critical_disagreement_count": 0,
        },
    }


def test_gpu_shard_contract_accepts_exact_worker_output() -> None:
    module = _module()
    assert module._shard_error(_shard(), gpu="2", worlds=4, steps=350, pose="collision") is None


def test_gpu_shard_contract_rejects_truthy_or_mismatched_evidence() -> None:
    module = _module()
    truthy = _shard()
    truthy["finite_state"] = "true"
    assert module._shard_error(truthy, gpu="2", worlds=4, steps=350, pose="collision")

    mismatched = _shard()
    mismatched["world_steps"] = 1399
    assert (
        module._shard_error(mismatched, gpu="2", worlds=4, steps=350, pose="collision")
        == "contract_mismatch"
    )
