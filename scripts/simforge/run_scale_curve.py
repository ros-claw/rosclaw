#!/usr/bin/env python3
"""Measure ShieldReach MJWarp throughput on one, two, and four isolated GPUs."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


def _validate_shard(value: Any, *, gpu: str, worlds: int, steps: int, offset: int) -> str | None:
    if not isinstance(value, dict):
        return "not_mapping"
    exact = {
        "schema_version": "rosclaw.mjwarp_shard.v1",
        "physical_gpu": gpu,
        "visible_devices": gpu,
        "worlds": worlds,
        "steps": steps,
        "world_steps": worlds * steps,
        "pose": "mixed",
        "world_offset": offset,
    }
    if any(value.get(name) != expected for name, expected in exact.items()):
        return "contract_mismatch"
    positive = ("wall_time_sec", "world_steps_per_sec", "compile_time_sec", "warmup_time_sec")
    if any(
        isinstance(value.get(name), bool)
        or not isinstance(value.get(name), (int, float))
        or not math.isfinite(float(value[name]))
        or float(value[name]) <= 0
        for name in positive
    ):
        return "invalid_timing"
    if value.get("finite_state") is not True or value.get("expected_collision_label") is not True:
        return "physics_label_failure"
    differential = value.get("differential")
    if (
        not isinstance(differential, dict)
        or differential.get("baseline_backend") != "mujoco_cpu"
        or differential.get("comparison_backend") != "mujoco_warp"
        or differential.get("critical_disagreement_count") != 0
    ):
        return "cross_backend_disagreement"
    metrics = value.get("shield_metrics")
    if not isinstance(metrics, dict) or metrics.get("candidate_unsafe_allow_count") != 0:
        return "shield_safety_failure"
    return None


def _run_scale(
    *,
    gpu_ids: list[str],
    worlds_per_gpu: int,
    steps: int,
    threshold: float,
    python: Path,
    worker: Path,
    output: Path,
    timeout_sec: float,
    inject_missing_shard: bool,
    inject_worker_crash_gpu: str | None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    launched: list[tuple[str, Path, Any, subprocess.Popen[str]]] = []
    started = time.perf_counter()
    launch_ids = gpu_ids[:-1] if inject_missing_shard else gpu_ids
    for shard_index, gpu in enumerate(launch_ids):
        shard_output = output / f"gpu_{gpu}.json"
        log = (output / f"gpu_{gpu}.log").open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["ROSCLAW_PHYSICAL_GPU"] = gpu
        process = subprocess.Popen(
            [
                str(python),
                str(worker),
                "--worlds",
                str(worlds_per_gpu),
                "--steps",
                str(steps),
                "--seed",
                str(20260723 + shard_index),
                "--pose",
                "mixed",
                "--world-offset",
                str(shard_index * worlds_per_gpu),
                "--candidate-threshold",
                str(threshold),
                "--output",
                str(shard_output),
            ],
            cwd=worker.parents[2],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        launched.append((gpu, shard_output, log, process))
    if inject_worker_crash_gpu is not None:
        crashed = next(
            (process for gpu, _output, _log, process in launched if gpu == inject_worker_crash_gpu),
            None,
        )
        if crashed is not None:
            crashed.kill()
    deadline = started + timeout_sec
    failures: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    for shard_index, (gpu, shard_output, log, process) in enumerate(launched):
        try:
            returncode = process.wait(timeout=max(0.0, deadline - time.perf_counter()))
        except subprocess.TimeoutExpired:
            process.kill()
            returncode = process.wait()
            failures.append({"gpu": gpu, "reason": "timeout"})
        finally:
            log.close()
        if returncode != 0 or not shard_output.is_file():
            failures.append({"gpu": gpu, "reason": "worker_failed", "returncode": returncode})
            continue
        try:
            shard = json.loads(shard_output.read_text())
        except (OSError, json.JSONDecodeError):
            failures.append({"gpu": gpu, "reason": "invalid_json"})
            continue
        reason = _validate_shard(
            shard,
            gpu=gpu,
            worlds=worlds_per_gpu,
            steps=steps,
            offset=shard_index * worlds_per_gpu,
        )
        if reason:
            failures.append({"gpu": gpu, "reason": reason})
        else:
            shards.append(shard)
    missing = sorted(set(gpu_ids) - {item["physical_gpu"] for item in shards})
    if missing:
        failures.append({"reason": "incomplete_shards", "missing_gpus": missing})
    complete = not failures and len(shards) == len(gpu_ids)
    return {
        "gpu_count": len(gpu_ids),
        "requested_gpus": gpu_ids,
        "worlds": len(gpu_ids) * worlds_per_gpu,
        "steps": steps,
        "world_steps": len(gpu_ids) * worlds_per_gpu * steps,
        "process_wall_time_sec": time.perf_counter() - started,
        "aggregate_world_steps_per_sec": sum(float(item["world_steps_per_sec"]) for item in shards),
        "complete": complete,
        "failures": failures,
        "shards": shards,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--worlds-per-gpu", type=int, default=256)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--candidate-threshold", type=float, default=0.5)
    parser.add_argument("--timeout-sec", type=float, default=1800)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inject-missing-shard", action="store_true")
    parser.add_argument("--inject-worker-crash-gpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output_dir.expanduser().resolve()
    if output.is_relative_to(root):
        parser.error("--output-dir must point outside the source checkout")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if (
        len(gpus) != 4
        or len(set(gpus)) != 4
        or not all(re.fullmatch(r"[0-9]+", item) for item in gpus)
    ):
        parser.error("--gpus requires four distinct numeric CUDA device indices")
    if not 1 <= args.worlds_per_gpu <= 4096:
        parser.error("--worlds-per-gpu must be in [1, 4096]")
    if not 1 <= args.steps <= 1_000_000:
        parser.error("--steps must be in [1, 1000000]")
    if not math.isfinite(args.candidate_threshold) or not 0.1 <= args.candidate_threshold <= 0.9:
        parser.error("--candidate-threshold must be in [0.1, 0.9]")
    if not math.isfinite(args.timeout_sec) or not 0 < args.timeout_sec <= 86_400:
        parser.error("--timeout-sec must be in (0, 86400]")
    if args.inject_missing_shard and args.inject_worker_crash_gpu is not None:
        parser.error("select only one injected fault")
    if args.inject_worker_crash_gpu is not None and args.inject_worker_crash_gpu not in gpus:
        parser.error("--inject-worker-crash-gpu must name one of --gpus")
    python = root / ".venv-mjwarp" / "bin" / "python"
    worker = Path(__file__).with_name("mjwarp_gpu_worker.py")
    if not python.is_file():
        parser.error("MJWarp environment is missing")
    output.mkdir(parents=True, exist_ok=True)
    fault_injected = args.inject_missing_shard or args.inject_worker_crash_gpu is not None
    counts = (4,) if fault_injected else (1, 2, 4)
    scales = []
    for count in counts:
        scales.append(
            _run_scale(
                gpu_ids=gpus[:count],
                worlds_per_gpu=args.worlds_per_gpu,
                steps=args.steps,
                threshold=args.candidate_threshold,
                python=python,
                worker=worker,
                output=output / f"gpu_count_{count}",
                timeout_sec=args.timeout_sec,
                inject_missing_shard=args.inject_missing_shard,
                inject_worker_crash_gpu=args.inject_worker_crash_gpu,
            )
        )
    baseline = scales[0]["aggregate_world_steps_per_sec"]
    for scale in scales:
        scale["speedup_vs_one_gpu"] = (
            scale["aggregate_world_steps_per_sec"] / baseline if baseline > 0 else 0.0
        )
    four_gpu = next((item for item in scales if item["gpu_count"] == 4), None)
    complete = all(item["complete"] for item in scales)
    target_met = bool(
        complete
        and four_gpu is not None
        and four_gpu["speedup_vs_one_gpu"] >= 2.5
        and four_gpu["worlds"] >= 1000
    )
    summary = {
        "schema_version": "rosclaw.simforge.scale_curve.v1",
        "task_id": "shield_reach_v1",
        "candidate_threshold": args.candidate_threshold,
        "minimum_worlds_required": 1000,
        "minimum_speedup_required": 2.5,
        "complete": complete,
        "target_met": target_met,
        "fault_injected": fault_injected,
        "fault_type": (
            "missing_shard"
            if args.inject_missing_shard
            else "worker_crash"
            if args.inject_worker_crash_gpu is not None
            else None
        ),
        "differential": {
            "baseline_backend": "mujoco_cpu",
            "comparison_backend": "mujoco_warp",
            "critical_label": "collision",
            "evaluated_worlds": sum(
                int(shard["worlds"]) for scale in scales for shard in scale.get("shards", [])
            ),
            "critical_disagreements": sum(
                int(shard["differential"]["critical_disagreement_count"])
                for scale in scales
                for shard in scale.get("shards", [])
            ),
        },
        "scales": scales,
    }
    target = output / "scale_curve.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True))
    temporary.replace(target)
    print(
        json.dumps(
            {
                "complete": complete,
                "target_met": target_met,
                "fault_type": summary["fault_type"],
                "differential": summary["differential"],
                "scales": [
                    {
                        key: scale[key]
                        for key in (
                            "gpu_count",
                            "worlds",
                            "world_steps",
                            "aggregate_world_steps_per_sec",
                            "speedup_vs_one_gpu",
                            "complete",
                            "failures",
                        )
                    }
                    for scale in scales
                ],
                "report": str(target),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if target_met else 2


if __name__ == "__main__":
    raise SystemExit(main())
