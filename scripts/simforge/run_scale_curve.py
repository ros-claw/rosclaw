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

from rosclaw.simforge.attestation import sign_scale_curve

_MAX_SHARD_BYTES = 16 * 1024 * 1024
_MAX_WORLD_STEPS_PER_GPU = 10_000_000


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
    positive = (
        "wall_time_sec",
        "world_steps_per_sec",
        "compile_time_sec",
        "warmup_time_sec",
        "cpu_baseline_time_sec",
    )
    if any(
        isinstance(value.get(name), bool)
        or not isinstance(value.get(name), (int, float))
        or not math.isfinite(float(value[name]))
        or float(value[name]) <= 0
        for name in positive
    ):
        return "invalid_timing"
    if (
        value.get("finite_state") is not True
        or value.get("expected_collision_label") is not True
        or value.get("scenario_label_valid") is not True
    ):
        return "physics_label_failure"
    differential = value.get("differential")
    if (
        not isinstance(differential, dict)
        or differential.get("baseline_backend") != "mujoco_cpu"
        or differential.get("comparison_backend") != "mujoco_warp"
        or differential.get("critical_disagreement_count") != 0
    ):
        return "cross_backend_disagreement"
    risks = value.get("risk_values")
    collisions = value.get("collision_worlds")
    cpu_collisions = value.get("cpu_collision_worlds")
    scenario_collisions = value.get("scenario_collision_worlds")
    randomization = value.get("randomization")
    if (
        value.get("device") != "cuda:0"
        or not isinstance(value.get("device_name"), str)
        or not 1 <= len(value["device_name"]) <= 256
        or not _is_sha256(value.get("model_hash"))
        or not _is_sha256(value.get("qpos_checksum"))
        or isinstance(value.get("gpu_memory_used_bytes"), bool)
        or not isinstance(value.get("gpu_memory_used_bytes"), int)
        or value["gpu_memory_used_bytes"] < 0
        or not isinstance(randomization, dict)
        or not _is_sha256(randomization.get("parameter_hash"))
        or not _valid_offsets(randomization.get("joint_control_offset_rad"), worlds)
        or not isinstance(risks, list)
        or len(risks) != worlds
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or not 0 <= float(item) <= 1
            for item in risks
        )
        or not isinstance(collisions, list)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < worlds
            for item in collisions
        )
        or len(set(collisions)) != len(collisions)
        or cpu_collisions != collisions
        or scenario_collisions != collisions
        or value.get("collision_world_count") != len(collisions)
    ):
        return "invalid_stress_labels"
    return None


def _valid_offsets(value: Any, worlds: int) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == worlds
        and all(
            isinstance(row, list)
            and len(row) == 6
            and all(
                not isinstance(item, bool)
                and isinstance(item, (int, float))
                and math.isfinite(float(item))
                and abs(float(item)) <= 0.0021
                for item in row
            )
            for row in value
        )
    )


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _run_scale(
    *,
    gpu_ids: list[str],
    worlds_per_gpu: int,
    steps: int,
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
    failures: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    try:
        for shard_index, gpu in enumerate(launch_ids):
            shard_output = output / f"gpu_{gpu}.json"
            log = (output / f"gpu_{gpu}.log").open("w", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["ROSCLAW_PHYSICAL_GPU"] = gpu
            try:
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
                        "--output",
                        str(shard_output),
                    ],
                    cwd=worker.parents[2],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except Exception:
                log.close()
                raise
            launched.append((gpu, shard_output, log, process))
        if inject_worker_crash_gpu is not None:
            crashed = next(
                (
                    process
                    for gpu, _output, _log, process in launched
                    if gpu == inject_worker_crash_gpu
                ),
                None,
            )
            if crashed is not None:
                crashed.kill()
        deadline = started + timeout_sec
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
                if shard_output.stat().st_size > _MAX_SHARD_BYTES:
                    raise ValueError("shard evidence exceeds size limit")
                shard = json.loads(shard_output.read_text())
            except (OSError, ValueError, json.JSONDecodeError):
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
    finally:
        for _gpu, _output, _log, process in launched:
            if process.poll() is None:
                process.terminate()
        for _gpu, _output, log, process in launched:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if not log.closed:
                log.close()
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
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=float, default=1800)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inject-missing-shard", action="store_true")
    parser.add_argument("--inject-worker-crash-gpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output_dir.expanduser().resolve()
    if output.is_relative_to(root):
        parser.error("--output-dir must point outside the source checkout")
    signing_key = args.signing_key.expanduser().resolve()
    if signing_key.is_relative_to(root):
        parser.error("--signing-key must point outside the source checkout")
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
    if args.worlds_per_gpu * args.steps > _MAX_WORLD_STEPS_PER_GPU:
        parser.error(f"--worlds-per-gpu * --steps cannot exceed {_MAX_WORLD_STEPS_PER_GPU}")
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
    try:
        summary["attestation"] = sign_scale_curve(
            summary,
            private_key_path=signing_key,
        )
    except (OSError, ValueError, PermissionError) as exc:
        parser.error(str(exc))
    target = output / "scale_curve.json"
    _atomic_json(target, summary)
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


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while storing scale-curve evidence")
            view = view[written:]
        os.fsync(descriptor)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
