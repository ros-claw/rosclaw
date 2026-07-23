#!/usr/bin/env python3
"""Launch isolated MJWarp shards, one process per requested GPU."""

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

_MAX_WORLDS_PER_GPU = 4096
_MAX_STEPS = 1_000_000
_MAX_TIMEOUT_SEC = 86_400.0


def _shard_error(
    shard: Any,
    *,
    gpu: str,
    worlds: int,
    steps: int,
    pose: str,
) -> str | None:
    if not isinstance(shard, dict):
        return "not_mapping"
    expected = {
        "schema_version": "rosclaw.mjwarp_shard.v1",
        "physical_gpu": gpu,
        "visible_devices": gpu,
        "worlds": worlds,
        "steps": steps,
        "world_steps": worlds * steps,
        "pose": pose,
    }
    if any(shard.get(name) != value for name, value in expected.items()):
        return "contract_mismatch"
    for name in ("wall_time_sec", "world_steps_per_sec"):
        value = shard.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            return f"invalid_{name}"
    if shard.get("finite_state") is not True:
        return "non_finite_state"
    if shard.get("expected_collision_label") is not True:
        return "collision_label_mismatch"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--worlds-per-gpu", type=int, default=4)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--timeout-sec", type=float, default=1800.0)
    parser.add_argument("--pose", choices=("safe", "collision"), default="collision")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.is_relative_to(root):
        parser.error("--output-dir must point outside the source checkout")
    python = root / ".venv-mjwarp" / "bin" / "python"
    worker = Path(__file__).with_name("mjwarp_gpu_worker.py")
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if len(gpus) != 4 or len(set(gpus)) != 4:
        parser.error("--gpus must contain exactly four distinct device identifiers")
    if not all(re.fullmatch(r"[0-9]+", gpu) for gpu in gpus):
        parser.error("--gpus accepts numeric CUDA device indices only")
    if not 1 <= args.worlds_per_gpu <= _MAX_WORLDS_PER_GPU:
        parser.error(f"--worlds-per-gpu must be between 1 and {_MAX_WORLDS_PER_GPU}")
    if not 1 <= args.steps <= _MAX_STEPS:
        parser.error(f"--steps must be between 1 and {_MAX_STEPS}")
    if not math.isfinite(args.timeout_sec) or not 0 < args.timeout_sec <= _MAX_TIMEOUT_SEC:
        parser.error(f"--timeout-sec must be between 0 and {_MAX_TIMEOUT_SEC:g}")
    if not python.is_file() or not worker.is_file():
        parser.error("MJWarp environment or worker script is missing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    launched = []
    started = time.perf_counter()
    try:
        for shard, gpu in enumerate(gpus):
            output = args.output_dir / f"gpu_{gpu}.json"
            log = (args.output_dir / f"gpu_{gpu}.log").open("w", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["ROSCLAW_PHYSICAL_GPU"] = gpu
            process = subprocess.Popen(
                [
                    str(python),
                    str(worker),
                    "--worlds",
                    str(args.worlds_per_gpu),
                    "--steps",
                    str(args.steps),
                    "--seed",
                    str(42 + shard),
                    "--pose",
                    args.pose,
                    "--output",
                    str(output),
                ],
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            launched.append((gpu, output, log, process))
    except Exception:
        for _gpu, _output, _log, process in launched:
            process.terminate()
        for _gpu, _output, log, process in launched:
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
        raise

    failures = []
    shards = []
    deadline = started + args.timeout_sec
    timed_out: set[str] = set()
    for gpu, _output, _log, process in launched:
        try:
            process.wait(timeout=max(0.0, deadline - time.perf_counter()))
        except subprocess.TimeoutExpired:
            timed_out.add(gpu)
            for _other_gpu, _other_output, _other_log, other_process in launched:
                if other_process.poll() is None:
                    other_process.terminate()
            break

    for gpu, output, log, process in launched:
        try:
            returncode = process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            returncode = process.wait()
            timed_out.add(gpu)
        log.close()
        if gpu in timed_out or returncode != 0 or not output.is_file():
            failures.append({"gpu": gpu, "returncode": returncode, "timed_out": gpu in timed_out})
        else:
            try:
                shard = json.loads(output.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                failures.append({"gpu": gpu, "returncode": returncode, "invalid_output": True})
            else:
                error = _shard_error(
                    shard,
                    gpu=gpu,
                    worlds=args.worlds_per_gpu,
                    steps=args.steps,
                    pose=args.pose,
                )
                if error is not None:
                    failures.append(
                        {
                            "gpu": gpu,
                            "returncode": returncode,
                            "invalid_output": True,
                            "reason": error,
                        }
                    )
                else:
                    shards.append(shard)
    summary = {
        "schema_version": "rosclaw.mjwarp_four_gpu.v1",
        "requested_gpus": gpus,
        "successful_gpus": [item["physical_gpu"] for item in shards],
        "failures": failures,
        "worlds": sum(int(item["worlds"]) for item in shards),
        "world_steps": sum(int(item["world_steps"]) for item in shards),
        "aggregate_world_steps_per_sec": sum(float(item["world_steps_per_sec"]) for item in shards),
        "wall_time_sec": time.perf_counter() - started,
        "finite_state": len(shards) == len(gpus)
        and all(item["finite_state"] is True for item in shards),
        "expected_collision_gate": bool(shards)
        and all(item["expected_collision_label"] is True for item in shards),
        "shards": shards,
    }
    summary_path = args.output_dir / "summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return (
        0 if not failures and summary["finite_state"] and summary["expected_collision_gate"] else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
