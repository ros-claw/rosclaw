#!/usr/bin/env python3
"""Launch isolated MJWarp shards, one process per requested GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--worlds-per-gpu", type=int, default=4)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--pose", choices=("safe", "collision"), default="collision")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    python = root / ".venv-mjwarp" / "bin" / "python"
    worker = Path(__file__).with_name("mjwarp_gpu_worker.py")
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    launched = []
    started = time.perf_counter()
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

    failures = []
    shards = []
    for gpu, output, log, process in launched:
        returncode = process.wait()
        log.close()
        if returncode != 0 or not output.is_file():
            failures.append({"gpu": gpu, "returncode": returncode})
        else:
            shards.append(json.loads(output.read_text(encoding="utf-8")))
    summary = {
        "schema_version": "rosclaw.mjwarp_four_gpu.v1",
        "requested_gpus": gpus,
        "successful_gpus": [item["physical_gpu"] for item in shards],
        "failures": failures,
        "worlds": sum(int(item["worlds"]) for item in shards),
        "world_steps": sum(int(item["world_steps"]) for item in shards),
        "aggregate_world_steps_per_sec": sum(float(item["world_steps_per_sec"]) for item in shards),
        "wall_time_sec": time.perf_counter() - started,
        "finite_state": bool(shards) and all(item["finite_state"] for item in shards),
        "expected_collision_gate": bool(shards)
        and all(item["all_worlds_expected_collision"] for item in shards),
        "shards": shards,
    }
    summary_path = args.output_dir / "summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures and summary["finite_state"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
