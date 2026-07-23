#!/usr/bin/env python3
"""Launch four isolated ContactPush MJWarp shards and verify completeness."""

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

from rosclaw.simforge.contact_push_stress import sign_contact_push_stress


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--worlds-per-gpu", type=int, default=250)
    parser.add_argument("--group-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--timeout-sec", type=float, default=7200.0)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[2]
    output = args.output_dir.expanduser().resolve()
    candidate = args.candidate.expanduser().resolve()
    signing_key = args.signing_key.expanduser().resolve()
    if output.is_relative_to(checkout):
        parser.error("--output-dir must stay outside the source checkout")
    if signing_key.is_relative_to(checkout):
        parser.error("--signing-key must stay outside the source checkout")
    if not candidate.is_file():
        parser.error("--candidate must identify an existing artifact")
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 4 or len(set(gpus)) != 4:
        parser.error("--gpus must contain exactly four distinct devices")
    if not all(re.fullmatch(r"[0-9]+", gpu) for gpu in gpus):
        parser.error("--gpus accepts numeric CUDA device indices only")
    if not 1 <= args.worlds_per_gpu <= 4096:
        parser.error("--worlds-per-gpu must be in [1, 4096]")
    if not 1 <= args.group_size <= 256:
        parser.error("--group-size must be in [1, 256]")
    if not math.isfinite(args.timeout_sec) or not 0 < args.timeout_sec <= 86_400:
        parser.error("--timeout-sec must be in (0, 86400]")
    python = checkout / ".venv-mjwarp" / "bin" / "python"
    worker = Path(__file__).with_name("contact_push_mjwarp_worker.py")
    if not python.is_file() or not worker.is_file():
        parser.error("MJWarp Python environment or worker is missing")
    candidate_value = json.loads(candidate.read_text(encoding="utf-8"))
    expected_candidate_hash = _candidate_hash(candidate_value)
    output.mkdir(parents=True, exist_ok=False)
    processes = []
    started = time.perf_counter()
    try:
        for shard_index, gpu in enumerate(gpus):
            shard_path = output / f"gpu_{gpu}.json"
            log = (output / f"gpu_{gpu}.log").open("w", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["ROSCLAW_PHYSICAL_GPU"] = gpu
            environment["PYTHONPATH"] = str(checkout / "src")
            process = subprocess.Popen(
                [
                    str(python),
                    str(worker),
                    "--candidate",
                    str(candidate),
                    "--worlds",
                    str(args.worlds_per_gpu),
                    "--group-size",
                    str(args.group_size),
                    "--seed",
                    str(args.seed + shard_index),
                    "--world-offset",
                    str(shard_index * args.worlds_per_gpu),
                    "--output",
                    str(shard_path),
                ],
                cwd=checkout,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            processes.append((gpu, shard_path, log, process))
    except Exception:
        _terminate(processes)
        raise
    deadline = started + args.timeout_sec
    timed_out: set[str] = set()
    for gpu, _path, _log, process in processes:
        try:
            process.wait(timeout=max(0.0, deadline - time.perf_counter()))
        except subprocess.TimeoutExpired:
            timed_out.add(gpu)
            _terminate(processes)
            break
    shards = []
    failures = []
    for gpu, path, log, process in processes:
        try:
            returncode = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            returncode = process.wait()
            timed_out.add(gpu)
        log.close()
        if returncode != 0 or gpu in timed_out or not path.is_file():
            failures.append(
                {
                    "gpu": gpu,
                    "returncode": returncode,
                    "timed_out": gpu in timed_out,
                    "log": str(path.with_suffix(".log")),
                }
            )
            continue
        try:
            shard = json.loads(path.read_text(encoding="utf-8"))
            error = _shard_error(
                shard,
                gpu=gpu,
                worlds=args.worlds_per_gpu,
                candidate_hash=expected_candidate_hash,
            )
        except (OSError, json.JSONDecodeError) as exc:
            failures.append({"gpu": gpu, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if error is not None:
            failures.append({"gpu": gpu, "error": error})
        else:
            shards.append(shard)
    worlds = sum(int(shard["worlds"]) for shard in shards)
    unique = sum(int(shard["unique_scenarios"]) for shard in shards)
    critical = sum(int(shard["critical_backend_disagreements"]) for shard in shards)
    force_violations = sum(int(shard["cpu_force_violations"]) for shard in shards)
    complete = (
        len(shards) == 4
        and worlds == args.worlds_per_gpu * 4
        and unique == worlds
        and critical == 0
        and force_violations == 0
        and all(shard["finite_state"] is True for shard in shards)
    )
    summary = {
        "schema_version": "rosclaw.contact_push_mjwarp_four_gpu.v1",
        "candidate_hash": expected_candidate_hash,
        "requested_gpus": gpus,
        "successful_gpus": [shard["physical_gpu"] for shard in shards],
        "failures": failures,
        "worlds": worlds,
        "unique_scenarios": unique,
        "world_steps": sum(int(shard["world_steps"]) for shard in shards),
        "aggregate_world_steps_per_sec": sum(
            float(shard["world_steps_per_sec"]) for shard in shards
        ),
        "wall_time_sec": time.perf_counter() - started,
        "critical_backend_disagreements": critical,
        "cpu_force_violations": force_violations,
        "minimum_exact_label_agreement_rate": (
            min(float(shard["exact_label_agreement_rate"]) for shard in shards) if shards else 0.0
        ),
        "finite_state": bool(shards) and all(shard["finite_state"] is True for shard in shards),
        "complete": complete,
        "shards": [
            {key: value for key, value in shard.items() if key != "world_records"}
            for shard in shards
        ],
    }
    summary["attestation"] = sign_contact_push_stress(
        summary,
        private_key_path=signing_key,
    )
    _atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if complete else 2


def _terminate(processes: list[tuple[str, Path, Any, subprocess.Popen[Any]]]) -> None:
    for _gpu, _path, _log, process in processes:
        if process.poll() is None:
            process.terminate()
    for _gpu, _path, _log, process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _shard_error(
    value: Any,
    *,
    gpu: str,
    worlds: int,
    candidate_hash: str,
) -> str | None:
    if not isinstance(value, dict):
        return "shard_not_mapping"
    expected = {
        "schema_version": "rosclaw.contact_push_mjwarp_shard.v1",
        "candidate_hash": candidate_hash,
        "physical_gpu": gpu,
        "visible_devices": gpu,
        "worlds": worlds,
        "unique_scenarios": worlds,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        return "shard_contract_mismatch"
    if value.get("finite_state") is not True:
        return "non_finite_state"
    if value.get("critical_backend_disagreements") != 0:
        return "critical_backend_disagreement"
    if value.get("cpu_force_violations") != 0:
        return "candidate_force_violation"
    return None


def _candidate_hash(value: dict[str, Any]) -> str:
    normalized = dict(value)
    normalized.pop("candidate_hash", None)
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return "sha256:" + __import__("hashlib").sha256(payload.encode()).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
