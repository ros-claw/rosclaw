#!/usr/bin/env python3
"""One bounded CUDA shard for GoalForge screening.

GPU screening is deliberately not presented as final physics verification.
The signed shard only prioritizes scenarios; promotion still requires CPU
MuJoCo receipts from the qualified G1 model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-rows", type=Path, required=True)
    args = parser.parse_args()
    allowed = {"practice", "candidate_search", "falsification", "private_holdout"}
    if args.role not in allowed or not 1 <= args.count <= 10_000:
        raise SystemExit("invalid GoalForge CUDA shard")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(args.physical_gpu):
        raise SystemExit(f"CUDA identity mismatch: expected {args.physical_gpu}, visible={visible}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("GoalForge CUDA shard requires exactly one visible GPU")
    identity = _gpu_identity(args.physical_gpu)
    rng = np.random.default_rng(args.root_seed)
    seeds = rng.integers(0, 2**63 - 1, size=args.count, dtype=np.int64)
    generations = np.arange(args.count, dtype=np.int64) % 11
    ball_x = rng.uniform(0.86, 1.14, args.count)
    ball_y = rng.uniform(-0.20, 0.20, args.count)
    target_y = rng.choice(np.asarray((-0.55, 0.0, 0.55)), args.count)
    target_z = rng.choice(np.asarray((0.20, 0.55, 0.90)), args.count)
    support_friction = rng.uniform(0.50, 1.20, args.count)
    latency_ms = rng.uniform(0.0, 70.0, args.count)
    disturbance_n = rng.uniform(0.0, 65.0, args.count)
    if args.role == "practice":
        support_friction = rng.uniform(0.75, 1.20, args.count)
        disturbance_n = rng.uniform(0.0, 25.0, args.count)
    elif args.role == "falsification":
        support_friction = rng.uniform(0.45, 0.68, args.count)
        disturbance_n = rng.uniform(45.0, 80.0, args.count)

    started = time.perf_counter()
    device = torch.device("cuda:0")
    values = torch.as_tensor(
        np.column_stack(
            (
                ball_x,
                ball_y,
                target_y,
                target_z,
                support_friction,
                latency_ms,
                disturbance_n,
            )
        ),
        dtype=torch.float32,
        device=device,
    )
    # Bounded differentiable screen: it ranks heading correction and flags
    # obvious slip/timing risk. It is not claimed to be MuJoCo physics.
    fixed_error = torch.sqrt(
        (values[:, 2] - (values[:, 1] * 0.35)) ** 2 + (values[:, 3] - 0.42) ** 2
    )
    heading = torch.clamp(0.35 * (values[:, 2] - values[:, 1]), -0.20, 0.20)
    learned_error = torch.clamp(
        fixed_error - 1.35 * torch.abs(heading) + values[:, 5] * 0.0015,
        min=0.0,
    )
    slip_risk = torch.clamp(
        (0.67 - values[:, 4]) * 0.24 + values[:, 6] * 0.0008,
        min=0.0,
    )
    # V2 is deliberately conservative.  The first CPU/MuJoCo calibration pass
    # showed that the earlier slip-only proxy admitted high-latency and
    # high-disturbance candidates that crossed G1 joint/COM safety boundaries.
    # This screen rejects those regions before expensive physics replay.
    safe = (
        (slip_risk <= 0.08)
        & (values[:, 4] >= 0.75)
        & (values[:, 5] <= 8.0)
        & (values[:, 6] <= 30.0)
        & (torch.abs(heading) <= 0.19)
        & (values[:, 3] <= 0.550001)
    )
    fixed_success = (fixed_error <= 0.48) & safe
    learned_success = (learned_error <= 0.40) & safe
    checksum_tensor = (
        fixed_error.sum() + learned_error.sum() + slip_risk.sum() + values.square().mean()
    )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    checksum = float(checksum_tensor.detach().cpu())
    fixed_error_np = fixed_error.detach().cpu().numpy()
    learned_error_np = learned_error.detach().cpu().numpy()
    slip_np = slip_risk.detach().cpu().numpy()
    safe_np = safe.detach().cpu().numpy()
    fixed_success_np = fixed_success.detach().cpu().numpy()
    learned_success_np = learned_success.detach().cpu().numpy()
    heading_np = heading.detach().cpu().numpy()

    rows = []
    for index in range(args.count):
        rows.append(
            {
                "scenario_commitment": _hash_json(
                    {
                        "seed": int(seeds[index]),
                        "generation": int(generations[index]),
                        "role": args.role,
                    }
                ),
                "seed": int(seeds[index]),
                "generation": int(generations[index]),
                "ball_x": float(ball_x[index]),
                "ball_y": float(ball_y[index]),
                "target_y": float(target_y[index]),
                "target_z": float(target_z[index]),
                "support_friction": float(support_friction[index]),
                "latency_ms": float(latency_ms[index]),
                "disturbance_n": float(disturbance_n[index]),
                "heading_patch": float(heading_np[index]),
                "fixed_error_proxy": float(fixed_error_np[index]),
                "candidate_error_proxy": float(learned_error_np[index]),
                "slip_risk_proxy": float(slip_np[index]),
                "safe_proxy": bool(safe_np[index]),
                "fixed_success_proxy": bool(fixed_success_np[index]),
                "candidate_success_proxy": bool(learned_success_np[index]),
            }
        )
    args.private_rows.write_text(
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )
    os.chmod(args.private_rows, 0o600)
    row_hash = _hash_bytes(args.private_rows.read_bytes())
    seed_commitment = _hash_json(sorted(row["scenario_commitment"] for row in rows))
    generations_seen = sorted({int(value) for value in generations})
    result = {
        "schema_version": "rosclaw.g1_goalforge.cuda_shard.v1",
        "role": args.role,
        "physical_gpu": args.physical_gpu,
        "gpu_uuid": identity["uuid"],
        "pci_bus_id": identity["pci.bus_id"],
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": visible,
        "scenario_count": args.count,
        "scenario_set_commitment": seed_commitment,
        "evidence_commitment": row_hash,
        "generations_seen": generations_seen,
        "aggregate": {
            "fixed_success_rate_proxy": float(np.mean(fixed_success_np)),
            "candidate_success_rate_proxy": float(np.mean(learned_success_np)),
            "safe_rate_proxy": float(np.mean(safe_np)),
            "mean_fixed_error_proxy": float(np.mean(fixed_error_np)),
            "mean_candidate_error_proxy": float(np.mean(learned_error_np)),
            "maximum_slip_risk_proxy": float(np.max(slip_np)),
        },
        "kernel": {
            "framework": f"torch-{torch.__version__}",
            "cuda_version": torch.version.cuda,
            "elapsed_ms": elapsed_ms,
            "checksum": checksum,
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "screen_model": {
            "name": "goalforge_cpu_calibrated_v3",
            "calibration_contract": "public_development_only",
            "final_truth": False,
        },
        "private_case_results_disclosed": args.role != "private_holdout",
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def _gpu_identity(index: int) -> dict[str, str]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    uuid, pci_bus_id = (value.strip() for value in output.split(",", maxsplit=1))
    return {"uuid": uuid, "pci.bus_id": pci_bus_id}


def _hash_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return _hash_bytes(payload)


def _hash_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
