#!/usr/bin/env python3
"""Small, evidence-producing MJWarp batch worker for one visible GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

SAFE_POSE = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
COLLISION_POSE = np.array(
    [
        3.4426358094526863,
        -0.7680767522686045,
        2.253070730803216,
        2.480201653011009,
        -5.099721659051599,
        5.976851207161098,
    ]
)
_MAX_WORLDS = 4096
_MAX_STEPS = 1_000_000
_MAX_WORLD_STEPS = 10_000_000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pose", choices=("safe", "collision", "mixed"), default="collision")
    parser.add_argument("--world-offset", type=int, default=0)
    parser.add_argument("--candidate-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    args.output = args.output.expanduser().resolve()
    if args.output.is_relative_to(repo_root):
        parser.error("--output must point outside the source checkout")
    if not 1 <= args.worlds <= _MAX_WORLDS:
        parser.error(f"--worlds must be between 1 and {_MAX_WORLDS}")
    if not 1 <= args.steps <= _MAX_STEPS:
        parser.error(f"--steps must be between 1 and {_MAX_STEPS}")
    if args.worlds * args.steps > _MAX_WORLD_STEPS:
        parser.error(f"worlds * steps cannot exceed {_MAX_WORLD_STEPS}")
    if args.world_offset < 0:
        parser.error("--world-offset must be non-negative")
    if not math.isfinite(args.candidate_threshold) or not 0.1 <= args.candidate_threshold <= 0.9:
        parser.error("--candidate-threshold must be in [0.1, 0.9]")
    physical_gpu = os.environ.get("ROSCLAW_PHYSICAL_GPU", "")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not physical_gpu.isdigit() or visible_devices != physical_gpu:
        parser.error("worker requires one matching numeric physical and visible GPU")

    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    from rosclaw.sandbox.backends.fingerprints import file_hash
    from rosclaw.sandbox.sandbox_api import Sandbox

    setup_started = time.perf_counter()
    wp.config.log_level = wp.LOG_WARNING
    wp.init()
    device = wp.get_device("cuda:0")
    free_before = int(device.free_memory)
    sandbox = Sandbox.create("ur5e", "tabletop", "mujoco")
    if not sandbox.has_physics:
        raise RuntimeError(sandbox.load_error or "PHYSICS_UNAVAILABLE")
    sandbox.reset("home")
    cpu_model = sandbox.physics_model
    cpu_data = sandbox.physics_data
    table_geom_id = mujoco.mj_name2id(
        cpu_model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "tabletop_surface",
    )
    if table_geom_id < 0:
        raise RuntimeError("TABLETOP_GEOM_NOT_FOUND")
    model = mjw.put_model(cpu_model)
    data = mjw.put_data(
        cpu_model,
        cpu_data,
        nworld=args.worlds,
        nconmax=max(128, args.worlds * 64),
        njmax=max(512, args.worlds * 256),
    )

    rng = np.random.default_rng(args.seed)
    categories: list[str] = []
    risks: list[float] = []
    scenario_collision_worlds: set[int] = set()
    if args.pose == "mixed":
        bases = np.empty((args.worlds, SAFE_POSE.size), dtype=np.float32)
        for local_world in range(args.worlds):
            bucket = (args.world_offset + local_world) % 10
            if bucket < 3:
                category, base, risk = "safe", SAFE_POSE, rng.uniform(0.08, 0.44)
            elif bucket < 6:
                category, base, risk = "unsafe", COLLISION_POSE, rng.uniform(0.56, 0.95)
                scenario_collision_worlds.add(local_world)
            elif bucket in {6, 8}:
                category, base, risk = "boundary_safe", SAFE_POSE, rng.uniform(0.44, 0.495)
            else:
                category, base, risk = (
                    "boundary_collision",
                    COLLISION_POSE,
                    rng.uniform(0.505, 0.56),
                )
                scenario_collision_worlds.add(local_world)
            categories.append(category)
            risks.append(float(risk))
            bases[local_world] = base.astype(np.float32)
    else:
        base = SAFE_POSE if args.pose == "safe" else COLLISION_POSE
        bases = np.tile(base.astype(np.float32), (args.worlds, 1))
        categories = [args.pose] * args.worlds
        risks = [0.2 if args.pose == "safe" else 0.8] * args.worlds
        if args.pose == "collision":
            scenario_collision_worlds = set(range(args.worlds))
    offsets = rng.uniform(-0.002, 0.002, size=(args.worlds, SAFE_POSE.size)).astype(np.float32)
    controls = bases + offsets
    data.ctrl.assign(controls)

    wp.synchronize()
    setup_time = time.perf_counter() - setup_started
    cpu_baseline_started = time.perf_counter()
    cpu_collision_worlds = _cpu_collision_labels(
        mujoco=mujoco,
        model=cpu_model,
        initial_data=cpu_data,
        controls=controls,
        steps=args.steps + 1,
        table_geom_id=table_geom_id,
    )
    cpu_baseline_time = time.perf_counter() - cpu_baseline_started

    warmup_started = time.perf_counter()
    mjw.step(model, data)
    wp.synchronize()
    warmup_time = time.perf_counter() - warmup_started
    started = time.perf_counter()
    for _ in range(args.steps):
        mjw.step(model, data)
    wp.synchronize()
    elapsed = time.perf_counter() - started

    world_ids = data.contact.worldid.numpy()
    geom_pairs = data.contact.geom.numpy()
    distances = data.contact.dist.numpy()
    collision_worlds = sorted(
        {
            int(world_id)
            for world_id, pair, distance in zip(world_ids, geom_pairs, distances, strict=True)
            if 0 <= int(world_id) < args.worlds
            and float(distance) < -1e-6
            and table_geom_id in (int(pair[0]), int(pair[1]))
        }
    )
    qpos = data.qpos.numpy()
    actual_collision_worlds = set(collision_worlds)
    expected_collision_label = actual_collision_worlds == cpu_collision_worlds
    scenario_label_valid = cpu_collision_worlds == scenario_collision_worlds
    critical_disagreement_worlds = sorted(
        actual_collision_worlds.symmetric_difference(cpu_collision_worlds)
    )
    baseline_threshold = 0.82
    baseline_allowed = [risk <= baseline_threshold for risk in risks]
    candidate_allowed = [risk <= args.candidate_threshold for risk in risks]
    physically_safe = [index not in actual_collision_worlds for index in range(args.worlds)]
    baseline_unsafe_allow = sum(
        allowed and not safe
        for allowed, safe in zip(baseline_allowed, physically_safe, strict=True)
    )
    candidate_unsafe_allow = sum(
        allowed and not safe
        for allowed, safe in zip(candidate_allowed, physically_safe, strict=True)
    )
    candidate_false_block = sum(
        not allowed and safe
        for allowed, safe in zip(candidate_allowed, physically_safe, strict=True)
    )
    payload = {
        "schema_version": "rosclaw.mjwarp_shard.v1",
        "backend": "mujoco_warp",
        "backend_version": getattr(mjw, "__version__", "unknown"),
        "warp_version": getattr(wp, "__version__", "unknown"),
        "physical_gpu": physical_gpu,
        "visible_devices": visible_devices,
        "device": str(device),
        "device_name": device.name,
        "worlds": args.worlds,
        "steps": args.steps,
        "world_steps": args.worlds * args.steps,
        "wall_time_sec": elapsed if math.isfinite(elapsed) else 0.0,
        "compile_time_sec": setup_time if math.isfinite(setup_time) else 0.0,
        "warmup_time_sec": warmup_time if math.isfinite(warmup_time) else 0.0,
        "cpu_baseline_time_sec": (cpu_baseline_time if math.isfinite(cpu_baseline_time) else 0.0),
        "world_steps_per_sec": (
            args.worlds * args.steps / max(elapsed, 1e-9) if math.isfinite(elapsed) else 0.0
        ),
        "seed": args.seed,
        "pose": args.pose,
        "world_offset": args.world_offset,
        "category_counts": {
            category: categories.count(category) for category in sorted(set(categories))
        },
        "risk_values": risks,
        "candidate_threshold": args.candidate_threshold,
        "randomization": {
            "joint_control_offset_rad": offsets.tolist(),
            "parameter_hash": "sha256:" + hashlib.sha256(offsets.tobytes()).hexdigest(),
        },
        "model_hash": file_hash(sandbox.model_path),
        "collision_worlds": collision_worlds,
        "collision_world_count": len(collision_worlds),
        "cpu_collision_worlds": sorted(cpu_collision_worlds),
        "scenario_collision_worlds": sorted(scenario_collision_worlds),
        "scenario_label_valid": scenario_label_valid,
        "expected_collision_label": expected_collision_label,
        "differential": {
            "baseline_backend": "mujoco_cpu",
            "comparison_backend": "mujoco_warp",
            "critical_label": "collision",
            "critical_disagreement_count": len(critical_disagreement_worlds),
            "critical_disagreement_worlds": critical_disagreement_worlds,
        },
        "shield_metrics": {
            "baseline_unsafe_allow_count": baseline_unsafe_allow,
            "candidate_unsafe_allow_count": candidate_unsafe_allow,
            "candidate_false_block_count": candidate_false_block,
            "candidate_decision_accuracy": sum(
                allowed == safe
                for allowed, safe in zip(candidate_allowed, physically_safe, strict=True)
            )
            / args.worlds,
        },
        "finite_state": bool(np.isfinite(qpos).all()),
        "qpos_checksum": "sha256:" + hashlib.sha256(qpos.tobytes()).hexdigest(),
        "gpu_memory_used_bytes": max(0, free_before - int(device.free_memory)),
    }
    sandbox.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, sort_keys=True))
    return (
        0
        if payload["finite_state"]
        and payload["expected_collision_label"]
        and payload["scenario_label_valid"]
        else 2
    )


def _cpu_collision_labels(
    *,
    mujoco: object,
    model: object,
    initial_data: object,
    controls: np.ndarray,
    steps: int,
    table_geom_id: int,
) -> set[int]:
    collision_worlds: set[int] = set()
    data = mujoco.MjData(model)  # type: ignore[attr-defined]
    for world, control in enumerate(controls):
        mujoco.mj_resetData(model, data)  # type: ignore[attr-defined]
        data.qpos[:] = initial_data.qpos  # type: ignore[attr-defined]
        data.qvel[:] = initial_data.qvel  # type: ignore[attr-defined]
        if model.na:  # type: ignore[attr-defined]
            data.act[:] = initial_data.act  # type: ignore[attr-defined]
        data.ctrl[:] = control
        mujoco.mj_forward(model, data)  # type: ignore[attr-defined]
        for _ in range(steps):
            mujoco.mj_step(model, data)  # type: ignore[attr-defined]
        if any(
            float(data.contact[index].dist) < -1e-6
            and table_geom_id
            in (
                int(data.contact[index].geom1),
                int(data.contact[index].geom2),
            )
            for index in range(int(data.ncon))
        ):
            collision_worlds.add(world)
    return collision_worlds


if __name__ == "__main__":
    raise SystemExit(main())
