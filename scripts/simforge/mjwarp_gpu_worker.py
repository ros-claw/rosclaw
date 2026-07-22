#!/usr/bin/env python3
"""Small, evidence-producing MJWarp batch worker for one visible GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pose", choices=("safe", "collision"), default="collision")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mujoco_warp as mjw
    import warp as wp

    from rosclaw.sandbox.backends.fingerprints import file_hash
    from rosclaw.sandbox.sandbox_api import Sandbox

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
    model = mjw.put_model(cpu_model)
    data = mjw.put_data(
        cpu_model,
        cpu_data,
        nworld=args.worlds,
        nconmax=max(128, args.worlds * 64),
        njmax=max(512, args.worlds * 256),
    )

    rng = np.random.default_rng(args.seed)
    base = SAFE_POSE if args.pose == "safe" else COLLISION_POSE
    offsets = rng.uniform(-0.002, 0.002, size=(args.worlds, base.size)).astype(np.float32)
    controls = np.tile(base.astype(np.float32), (args.worlds, 1)) + offsets
    data.ctrl.assign(controls)

    wp.synchronize()
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
            and 30 in (int(pair[0]), int(pair[1]))
        }
    )
    qpos = data.qpos.numpy()
    payload = {
        "schema_version": "rosclaw.mjwarp_shard.v1",
        "backend": "mujoco_warp",
        "backend_version": getattr(mjw, "__version__", "unknown"),
        "warp_version": getattr(wp, "__version__", "unknown"),
        "physical_gpu": os.environ.get("ROSCLAW_PHYSICAL_GPU", "unknown"),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device": str(device),
        "device_name": device.name,
        "worlds": args.worlds,
        "steps": args.steps,
        "world_steps": args.worlds * args.steps,
        "wall_time_sec": elapsed,
        "world_steps_per_sec": args.worlds * args.steps / max(elapsed, 1e-9),
        "seed": args.seed,
        "pose": args.pose,
        "randomization": {
            "joint_control_offset_rad": offsets.tolist(),
            "parameter_hash": "sha256:" + hashlib.sha256(offsets.tobytes()).hexdigest(),
        },
        "model_hash": file_hash(sandbox.model_path),
        "collision_worlds": collision_worlds,
        "collision_world_count": len(collision_worlds),
        "all_worlds_expected_collision": (
            len(collision_worlds) == args.worlds if args.pose == "collision" else None
        ),
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
    return 0 if payload["finite_state"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
