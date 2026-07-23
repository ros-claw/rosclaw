"""Stable hashes and backend fingerprints for simulation receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def file_hash(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def mujoco_fingerprint(model: Any, model_hash: str, world_asset_hash: str) -> str:
    import mujoco

    return canonical_hash(
        {
            "name": "mujoco_cpu",
            "version": getattr(mujoco, "__version__", "unknown"),
            "model_hash": model_hash,
            "world_asset_hash": world_asset_hash,
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "integrator": int(model.opt.integrator),
            "timestep_sec": float(model.opt.timestep),
            "solver_iterations": int(model.opt.iterations),
        }
    )


__all__ = ["canonical_hash", "file_hash", "mujoco_fingerprint"]
