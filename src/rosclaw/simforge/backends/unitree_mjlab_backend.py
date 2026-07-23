"""Qualification helpers for Unitree's public MJLab training checkout."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MJLabQualification:
    eligible: bool
    checkout: Path
    commit: str
    g1_velocity_policy: Path | None
    multi_gpu_entrypoint: Path | None
    supports_g1: bool
    supports_motion_imitation: bool
    errors: tuple[str, ...]
    schema_version: str = "rosclaw.g1_goalforge.mjlab_qualification.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["checkout"] = str(self.checkout)
        value["g1_velocity_policy"] = (
            str(self.g1_velocity_policy) if self.g1_velocity_policy else None
        )
        value["multi_gpu_entrypoint"] = (
            str(self.multi_gpu_entrypoint) if self.multi_gpu_entrypoint else None
        )
        return value

    @property
    def qualification_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()


def qualify_unitree_mjlab(checkout: Path) -> MJLabQualification:
    root = checkout.expanduser().resolve()
    train = root / "scripts/train.py"
    policy = root / "deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"
    readme = root / "README.md"
    errors = []
    text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    supports_g1 = "Unitree-G1-Flat" in text
    supports_motion = "Motion Imitation Training" in text
    if not train.is_file():
        errors.append("missing_train_entrypoint")
    if not policy.is_file():
        errors.append("missing_g1_velocity_policy")
    if "--gpu-ids" not in text:
        errors.append("multi_gpu_contract_not_documented")
    if not supports_g1:
        errors.append("g1_task_not_documented")
    if not supports_motion:
        errors.append("motion_imitation_not_documented")
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
        errors.append("git_commit_unavailable")
    return MJLabQualification(
        eligible=not errors,
        checkout=root,
        commit=commit,
        g1_velocity_policy=policy if policy.is_file() else None,
        multi_gpu_entrypoint=train if train.is_file() else None,
        supports_g1=supports_g1,
        supports_motion_imitation=supports_motion,
        errors=tuple(errors),
    )


__all__ = ["MJLabQualification", "qualify_unitree_mjlab"]
