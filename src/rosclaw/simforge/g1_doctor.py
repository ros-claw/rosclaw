"""Fail-closed environment doctor for G1 GoalForge."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.simforge.backends.unitree_mjlab_backend import qualify_unitree_mjlab
from rosclaw.simforge.backends.unitree_mujoco_backend import qualify_g1_assets
from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_bytes, hash_json


@dataclass(frozen=True)
class GoalForgeDoctorCheck:
    name: str
    passed: bool
    required: bool
    detail: str


@dataclass(frozen=True)
class GoalForgeDoctorReport:
    checks: tuple[GoalForgeDoctorCheck, ...]
    gpu_uuids: tuple[str, ...]
    body_hash: str | None
    kick_prior_hash: str | None
    external_source_commits: tuple[tuple[str, str], ...]
    real_hardware_opened: bool = False
    schema_version: str = "rosclaw.g1_goalforge.doctor.v1"

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks if check.required) and not (
            self.real_hardware_opened
        )

    @property
    def report_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "checks": [asdict(check) for check in self.checks],
            "gpu_uuids": list(self.gpu_uuids),
            "body_hash": self.body_hash,
            "kick_prior_hash": self.kick_prior_hash,
            "external_source_commits": dict(self.external_source_commits),
            "real_hardware_opened": self.real_hardware_opened,
        }


def doctor_goalforge(
    *,
    asset_root: Path,
    mjlab_root: Path,
    unitree_mujoco_root: Path,
    isaaclab_root: Path | None = None,
) -> GoalForgeDoctorReport:
    checks: list[GoalForgeDoctorCheck] = []
    assets = qualify_g1_assets(asset_root)
    checks.append(
        GoalForgeDoctorCheck(
            "official_g1_kick_assets",
            assets.eligible,
            True,
            ";".join(assets.errors) if assets.errors else assets.body_hash,
        )
    )
    mjlab = qualify_unitree_mjlab(mjlab_root)
    checks.append(
        GoalForgeDoctorCheck(
            "unitree_rl_mjlab",
            mjlab.eligible,
            True,
            ";".join(mjlab.errors) if mjlab.errors else mjlab.commit,
        )
    )
    official_model = (
        unitree_mujoco_root.expanduser().resolve() / "unitree_robots/g1/scene_29dof.xml"
    )
    checks.append(
        GoalForgeDoctorCheck(
            "unitree_mujoco_g1_29dof",
            official_model.is_file(),
            True,
            hash_bytes(official_model.read_bytes())
            if official_model.is_file()
            else str(official_model),
        )
    )
    required_imports = ("mujoco", "onnxruntime", "torch", "unitree_sdk2py", "cyclonedds")
    for module in required_imports:
        available = importlib.util.find_spec(module) is not None
        checks.append(
            GoalForgeDoctorCheck(
                f"python_import_{module}",
                available,
                True,
                "available" if available else "missing",
            )
        )
    gpu_uuids = _gpu_uuids()
    checks.append(
        GoalForgeDoctorCheck(
            "four_distinct_cuda_gpus",
            len(set(gpu_uuids)) >= 4,
            True,
            f"count={len(gpu_uuids)}",
        )
    )
    if isaaclab_root is not None:
        isaac = isaaclab_root.expanduser().resolve()
        contract = (
            (isaac / "sim_main.py").is_file()
            and (isaac / "dds/g1_robot_dds.py").is_file()
            and (isaac / "tasks/common_observations/g1_29dof_state.py").is_file()
        )
        runtime = importlib.util.find_spec("isaaclab") is not None
        checks.extend(
            (
                GoalForgeDoctorCheck(
                    "unitree_sim_isaaclab_contract",
                    contract,
                    False,
                    _git_commit(isaac) if contract else "checkout incomplete",
                ),
                GoalForgeDoctorCheck(
                    "isaaclab_runtime",
                    runtime,
                    False,
                    "available" if runtime else "not installed; optional second backend",
                ),
            )
        )
    source_roots = {
        "robonaldo": asset_root.expanduser().resolve(),
        "unitree_rl_mjlab": mjlab_root.expanduser().resolve(),
        "unitree_mujoco": unitree_mujoco_root.expanduser().resolve(),
    }
    if isaaclab_root is not None:
        source_roots["unitree_sim_isaaclab"] = isaaclab_root.expanduser().resolve()
    return GoalForgeDoctorReport(
        checks=tuple(checks),
        gpu_uuids=tuple(gpu_uuids),
        body_hash=assets.body_hash if assets.eligible else None,
        kick_prior_hash=assets.kick_prior_hash if assets.eligible else None,
        external_source_commits=tuple(
            sorted((name, _git_commit(root)) for name, root in source_roots.items())
        ),
    )


def _gpu_uuids() -> list[str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_doctor_report(report: GoalForgeDoctorReport, path: Path) -> None:
    path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "GoalForgeDoctorCheck",
    "GoalForgeDoctorReport",
    "doctor_goalforge",
    "write_doctor_report",
]
