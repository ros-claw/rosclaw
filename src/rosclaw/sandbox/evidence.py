"""Independent verification helpers for promotion-grade simulation evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from rosclaw.sandbox.backends.base import ReplayReport


@dataclass(frozen=True)
class SimulationEvidenceVerification:
    verified: bool
    replay: ReplayReport
    errors: tuple[str, ...] = ()


def _failed_report(reason: str) -> ReplayReport:
    return ReplayReport(
        verified=False,
        environment_match=False,
        hashes_verified=False,
        deterministic_label=False,
        final_qpos_max_abs_error=None,
        reason=reason,
        mismatches=(reason,),
    )


def _local_artifact_path(reference: str) -> Path | None:
    parsed = urlparse(reference)
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            return None
        return Path(unquote(parsed.path)).resolve()
    if parsed.scheme:
        return None
    return Path(reference).expanduser().resolve()


def artifacts_within(receipt: dict[str, Any], root: Path) -> bool:
    """Return whether every hash-addressed artifact is inside ``root``."""

    expected = receipt.get("artifact_hashes")
    references = receipt.get("artifacts")
    if not isinstance(expected, dict) or not expected or not isinstance(references, list):
        return False
    resolved_root = root.expanduser().resolve()
    paths: dict[str, Path] = {}
    for reference in references:
        if not isinstance(reference, str):
            continue
        path = _local_artifact_path(reference)
        if path is None or path.name in paths:
            continue
        paths[path.name] = path
    return all(
        (path := paths.get(str(name))) is not None
        and path.is_relative_to(resolved_root)
        and path.is_file()
        for name in expected
    )


def verify_simulation_receipt(receipt: dict[str, Any]) -> SimulationEvidenceVerification:
    """Independently re-execute a physics receipt and verify its core evidence."""

    errors: list[str] = []
    if receipt.get("schema_version") != "rosclaw.simulation_receipt.v1":
        errors.append("schema_version")
    if receipt.get("evidence_domain") != "SIMULATION":
        errors.append("evidence_domain")
    if receipt.get("physics_executed") is not True:
        errors.append("physics_executed")
    backend = receipt.get("backend") or {}
    if not isinstance(backend, dict) or backend.get("name") != "mujoco_cpu":
        errors.append("backend")

    embedded_replay = receipt.get("replay_report") or {}
    if not isinstance(embedded_replay, dict) or not (
        embedded_replay.get("verified") is True
        and embedded_replay.get("environment_match") is True
        and embedded_replay.get("hashes_verified") is True
        and embedded_replay.get("deterministic_label") is True
        and not embedded_replay.get("mismatches")
    ):
        errors.append("embedded_replay")
    quality = receipt.get("data_quality") or {}
    if not isinstance(quality, dict) or not (
        quality.get("artifact_hash_valid") is True
        and quality.get("body_snapshot_match") is True
        and quality.get("replayable") is True
    ):
        errors.append("data_quality")

    request = receipt.get("request") or {}
    scenario = request.get("scenario") if isinstance(request, dict) else None
    try:
        if not isinstance(scenario, dict):
            raise TypeError
        robot_id = scenario["robot_id"]
        world_id = scenario["world_id"]
        if (
            not isinstance(robot_id, str)
            or not robot_id
            or not isinstance(world_id, str)
            or not world_id
            or scenario.get("schema_version") != "rosclaw.scenario.v1"
        ):
            raise ValueError
    except (KeyError, TypeError):
        errors.append("scenario")
        report = _failed_report("SIMULATION_SCENARIO_INVALID")
        return SimulationEvidenceVerification(False, report, tuple(dict.fromkeys(errors)))
    except ValueError:
        errors.append("scenario")

    if errors:
        report = _failed_report("SIMULATION_RECEIPT_CONTRACT_INVALID")
        return SimulationEvidenceVerification(False, report, tuple(dict.fromkeys(errors)))

    from rosclaw.sandbox.backends.mujoco_cpu import MujocoCpuBackend
    from rosclaw.sandbox.sandbox_api import Sandbox

    sandbox: Sandbox | None = None
    try:
        sandbox = Sandbox.create(robot_id, world_id, "mujoco")
        report = MujocoCpuBackend(sandbox).replay(receipt, strict=True)
    except Exception as exc:  # noqa: BLE001
        report = _failed_report(f"SIMULATION_REPLAY_FAILED:{type(exc).__name__}")
    finally:
        if sandbox is not None:
            sandbox.close()
    if not report.verified:
        errors.append("strict_replay")
    return SimulationEvidenceVerification(
        not errors and report.verified,
        report,
        tuple(dict.fromkeys(errors)),
    )


def verify_promotion_receipt(receipt: dict[str, Any]) -> SimulationEvidenceVerification:
    """Re-execute a receipt and enforce the stricter promotion contract."""

    errors: list[str] = []
    if receipt.get("evaluation_variant") not in {"baseline", "candidate"}:
        errors.append("evaluation_variant")
    if not receipt.get("pair_id") or receipt.get("pair_id") != receipt.get("scenario_id"):
        errors.append("pair_id")

    raw_randomization = receipt.get("randomization")
    randomization = raw_randomization if isinstance(raw_randomization, dict) else {}
    jitter = randomization.get("initial_qpos_jitter_rad")
    seed = receipt.get("seed")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or not isinstance(raw_randomization, dict)
        or randomization.get("method") != "uniform_initial_qpos_v1"
        or randomization.get("seed_applied") is not True
        or randomization.get("seed") != seed
        or isinstance(jitter, bool)
        or not isinstance(jitter, (int, float))
        or not math.isfinite(float(jitter))
        or not 0.0 < float(jitter) <= 0.1
        or not randomization.get("initial_state_hash")
        or not randomization.get("parameter_hash")
        or not randomization.get("offset_hash")
    ):
        errors.append("seed_randomization")

    if errors:
        report = _failed_report("PROMOTION_RECEIPT_CONTRACT_INVALID")
        return SimulationEvidenceVerification(False, report, tuple(dict.fromkeys(errors)))

    verification = verify_simulation_receipt(receipt)
    return SimulationEvidenceVerification(
        verification.verified,
        verification.replay,
        verification.errors,
    )


__all__ = [
    "SimulationEvidenceVerification",
    "artifacts_within",
    "verify_promotion_receipt",
    "verify_simulation_receipt",
]
