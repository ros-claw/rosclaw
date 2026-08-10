"""Derive a replay-bound per-joint SONIC authority calibration from SIM traces."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS


@dataclass(frozen=True)
class G1SonicAuthorityCalibration:
    joint_gain_scales: tuple[float, ...]
    strike_gain_scales: tuple[float, ...]
    follow_through_gain_scales: tuple[float, ...]
    source_trajectory_hashes: tuple[str, ...]
    body_hash: str
    implementation_hash: str
    demand_quantile: float
    target_demand_ratio: float
    base_calibration_hash: str | None
    approach_gain_frozen: bool
    calibration_step_fraction: float
    calibration_hash: str
    schema_version: str = "rosclaw.growth.g1_sonic_authority_calibration.v7"

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "joint_gain_scales": list(self.joint_gain_scales),
            "strike_gain_scales": list(self.strike_gain_scales),
            "follow_through_gain_scales": list(self.follow_through_gain_scales),
            "source_trajectory_hashes": list(self.source_trajectory_hashes),
            "body_hash": self.body_hash,
            "implementation_hash": self.implementation_hash,
            "demand_quantile": self.demand_quantile,
            "target_demand_ratio": self.target_demand_ratio,
            "base_calibration_hash": self.base_calibration_hash,
            "approach_gain_frozen": self.approach_gain_frozen,
            "calibration_step_fraction": self.calibration_step_fraction,
            "evidence_domain": "SIM_ONLY",
            "promotion_truth_allowed": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
        if include_hash:
            value["calibration_hash"] = self.calibration_hash
        return value


def derive_g1_sonic_authority_calibration(
    *,
    trajectory_paths: tuple[Path, ...],
    evidence_paths: tuple[Path, ...],
    output_path: Path,
    source_checkout: Path,
    demand_quantile: float = 0.995,
    target_demand_ratio: float = 0.90,
    base_calibration_path: Path | None = None,
    freeze_approach_gain: bool = False,
    calibration_step_fraction: float = 1.0,
) -> G1SonicAuthorityCalibration:
    """Fit gain scales from measured APPROACH/ALIGN raw torque demand."""

    if not trajectory_paths or len(trajectory_paths) != len(evidence_paths):
        raise ValueError("SONIC calibration requires paired trajectory/evidence paths")
    if not 0.90 <= demand_quantile < 1.0:
        raise ValueError("SONIC calibration quantile must be in [0.90, 1.0)")
    if not 0.50 <= target_demand_ratio <= 0.98:
        raise ValueError("SONIC target demand ratio must be in [0.50, 0.98]")
    if not math.isfinite(calibration_step_fraction) or not (
        0.01 <= calibration_step_fraction <= 1.0
    ):
        raise ValueError("SONIC calibration step fraction must be in [0.01, 1]")
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("SONIC calibration evidence must be outside the source checkout")
    if output.exists():
        raise FileExistsError("SONIC calibration output already exists")
    base_calibration = (
        None
        if base_calibration_path is None
        else load_g1_sonic_authority_calibration(base_calibration_path)
    )
    ratios: list[np.ndarray] = []
    strike_ratios: list[np.ndarray] = []
    follow_through_ratios: list[np.ndarray] = []
    source_hashes: list[str] = []
    body_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    source_joint_scales: set[tuple[float, ...]] = set()
    source_strike_scales: set[tuple[float, ...]] = set()
    source_follow_through_scales: set[tuple[float, ...]] = set()
    source_calibration_hashes: set[str | None] = set()
    limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    for trajectory_path, evidence_path in zip(trajectory_paths, evidence_paths, strict=True):
        trajectory = trajectory_path.expanduser().resolve()
        evidence_file = evidence_path.expanduser().resolve()
        evidence = json.loads(evidence_file.read_text(encoding="utf-8"))
        source_hash = _file_hash(trajectory)
        if Path(str(evidence.get("trajectory_path", ""))).resolve() != trajectory:
            raise ValueError("SONIC calibration evidence does not bind its trajectory")
        if evidence.get("trajectory_hash") != source_hash:
            raise ValueError("SONIC calibration trajectory hash mismatch")
        if evidence.get("strict_replay") is not True:
            raise ValueError("SONIC calibration requires strict replay evidence")
        body_hashes.add(str(evidence.get("body_hash", "")))
        implementation_hashes.add(str(evidence.get("implementation_hash", "")))
        sonic_config = evidence.get("sonic_runup_config", {})
        flow_config = evidence.get("flow_config", {})
        source_joint_scales.add(
            _gain_scale_tuple(sonic_config.get("joint_gain_scales"), label="approach")
        )
        follow_scales = _gain_scale_tuple(
            flow_config.get("follow_through_gain_scales"), label="follow-through"
        )
        source_follow_through_scales.add(follow_scales)
        raw_strike_scales = flow_config.get("strike_gain_scales")
        if (
            raw_strike_scales is None
            and int(flow_config.get("strike_gain_schedule_start_policy_frame", 0)) > 0
        ):
            # The v12 prototype reused follow-through scales for its scheduled
            # pre-contact window. Preserve that effective controller when the
            # v13 split-field calibration is derived from those traces.
            raw_strike_scales = follow_scales
        source_strike_scales.add(_gain_scale_tuple(raw_strike_scales, label="strike"))
        source_calibration_hashes.add(flow_config.get("authority_calibration_hash"))
        with np.load(trajectory, allow_pickle=False) as archive:
            if not {"commanded_torque", "event_phase"}.issubset(archive.files):
                raise ValueError("SONIC calibration trace lacks torque or event phase")
            commanded = np.asarray(archive["commanded_torque"], dtype=np.float64)
            demand = np.asarray(
                archive[
                    "commanded_torque_peak_abs"
                    if "commanded_torque_peak_abs" in archive.files
                    else "commanded_torque"
                ],
                dtype=np.float64,
            )
            phase = np.asarray(archive["event_phase"], dtype=np.int64)
        if (
            commanded.ndim != 2
            or commanded.shape[1] != 29
            or demand.shape != commanded.shape
            or phase.shape != (len(commanded),)
        ):
            raise ValueError("SONIC calibration trace shapes are invalid")
        selected = np.isin(phase, (0, 1))
        if (
            np.count_nonzero(selected) < 10
            or not np.all(np.isfinite(commanded))
            or not np.all(np.isfinite(demand))
            or np.any(demand < 0.0)
        ):
            raise ValueError("SONIC calibration trace lacks finite approach samples")
        ratios.append(np.abs(demand[selected]) / limits[None, :])
        strike = np.isin(phase, (3, 4, 5))
        if np.count_nonzero(strike) < 10:
            raise ValueError("SONIC calibration trace lacks strike samples")
        strike_ratios.append(np.abs(demand[strike]) / limits[None, :])
        # CONTACT is already controlled by the runtime follow-through gains:
        # contact_time is latched inside the physics substeps, before the next
        # policy frame is evaluated.  Omitting phase 5 here made the learner
        # blind to the impact transient it was responsible for regulating and
        # preserved unsafe one-frame demand spikes indefinitely.
        follow_through = np.isin(phase, (5, 6))
        if np.count_nonzero(follow_through) < 10:
            raise ValueError("SONIC calibration trace lacks follow-through samples")
        follow_through_ratios.append(np.abs(demand[follow_through]) / limits[None, :])
        source_hashes.append(source_hash)
    if len(set(source_hashes)) != len(source_hashes):
        raise ValueError("SONIC calibration requires independent trajectory hashes")
    if len(body_hashes) != 1 or not next(iter(body_hashes)).startswith("sha256:"):
        raise ValueError("SONIC calibration body hashes disagree")
    if len(implementation_hashes) != 1 or not next(iter(implementation_hashes)).startswith(
        "sha256:"
    ):
        raise ValueError("SONIC calibration implementation hashes disagree")
    if any(
        len(values) != 1
        for values in (
            source_joint_scales,
            source_strike_scales,
            source_follow_through_scales,
            source_calibration_hashes,
        )
    ):
        raise ValueError("SONIC calibration sources disagree on effective gain schedules")
    body_hash = next(iter(body_hashes))
    if base_calibration is not None and base_calibration.body_hash != body_hash:
        raise ValueError("SONIC base calibration body hash differs from the new evidence")
    if base_calibration is not None and next(iter(source_calibration_hashes)) not in {
        None,
        base_calibration.calibration_hash,
    }:
        raise ValueError("SONIC evidence is not bound to the declared base calibration")
    measured = np.quantile(np.concatenate(ratios, axis=0), demand_quantile, axis=0)
    approach_adjustment = np.clip(target_demand_ratio / np.maximum(measured, 1e-9), 0.50, 1.0)
    source_approach_scales = np.asarray(next(iter(source_joint_scales)), dtype=np.float64)
    fitted_approach_scales = (
        source_approach_scales.copy()
        if freeze_approach_gain
        else np.clip(
            approach_adjustment * source_approach_scales,
            0.50,
            1.0,
        )
    )
    scales = source_approach_scales + calibration_step_fraction * (
        fitted_approach_scales - source_approach_scales
    )
    measured_strike = np.quantile(np.concatenate(strike_ratios, axis=0), demand_quantile, axis=0)
    fitted_strike_scales = np.clip(
        target_demand_ratio / np.maximum(measured_strike, 1e-9), 0.50, 1.0
    )
    source_strike_scale_array = np.asarray(next(iter(source_strike_scales)), dtype=np.float64)
    fitted_strike_scales = np.clip(
        fitted_strike_scales * source_strike_scale_array,
        0.50,
        1.0,
    )
    strike_scales = source_strike_scale_array + calibration_step_fraction * (
        fitted_strike_scales - source_strike_scale_array
    )
    measured_follow_through = np.quantile(
        np.concatenate(follow_through_ratios, axis=0), demand_quantile, axis=0
    )
    fitted_follow_through_scales = np.clip(
        target_demand_ratio / np.maximum(measured_follow_through, 1e-9),
        0.50,
        1.0,
    )
    source_follow_through_scale_array = np.asarray(
        next(iter(source_follow_through_scales)), dtype=np.float64
    )
    fitted_follow_through_scales = np.clip(
        fitted_follow_through_scales * source_follow_through_scale_array,
        0.50,
        1.0,
    )
    follow_through_scales = source_follow_through_scale_array + calibration_step_fraction * (
        fitted_follow_through_scales - source_follow_through_scale_array
    )
    unsigned = {
        "schema_version": "rosclaw.growth.g1_sonic_authority_calibration.v7",
        "joint_gain_scales": [float(value) for value in scales],
        "strike_gain_scales": [float(value) for value in strike_scales],
        "follow_through_gain_scales": [float(value) for value in follow_through_scales],
        "source_trajectory_hashes": source_hashes,
        "body_hash": body_hash,
        "implementation_hash": next(iter(implementation_hashes)),
        "demand_quantile": demand_quantile,
        "target_demand_ratio": target_demand_ratio,
        "base_calibration_hash": (
            None if base_calibration is None else base_calibration.calibration_hash
        ),
        "approach_gain_frozen": freeze_approach_gain,
        "calibration_step_fraction": calibration_step_fraction,
        "evidence_domain": "SIM_ONLY",
        "promotion_truth_allowed": False,
        "activation_authorized": False,
        "hardware_authorized": False,
    }
    calibration = G1SonicAuthorityCalibration(
        joint_gain_scales=tuple(float(value) for value in scales),
        strike_gain_scales=tuple(float(value) for value in strike_scales),
        follow_through_gain_scales=tuple(float(value) for value in follow_through_scales),
        source_trajectory_hashes=tuple(source_hashes),
        body_hash=body_hash,
        implementation_hash=next(iter(implementation_hashes)),
        demand_quantile=demand_quantile,
        target_demand_ratio=target_demand_ratio,
        base_calibration_hash=(
            None if base_calibration is None else base_calibration.calibration_hash
        ),
        approach_gain_frozen=freeze_approach_gain,
        calibration_step_fraction=calibration_step_fraction,
        calibration_hash=canonical_hash(unsigned),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return calibration


def load_g1_sonic_authority_calibration(path: Path) -> G1SonicAuthorityCalibration:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    claimed = value.pop("calibration_hash", None)
    if claimed != canonical_hash(value):
        raise ValueError("SONIC authority calibration hash mismatch")
    scales = tuple(float(item) for item in value["joint_gain_scales"])
    strike_scales = tuple(float(item) for item in value.get("strike_gain_scales", (1.0,) * 29))
    follow_through_scales = tuple(
        float(item) for item in value.get("follow_through_gain_scales", (1.0,) * 29)
    )
    if len(scales) != 29 or not all(math.isfinite(item) and 0.50 <= item <= 1.0 for item in scales):
        raise ValueError("SONIC authority calibration gain scales are invalid")
    if len(strike_scales) != 29 or not all(
        math.isfinite(item) and 0.50 <= item <= 1.0 for item in strike_scales
    ):
        raise ValueError("SONIC strike gain scales are invalid")
    if len(follow_through_scales) != 29 or not all(
        math.isfinite(item) and 0.50 <= item <= 1.0 for item in follow_through_scales
    ):
        raise ValueError("SONIC follow-through gain scales are invalid")
    step_fraction = float(value.get("calibration_step_fraction", 1.0))
    if not math.isfinite(step_fraction) or not 0.01 <= step_fraction <= 1.0:
        raise ValueError("SONIC calibration step fraction is invalid")
    return G1SonicAuthorityCalibration(
        joint_gain_scales=scales,
        strike_gain_scales=strike_scales,
        follow_through_gain_scales=follow_through_scales,
        source_trajectory_hashes=tuple(value["source_trajectory_hashes"]),
        body_hash=str(value["body_hash"]),
        implementation_hash=str(value["implementation_hash"]),
        demand_quantile=float(value["demand_quantile"]),
        target_demand_ratio=float(value["target_demand_ratio"]),
        base_calibration_hash=(
            None
            if value.get("base_calibration_hash") is None
            else str(value["base_calibration_hash"])
        ),
        approach_gain_frozen=bool(value.get("approach_gain_frozen", False)),
        calibration_step_fraction=step_fraction,
        calibration_hash=str(claimed),
        schema_version=str(value["schema_version"]),
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _gain_scale_tuple(value: Any, *, label: str) -> tuple[float, ...]:
    if value is None:
        return (1.0,) * 29
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"SONIC {label} gain scales must be a sequence")
    scales = tuple(float(item) for item in value)
    if len(scales) != 29 or not all(math.isfinite(item) and 0.50 <= item <= 1.0 for item in scales):
        raise ValueError(f"SONIC {label} gain scales are invalid")
    return scales


__all__ = [
    "G1SonicAuthorityCalibration",
    "derive_g1_sonic_authority_calibration",
    "load_g1_sonic_authority_calibration",
]
