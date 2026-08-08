"""Fit a replay-bound proprioceptive strike-phase selector from paired SIM probes.

The selector is deliberately small: it is a one-split expert router over the
measured pelvis yaw at the run-to-strike handoff.  Growth may learn the split
from paired counterfactual rollouts, but the resulting artifact remains
SIM-only and cannot authorize activation or hardware execution.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosclaw.feedback.contracts import canonical_hash

_MISS_PENALTY_M = 2.0


@dataclass(frozen=True)
class G1ContextualPhaseCalibration:
    normal_phase_start_frame: int
    high_yaw_phase_start_frame: int
    yaw_threshold_rad: float
    development_seeds: tuple[int, ...]
    holdout_seeds: tuple[int, ...]
    source_evidence_hashes: tuple[str, ...]
    body_hash: str
    implementation_hash: str
    experiment_context_hash: str
    development_baseline_mean_penalized_error_m: float
    development_selected_mean_penalized_error_m: float
    holdout_baseline_mean_penalized_error_m: float
    holdout_selected_mean_penalized_error_m: float
    development_baseline_precision_hits: int
    development_selected_precision_hits: int
    holdout_baseline_precision_hits: int
    holdout_selected_precision_hits: int
    selected_unsafe_episodes: int
    accepted: bool
    failure_codes: tuple[str, ...]
    calibration_hash: str
    schema_version: str = "rosclaw.growth.g1_contextual_phase_calibration.v1"

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "normal_phase_start_frame": self.normal_phase_start_frame,
            "high_yaw_phase_start_frame": self.high_yaw_phase_start_frame,
            "yaw_threshold_rad": self.yaw_threshold_rad,
            "development_seeds": list(self.development_seeds),
            "holdout_seeds": list(self.holdout_seeds),
            "source_evidence_hashes": list(self.source_evidence_hashes),
            "body_hash": self.body_hash,
            "implementation_hash": self.implementation_hash,
            "experiment_context_hash": self.experiment_context_hash,
            "miss_penalty_m": _MISS_PENALTY_M,
            "development_baseline_mean_penalized_error_m": (
                self.development_baseline_mean_penalized_error_m
            ),
            "development_selected_mean_penalized_error_m": (
                self.development_selected_mean_penalized_error_m
            ),
            "holdout_baseline_mean_penalized_error_m": (
                self.holdout_baseline_mean_penalized_error_m
            ),
            "holdout_selected_mean_penalized_error_m": (
                self.holdout_selected_mean_penalized_error_m
            ),
            "development_baseline_precision_hits": (self.development_baseline_precision_hits),
            "development_selected_precision_hits": (self.development_selected_precision_hits),
            "holdout_baseline_precision_hits": self.holdout_baseline_precision_hits,
            "holdout_selected_precision_hits": self.holdout_selected_precision_hits,
            "selected_unsafe_episodes": self.selected_unsafe_episodes,
            "accepted": self.accepted,
            "failure_codes": list(self.failure_codes),
            "evidence_domain": "SIM_ONLY",
            "promotion_truth_allowed": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
        if include_hash:
            value["calibration_hash"] = self.calibration_hash
        return value


@dataclass(frozen=True)
class _Probe:
    seed: int
    phase_start_frame: int
    handoff_yaw_rad: float
    target_error_m: float
    handoff_to_contact_sec: float
    saturation_steps: int
    precision_radius_m: float
    goal_crossed: bool
    safe: bool

    @property
    def objective(self) -> float:
        # Accuracy remains dominant.  The small secondary terms break ties in
        # favour of a more continuous and lower-authority strike.
        return (
            self.target_error_m
            + 0.03 * self.handoff_to_contact_sec
            + 0.0002 * self.saturation_steps
            + (1000.0 if not self.safe else 0.0)
        )

    @property
    def precision_hit(self) -> bool:
        return self.safe and self.goal_crossed and self.target_error_m <= self.precision_radius_m


def derive_g1_contextual_phase_calibration(
    *,
    evidence_paths: tuple[Path, ...],
    output_path: Path,
    source_checkout: Path,
    normal_phase_start_frame: int,
    high_yaw_phase_start_frame: int,
    holdout_seeds: tuple[int, ...],
    minimum_development_improvement_m: float = 0.05,
    maximum_holdout_regression_m: float = 0.01,
) -> G1ContextualPhaseCalibration:
    """Derive and gate a yaw-conditioned selector from paired phase probes."""

    if normal_phase_start_frame == high_yaw_phase_start_frame:
        raise ValueError("contextual calibration requires two distinct phase experts")
    if not 185 <= normal_phase_start_frame <= 335 or not 185 <= high_yaw_phase_start_frame <= 335:
        raise ValueError("contextual phase frames must be in [185, 335]")
    if not holdout_seeds:
        raise ValueError("contextual calibration requires declared holdout seeds")
    if len(set(holdout_seeds)) != len(holdout_seeds) or any(seed < 0 for seed in holdout_seeds):
        raise ValueError("contextual holdout seeds must be unique non-negative integers")
    if (
        not math.isfinite(minimum_development_improvement_m)
        or not 0.0 <= minimum_development_improvement_m <= 1.0
    ):
        raise ValueError("minimum development improvement must be in [0, 1] m")
    if (
        not math.isfinite(maximum_holdout_regression_m)
        or not 0.0 <= maximum_holdout_regression_m <= 0.25
    ):
        raise ValueError("maximum holdout regression must be in [0, 0.25] m")

    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("contextual calibration evidence must be outside the source checkout")
    if output.exists():
        raise FileExistsError("contextual calibration output already exists")
    if len(evidence_paths) < 8:
        raise ValueError("contextual calibration requires at least four paired seeds")

    probes: dict[int, dict[int, _Probe]] = {}
    body_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    context_hashes: set[str] = set()
    source_hashes: list[str] = []
    phases = {normal_phase_start_frame, high_yaw_phase_start_frame}
    for raw_path in evidence_paths:
        path = raw_path.expanduser().resolve()
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if evidence.get("strict_replay") is not True:
            raise ValueError("contextual calibration requires strict replay evidence")
        trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
        if not trajectory.is_file() or evidence.get("trajectory_hash") != _file_hash(trajectory):
            raise ValueError("contextual calibration trajectory binding is invalid")
        body_hashes.add(str(evidence.get("body_hash", "")))
        implementation_hashes.add(str(evidence.get("implementation_hash", "")))
        flow = dict(evidence.get("flow_config", {}))
        sonic = dict(evidence.get("sonic_runup_config", {}))
        result = evidence.get("result", {})
        phase = int(flow.get("kick_phase_start_frame", -1))
        selected_phase = int(result.get("selected_kick_phase_start_frame", -1))
        if phase not in phases or selected_phase != phase:
            raise ValueError("contextual calibration evidence used an unexpected phase")
        if float(flow.get("contextual_phase_yaw_threshold_rad", 0.0)) != 0.0:
            raise ValueError("contextual calibration probes must disable contextual routing")
        if result.get("contextual_phase_expert_executed") is not False:
            raise ValueError("contextual calibration probes must execute their declared phase")
        seed = int(sonic.pop("planner_seed", -1))
        if seed < 0 or phase in probes.setdefault(seed, {}):
            raise ValueError("contextual calibration has duplicate or invalid seed/phase probes")
        flow.pop("kick_phase_start_frame", None)
        flow.pop("contextual_phase_yaw_threshold_rad", None)
        flow.pop("contextual_high_yaw_kick_phase_start_frame", None)
        flow.pop("contextual_phase_calibration_hash", None)
        context_hashes.add(
            canonical_hash(
                {
                    "flow_config": flow,
                    "sonic_runup_config": sonic,
                    "runup_config": evidence.get("runup_config"),
                    "goal_spec": evidence.get("goal_spec"),
                }
            )
        )
        numeric = (
            result.get("handoff_yaw_rad"),
            result.get("handoff_to_contact_sec"),
            result.get("precision_radius_m"),
        )
        if not all(
            isinstance(value, (int, float)) and math.isfinite(float(value)) for value in numeric
        ):
            raise ValueError("contextual calibration probe metrics must be finite")
        raw_error = result.get("goal_plane_target_error_m")
        goal_crossed = result.get("goal_crossed") is True
        target_error_m = (
            float(raw_error)
            if goal_crossed
            and isinstance(raw_error, (int, float))
            and math.isfinite(float(raw_error))
            else _MISS_PENALTY_M
        )
        probe = _Probe(
            seed=seed,
            phase_start_frame=phase,
            handoff_yaw_rad=float(result["handoff_yaw_rad"]),
            target_error_m=target_error_m,
            handoff_to_contact_sec=float(result["handoff_to_contact_sec"]),
            saturation_steps=int(result.get("actuator_saturation_steps", 0)),
            precision_radius_m=float(result["precision_radius_m"]),
            goal_crossed=goal_crossed,
            safe=(
                result.get("finite_state") is True
                and result.get("post_kick_fall") is False
                and result.get("joint_limit_violation") is False
                and result.get("torque_limit_violation") is False
            ),
        )
        probes[seed][phase] = probe
        source_hashes.append(_file_hash(path))

    if len(set(source_hashes)) != len(source_hashes):
        raise ValueError("contextual calibration evidence paths must be independent")
    if len(body_hashes) != 1 or not next(iter(body_hashes)).startswith("sha256:"):
        raise ValueError("contextual calibration body hashes disagree")
    if len(implementation_hashes) != 1 or not next(iter(implementation_hashes)).startswith(
        "sha256:"
    ):
        raise ValueError("contextual calibration implementation hashes disagree")
    if len(context_hashes) != 1:
        raise ValueError("contextual calibration probe contexts disagree")
    if any(set(by_phase) != phases for by_phase in probes.values()):
        raise ValueError("contextual calibration requires paired phase probes per seed")
    declared_holdout = set(holdout_seeds)
    if not declared_holdout.issubset(probes):
        raise ValueError("contextual calibration holdout seed lacks paired probes")
    development = tuple(sorted(set(probes) - declared_holdout))
    holdout = tuple(sorted(declared_holdout))
    if len(development) < 3:
        raise ValueError("contextual calibration requires at least three development seeds")

    for seed, by_phase in probes.items():
        yaw_values = [item.handoff_yaw_rad for item in by_phase.values()]
        if not math.isclose(yaw_values[0], yaw_values[1], abs_tol=1e-9):
            raise ValueError(f"contextual probe handoff yaw differs across phases for seed {seed}")

    threshold = _fit_threshold(
        probes=probes,
        development_seeds=development,
        normal_phase=normal_phase_start_frame,
        high_yaw_phase=high_yaw_phase_start_frame,
    )
    development_metrics = _compare(
        probes,
        development,
        threshold,
        normal_phase_start_frame,
        high_yaw_phase_start_frame,
    )
    holdout_metrics = _compare(
        probes,
        holdout,
        threshold,
        normal_phase_start_frame,
        high_yaw_phase_start_frame,
    )
    failure_codes: list[str] = []
    if development_metrics[1] > development_metrics[0] - minimum_development_improvement_m:
        failure_codes.append("INSUFFICIENT_DEVELOPMENT_IMPROVEMENT")
    if development_metrics[3] < development_metrics[2]:
        failure_codes.append("DEVELOPMENT_PRECISION_REGRESSION")
    if holdout_metrics[1] > holdout_metrics[0] + maximum_holdout_regression_m:
        failure_codes.append("HOLDOUT_ERROR_REGRESSION")
    if holdout_metrics[3] < holdout_metrics[2]:
        failure_codes.append("HOLDOUT_PRECISION_REGRESSION")
    selected_unsafe = development_metrics[4] + holdout_metrics[4]
    if selected_unsafe:
        failure_codes.append("SELECTED_UNSAFE_EPISODE")
    accepted = not failure_codes

    unsigned: dict[str, Any] = {
        "schema_version": "rosclaw.growth.g1_contextual_phase_calibration.v1",
        "normal_phase_start_frame": normal_phase_start_frame,
        "high_yaw_phase_start_frame": high_yaw_phase_start_frame,
        "yaw_threshold_rad": threshold,
        "development_seeds": list(development),
        "holdout_seeds": list(holdout),
        "source_evidence_hashes": source_hashes,
        "body_hash": next(iter(body_hashes)),
        "implementation_hash": next(iter(implementation_hashes)),
        "experiment_context_hash": next(iter(context_hashes)),
        "miss_penalty_m": _MISS_PENALTY_M,
        "development_baseline_mean_penalized_error_m": development_metrics[0],
        "development_selected_mean_penalized_error_m": development_metrics[1],
        "holdout_baseline_mean_penalized_error_m": holdout_metrics[0],
        "holdout_selected_mean_penalized_error_m": holdout_metrics[1],
        "development_baseline_precision_hits": development_metrics[2],
        "development_selected_precision_hits": development_metrics[3],
        "holdout_baseline_precision_hits": holdout_metrics[2],
        "holdout_selected_precision_hits": holdout_metrics[3],
        "selected_unsafe_episodes": selected_unsafe,
        "accepted": accepted,
        "failure_codes": failure_codes,
        "evidence_domain": "SIM_ONLY",
        "promotion_truth_allowed": False,
        "activation_authorized": False,
        "hardware_authorized": False,
    }
    calibration = G1ContextualPhaseCalibration(
        normal_phase_start_frame=normal_phase_start_frame,
        high_yaw_phase_start_frame=high_yaw_phase_start_frame,
        yaw_threshold_rad=threshold,
        development_seeds=development,
        holdout_seeds=holdout,
        source_evidence_hashes=tuple(source_hashes),
        body_hash=next(iter(body_hashes)),
        implementation_hash=next(iter(implementation_hashes)),
        experiment_context_hash=next(iter(context_hashes)),
        development_baseline_mean_penalized_error_m=development_metrics[0],
        development_selected_mean_penalized_error_m=development_metrics[1],
        holdout_baseline_mean_penalized_error_m=holdout_metrics[0],
        holdout_selected_mean_penalized_error_m=holdout_metrics[1],
        development_baseline_precision_hits=development_metrics[2],
        development_selected_precision_hits=development_metrics[3],
        holdout_baseline_precision_hits=holdout_metrics[2],
        holdout_selected_precision_hits=holdout_metrics[3],
        selected_unsafe_episodes=selected_unsafe,
        accepted=accepted,
        failure_codes=tuple(failure_codes),
        calibration_hash=canonical_hash(unsigned),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return calibration


def load_g1_contextual_phase_calibration(path: Path) -> G1ContextualPhaseCalibration:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    claimed = value.pop("calibration_hash", None)
    if claimed != canonical_hash(value):
        raise ValueError("contextual phase calibration hash mismatch")
    if (
        value.get("evidence_domain") != "SIM_ONLY"
        or value.get("miss_penalty_m") != _MISS_PENALTY_M
        or value.get("promotion_truth_allowed") is not False
        or value.get("activation_authorized") is not False
        or value.get("hardware_authorized") is not False
    ):
        raise ValueError("contextual phase calibration safety boundary is invalid")
    calibration = G1ContextualPhaseCalibration(
        normal_phase_start_frame=int(value["normal_phase_start_frame"]),
        high_yaw_phase_start_frame=int(value["high_yaw_phase_start_frame"]),
        yaw_threshold_rad=float(value["yaw_threshold_rad"]),
        development_seeds=tuple(int(item) for item in value["development_seeds"]),
        holdout_seeds=tuple(int(item) for item in value["holdout_seeds"]),
        source_evidence_hashes=tuple(str(item) for item in value["source_evidence_hashes"]),
        body_hash=str(value["body_hash"]),
        implementation_hash=str(value["implementation_hash"]),
        experiment_context_hash=str(value["experiment_context_hash"]),
        development_baseline_mean_penalized_error_m=float(
            value["development_baseline_mean_penalized_error_m"]
        ),
        development_selected_mean_penalized_error_m=float(
            value["development_selected_mean_penalized_error_m"]
        ),
        holdout_baseline_mean_penalized_error_m=float(
            value["holdout_baseline_mean_penalized_error_m"]
        ),
        holdout_selected_mean_penalized_error_m=float(
            value["holdout_selected_mean_penalized_error_m"]
        ),
        development_baseline_precision_hits=int(value["development_baseline_precision_hits"]),
        development_selected_precision_hits=int(value["development_selected_precision_hits"]),
        holdout_baseline_precision_hits=int(value["holdout_baseline_precision_hits"]),
        holdout_selected_precision_hits=int(value["holdout_selected_precision_hits"]),
        selected_unsafe_episodes=int(value["selected_unsafe_episodes"]),
        accepted=bool(value["accepted"]),
        failure_codes=tuple(str(item) for item in value["failure_codes"]),
        calibration_hash=str(claimed),
        schema_version=str(value["schema_version"]),
    )
    if calibration.schema_version != "rosclaw.growth.g1_contextual_phase_calibration.v1":
        raise ValueError("unsupported contextual phase calibration schema")
    if (
        calibration.normal_phase_start_frame == calibration.high_yaw_phase_start_frame
        or not 185 <= calibration.normal_phase_start_frame <= 335
        or not 185 <= calibration.high_yaw_phase_start_frame <= 335
    ):
        raise ValueError("contextual phase calibration experts are invalid")
    if not 0.05 <= calibration.yaw_threshold_rad <= 0.35:
        raise ValueError("contextual phase calibration threshold is invalid")
    if (
        not calibration.development_seeds
        or not calibration.holdout_seeds
        or set(calibration.development_seeds) & set(calibration.holdout_seeds)
        or len(calibration.source_evidence_hashes)
        != 2 * (len(calibration.development_seeds) + len(calibration.holdout_seeds))
        or not all(
            value.startswith("sha256:")
            for value in (
                *calibration.source_evidence_hashes,
                calibration.body_hash,
                calibration.implementation_hash,
                calibration.experiment_context_hash,
            )
        )
    ):
        raise ValueError("contextual phase calibration provenance is invalid")
    metrics = (
        calibration.development_baseline_mean_penalized_error_m,
        calibration.development_selected_mean_penalized_error_m,
        calibration.holdout_baseline_mean_penalized_error_m,
        calibration.holdout_selected_mean_penalized_error_m,
    )
    if not all(math.isfinite(item) and item >= 0.0 for item in metrics):
        raise ValueError("contextual phase calibration metrics are invalid")
    if not calibration.accepted or calibration.failure_codes:
        raise ValueError("contextual phase calibration was not accepted")
    return calibration


def _fit_threshold(
    *,
    probes: dict[int, dict[int, _Probe]],
    development_seeds: tuple[int, ...],
    normal_phase: int,
    high_yaw_phase: int,
) -> float:
    yaws = sorted({abs(probes[seed][normal_phase].handoff_yaw_rad) for seed in development_seeds})
    candidates = {0.05, 0.35}
    candidates.update((left + right) / 2.0 for left, right in zip(yaws, yaws[1:], strict=False))
    valid = sorted(value for value in candidates if 0.05 <= value <= 0.35)
    return min(
        valid,
        key=lambda threshold: (
            sum(
                probes[seed][
                    high_yaw_phase
                    if abs(probes[seed][normal_phase].handoff_yaw_rad) >= threshold
                    else normal_phase
                ].objective
                for seed in development_seeds
            ),
            threshold,
        ),
    )


def _compare(
    probes: dict[int, dict[int, _Probe]],
    seeds: tuple[int, ...],
    threshold: float,
    normal_phase: int,
    high_yaw_phase: int,
) -> tuple[float, float, int, int, int]:
    baseline = [probes[seed][normal_phase] for seed in seeds]
    selected = [
        probes[seed][
            high_yaw_phase
            if abs(probes[seed][normal_phase].handoff_yaw_rad) >= threshold
            else normal_phase
        ]
        for seed in seeds
    ]
    return (
        sum(item.target_error_m for item in baseline) / len(baseline),
        sum(item.target_error_m for item in selected) / len(selected),
        sum(item.precision_hit for item in baseline),
        sum(item.precision_hit for item in selected),
        sum(not item.safe for item in selected),
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "G1ContextualPhaseCalibration",
    "derive_g1_contextual_phase_calibration",
    "load_g1_contextual_phase_calibration",
]
