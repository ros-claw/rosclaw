"""Pre-contact event gate for discontinuous G1 foot-ball contact islands.

The gate learns a replay-anchored atlas of discrete event islands from strict
MuJoCo controls.  It deliberately refuses interpolation between qualified and
rejected event coordinates before a continuous actor is allowed to optimize
inside a supported island.  It has no motor, activation, promotion, or
hardware authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.ballistic_contact_residual import (
    G1_BALLISTIC_CONTACT_JOINT_NAMES,
    G1BallisticContactResidualConfig,
)

_MIN_SAMPLES = 12
_MIN_CLASS_COUNT = 4
_MIN_LOO_BALANCED_ACCURACY = 0.75
_MIN_LOO_REJECTED_RECALL = 1.0
_MAX_SKILL_ERROR_M = 0.75
_MIN_SKILL_CROSSING_HEIGHT_M = 0.65
_MAX_TRAJECTORY_BYTES = 2 * 1024 * 1024 * 1024
_EVENT_FEATURE_INDICES = (6, 7, 8)
_EVENT_TOLERANCE_FRACTION = 0.20
_MIN_EVENT_TOLERANCE = 1e-9
_EVENT_COMPARISON_EPS = 1e-12

G1_BALLISTIC_CONTACT_ISLAND_FEATURE_NAMES = (
    *G1_BALLISTIC_CONTACT_JOINT_NAMES,
    "contact_policy_frame",
    "lead_duration_sec",
    "trail_duration_sec",
)

_SUPPORT_PADDING = (0.02,) * 6 + (1.0, 0.001, 0.005)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 71 and value.startswith("sha256:") and all(
        item in "0123456789abcdef" for item in value[7:]
    )


@dataclass(frozen=True)
class G1BallisticContactIslandProbe:
    controls: tuple[float, ...]
    qualified_contact_island: bool
    hard_safe: bool
    perceptual_continuity_passed: bool
    goal_plane_target_error_m: float | None
    goal_crossing_height_m: float | None
    evidence_path: str
    evidence_hash: str
    trajectory_hash: str

    def __post_init__(self) -> None:
        _validate_controls(self.controls)
        if not _is_sha256(self.evidence_hash) or not _is_sha256(self.trajectory_hash):
            raise ValueError("ballistic contact island probe hashes are invalid")
        for value in (
            self.goal_plane_target_error_m,
            self.goal_crossing_height_m,
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError("ballistic contact island probe metric is invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class G1BallisticContactIslandDecision:
    in_replay_support: bool
    on_qualified_side: bool
    island_admissible: bool
    boundary_margin: float
    nearest_qualified_distance: float
    nearest_rejected_distance: float
    reason: str


@dataclass(frozen=True)
class G1BallisticContactIslandGate:
    event_feature_index: int
    event_feature_name: str
    event_tolerance: float
    qualified_event_values: tuple[float, ...]
    rejected_event_values: tuple[float, ...]
    qualified_support_min: tuple[float, ...]
    qualified_support_max: tuple[float, ...]
    support_padding: tuple[float, ...]
    training_balanced_accuracy: float
    leave_one_out_balanced_accuracy: float
    leave_one_out_qualified_recall: float
    leave_one_out_rejected_recall: float
    positive_count: int
    negative_count: int
    probes: tuple[G1BallisticContactIslandProbe, ...]
    body_hash: str
    implementation_hash: str
    experiment_context_hash: str
    source_evidence_hashes: tuple[str, ...]
    training_ready: bool
    failure_codes: tuple[str, ...]
    gate_hash: str
    schema_version: str = "rosclaw.growth.g1_ballistic_contact_island_gate.v3"

    def predict(self, controls: tuple[float, ...]) -> G1BallisticContactIslandDecision:
        _validate_controls(controls)
        values = np.asarray(controls, dtype=np.float64)
        lower = np.asarray(self.qualified_support_min) - np.asarray(self.support_padding)
        upper = np.asarray(self.qualified_support_max) + np.asarray(self.support_padding)
        in_support = bool(np.all(values >= lower) and np.all(values <= upper))
        event_value = float(values[self.event_feature_index])
        qualified_distance = _nearest_distance(
            event_value, self.qualified_event_values
        )
        rejected_distance = _nearest_distance(event_value, self.rejected_event_values)
        matches_qualified = (
            qualified_distance <= self.event_tolerance + _EVENT_COMPARISON_EPS
        )
        matches_rejected = (
            rejected_distance <= self.event_tolerance + _EVENT_COMPARISON_EPS
        )
        on_qualified_side = bool(matches_qualified and not matches_rejected)
        admissible = bool(self.training_ready and in_support and on_qualified_side)
        if not self.training_ready:
            reason = "GATE_NOT_TRAINING_READY"
        elif matches_rejected:
            reason = "REJECTED_CONTACT_ISLAND"
        elif not in_support:
            reason = "OUTSIDE_QUALIFIED_REPLAY_SUPPORT"
        elif not matches_qualified:
            reason = "UNSEEN_CONTACT_EVENT"
        else:
            reason = "QUALIFIED_CONTACT_ISLAND"
        return G1BallisticContactIslandDecision(
            in_replay_support=in_support,
            on_qualified_side=on_qualified_side,
            island_admissible=admissible,
            boundary_margin=rejected_distance - qualified_distance,
            nearest_qualified_distance=qualified_distance,
            nearest_rejected_distance=rejected_distance,
            reason=reason,
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "feature_names": list(G1_BALLISTIC_CONTACT_ISLAND_FEATURE_NAMES),
            "event_feature_index": self.event_feature_index,
            "event_feature_name": self.event_feature_name,
            "event_tolerance": self.event_tolerance,
            "qualified_event_values": list(self.qualified_event_values),
            "rejected_event_values": list(self.rejected_event_values),
            "qualified_support_min": list(self.qualified_support_min),
            "qualified_support_max": list(self.qualified_support_max),
            "support_padding": list(self.support_padding),
            "training_balanced_accuracy": self.training_balanced_accuracy,
            "leave_one_out_balanced_accuracy": self.leave_one_out_balanced_accuracy,
            "leave_one_out_qualified_recall": self.leave_one_out_qualified_recall,
            "leave_one_out_rejected_recall": self.leave_one_out_rejected_recall,
            "minimum_leave_one_out_balanced_accuracy": _MIN_LOO_BALANCED_ACCURACY,
            "minimum_leave_one_out_rejected_recall": _MIN_LOO_REJECTED_RECALL,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "sample_count": len(self.probes),
            "probes": [probe.to_dict() for probe in self.probes],
            "body_hash": self.body_hash,
            "implementation_hash": self.implementation_hash,
            "experiment_context_hash": self.experiment_context_hash,
            "source_evidence_hashes": list(self.source_evidence_hashes),
            "training_ready": self.training_ready,
            "failure_codes": list(self.failure_codes),
            "evidence_domain": "SIM_ONLY_DEVELOPMENT",
            "model_role": "PRE_CONTACT_DISCONTINUOUS_EVENT_GATE",
            "island_conditioned_actor_training_allowed": self.training_ready,
            "sealed_generalization_evidence": False,
            "direct_torque_output": False,
            "online_hot_swap_allowed": False,
            "activation_authorized": False,
            "promotion_authorized": False,
            "hardware_authorized": False,
        }
        if include_hash:
            value["gate_hash"] = self.gate_hash
        return value


@dataclass(frozen=True)
class _EventAxis:
    feature_index: int
    tolerance: float
    qualified_values: tuple[float, ...]
    rejected_values: tuple[float, ...]
    training_balanced_accuracy: float
    leave_one_out_balanced_accuracy: float
    leave_one_out_qualified_recall: float
    leave_one_out_rejected_recall: float


def derive_g1_ballistic_contact_island_gate(
    *,
    evidence_paths: tuple[Path, ...],
    output_path: Path,
    source_checkout: Path,
) -> G1BallisticContactIslandGate:
    """Fit a support-bound event atlas before continuous actor learning."""

    if len(evidence_paths) < _MIN_SAMPLES:
        raise ValueError(
            f"ballistic contact island gate requires at least {_MIN_SAMPLES} samples"
        )
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("ballistic contact island gate output must be outside the checkout")
    if output.exists():
        raise FileExistsError("ballistic contact island gate output already exists")

    probes: list[G1BallisticContactIslandProbe] = []
    body_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    context_hashes: set[str] = set()
    source_hashes: set[str] = set()
    seen_controls: set[tuple[float, ...]] = set()
    for raw_path in evidence_paths:
        path = raw_path.expanduser().resolve()
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if evidence.get("strict_replay") is not True:
            raise ValueError("ballistic contact island gate requires strict replay evidence")
        if dict(evidence.get("claims", {})).get(
            "contact_dynamics_observed_from_physics"
        ) is not True:
            raise ValueError("ballistic contact island gate requires physics contact evidence")
        trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
        if (
            not trajectory.is_file()
            or not 1 <= trajectory.stat().st_size <= _MAX_TRAJECTORY_BYTES
            or evidence.get("trajectory_hash") != _file_hash(trajectory)
        ):
            raise ValueError("ballistic contact island gate trajectory binding is invalid")
        _verify_contact_trace(evidence, trajectory)
        evidence_hash = _file_hash(path)
        if evidence_hash in source_hashes:
            raise ValueError("ballistic contact island gate evidence must be unique")
        source_hashes.add(evidence_hash)
        body_hashes.add(str(evidence.get("body_hash", "")))
        implementation_hashes.add(str(evidence.get("implementation_hash", "")))
        context_hashes.add(_experiment_context_hash(evidence))
        probe = _probe_from_evidence(evidence, path, evidence_hash)
        if probe.controls in seen_controls:
            raise ValueError("ballistic contact island gate controls must be independent")
        seen_controls.add(probe.controls)
        probes.append(probe)

    if len(body_hashes) != 1 or not _is_sha256(next(iter(body_hashes), "")):
        raise ValueError("ballistic contact island gate Body hashes disagree")
    if len(implementation_hashes) != 1 or not _is_sha256(
        next(iter(implementation_hashes), "")
    ):
        raise ValueError("ballistic contact island gate implementation hashes disagree")
    if len(context_hashes) != 1:
        raise ValueError("ballistic contact island gate experiment contexts disagree")

    controls = np.asarray([probe.controls for probe in probes], dtype=np.float64)
    labels = np.asarray(
        [probe.qualified_contact_island for probe in probes], dtype=np.bool_
    )
    positive_count = int(np.sum(labels))
    negative_count = len(probes) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ballistic contact island gate requires both outcome classes")
    event_axis = _fit_event_axis(controls, labels)
    loo_balanced_accuracy = event_axis.leave_one_out_balanced_accuracy

    failures: list[str] = []
    if positive_count < _MIN_CLASS_COUNT:
        failures.append("INSUFFICIENT_QUALIFIED_ISLAND_SAMPLES")
    if negative_count < _MIN_CLASS_COUNT:
        failures.append("INSUFFICIENT_REJECTED_ISLAND_SAMPLES")
    if event_axis.training_balanced_accuracy < 1.0:
        failures.append("EVENT_ATLAS_TRAINING_ERROR")
    if loo_balanced_accuracy < _MIN_LOO_BALANCED_ACCURACY:
        failures.append("LEAVE_ONE_OUT_ISLAND_ERROR_TOO_HIGH")
    if event_axis.leave_one_out_rejected_recall < _MIN_LOO_REJECTED_RECALL:
        failures.append("LEAVE_ONE_OUT_REJECTED_RECALL_TOO_LOW")
    training_ready = not failures
    qualified_controls = controls[labels]

    unsigned: dict[str, Any] = {
        "schema_version": "rosclaw.growth.g1_ballistic_contact_island_gate.v3",
        "feature_names": list(G1_BALLISTIC_CONTACT_ISLAND_FEATURE_NAMES),
        "event_feature_index": event_axis.feature_index,
        "event_feature_name": G1_BALLISTIC_CONTACT_ISLAND_FEATURE_NAMES[
            event_axis.feature_index
        ],
        "event_tolerance": event_axis.tolerance,
        "qualified_event_values": list(event_axis.qualified_values),
        "rejected_event_values": list(event_axis.rejected_values),
        "qualified_support_min": np.min(qualified_controls, axis=0).tolist(),
        "qualified_support_max": np.max(qualified_controls, axis=0).tolist(),
        "support_padding": list(_SUPPORT_PADDING),
        "training_balanced_accuracy": event_axis.training_balanced_accuracy,
        "leave_one_out_balanced_accuracy": loo_balanced_accuracy,
        "leave_one_out_qualified_recall": (
            event_axis.leave_one_out_qualified_recall
        ),
        "leave_one_out_rejected_recall": event_axis.leave_one_out_rejected_recall,
        "minimum_leave_one_out_balanced_accuracy": _MIN_LOO_BALANCED_ACCURACY,
        "minimum_leave_one_out_rejected_recall": _MIN_LOO_REJECTED_RECALL,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "sample_count": len(probes),
        "probes": [probe.to_dict() for probe in probes],
        "body_hash": next(iter(body_hashes)),
        "implementation_hash": next(iter(implementation_hashes)),
        "experiment_context_hash": next(iter(context_hashes)),
        "source_evidence_hashes": [probe.evidence_hash for probe in probes],
        "training_ready": training_ready,
        "failure_codes": failures,
        "evidence_domain": "SIM_ONLY_DEVELOPMENT",
        "model_role": "PRE_CONTACT_DISCONTINUOUS_EVENT_GATE",
        "island_conditioned_actor_training_allowed": training_ready,
        "sealed_generalization_evidence": False,
        "direct_torque_output": False,
        "online_hot_swap_allowed": False,
        "activation_authorized": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
    gate = _gate_from_dict(unsigned, canonical_hash(unsigned))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(gate.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return gate


def load_g1_ballistic_contact_island_gate(path: Path) -> G1BallisticContactIslandGate:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    claimed = str(value.pop("gate_hash", ""))
    if claimed != canonical_hash(value):
        raise ValueError("ballistic contact island gate hash mismatch")
    return _gate_from_dict(value, claimed)


def _fit_event_axis(controls: np.ndarray, labels: np.ndarray) -> _EventAxis:
    if (
        controls.ndim != 2
        or controls.shape[1] != len(G1_BALLISTIC_CONTACT_ISLAND_FEATURE_NAMES)
        or labels.shape != (len(controls),)
        or not np.all(np.isfinite(controls))
        or len(np.unique(labels)) != 2
    ):
        raise ValueError("ballistic contact island event data are invalid")
    candidates: list[tuple[tuple[float, float, float, int], _EventAxis]] = []
    for feature_index in _EVENT_FEATURE_INDICES:
        values = controls[:, feature_index]
        qualified_values = tuple(float(item) for item in np.unique(values[labels]))
        rejected_values = tuple(float(item) for item in np.unique(values[~labels]))
        tolerance, minimum_cross_distance = _event_tolerance(
            qualified_values, rejected_values
        )
        predictions = np.asarray(
            [
                _classify_event(
                    float(value), qualified_values, rejected_values, tolerance
                )
                for value in values
            ],
            dtype=np.bool_,
        )
        training_balanced = _balanced_accuracy(labels, predictions)
        loo_predictions = np.zeros(len(labels), dtype=np.bool_)
        for index, value in enumerate(values):
            keep = np.arange(len(labels)) != index
            fold_qualified = tuple(
                float(item) for item in np.unique(values[keep & labels])
            )
            fold_rejected = tuple(
                float(item) for item in np.unique(values[keep & ~labels])
            )
            fold_tolerance, _ = _event_tolerance(
                fold_qualified, fold_rejected
            )
            loo_predictions[index] = _classify_event(
                float(value), fold_qualified, fold_rejected, fold_tolerance
            )
        loo_balanced = _balanced_accuracy(labels, loo_predictions)
        loo_qualified_recall = float(np.mean(loo_predictions[labels]))
        loo_rejected_recall = float(np.mean(~loo_predictions[~labels]))
        axis = _EventAxis(
            feature_index=feature_index,
            tolerance=tolerance,
            qualified_values=qualified_values,
            rejected_values=rejected_values,
            training_balanced_accuracy=training_balanced,
            leave_one_out_balanced_accuracy=loo_balanced,
            leave_one_out_qualified_recall=loo_qualified_recall,
            leave_one_out_rejected_recall=loo_rejected_recall,
        )
        key = (
            loo_balanced,
            training_balanced,
            minimum_cross_distance,
            -feature_index,
        )
        candidates.append((key, axis))
    if not candidates:
        raise ValueError("ballistic contact island controls contain no event feature")
    return max(candidates, key=lambda item: item[0])[1]


def _event_tolerance(
    qualified_values: tuple[float, ...], rejected_values: tuple[float, ...]
) -> tuple[float, float]:
    cross_distances = np.abs(
        np.asarray(qualified_values)[:, None] - np.asarray(rejected_values)[None, :]
    )
    positive_distances = cross_distances[cross_distances > 1e-12]
    minimum_cross_distance = (
        float(np.min(positive_distances))
        if len(positive_distances)
        else _MIN_EVENT_TOLERANCE
    )
    return (
        max(
            _MIN_EVENT_TOLERANCE,
            _EVENT_TOLERANCE_FRACTION * minimum_cross_distance,
        ),
        minimum_cross_distance,
    )


def _nearest_distance(value: float, anchors: tuple[float, ...]) -> float:
    if not anchors:
        return math.inf
    return min(abs(value - anchor) for anchor in anchors)


def _classify_event(
    value: float,
    qualified_values: tuple[float, ...],
    rejected_values: tuple[float, ...],
    tolerance: float,
) -> bool:
    return bool(
        _nearest_distance(value, qualified_values)
        <= tolerance + _EVENT_COMPARISON_EPS
        and _nearest_distance(value, rejected_values)
        > tolerance + _EVENT_COMPARISON_EPS
    )


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    positive = labels
    negative = ~labels
    if not np.any(positive) or not np.any(negative):
        raise ValueError("ballistic contact island balanced accuracy needs two classes")
    return 0.5 * (
        float(np.mean(predictions[positive]))
        + float(np.mean(~predictions[negative]))
    )


def _experiment_context_hash(evidence: dict[str, Any]) -> str:
    flow = dict(evidence.get("flow_config", {}))
    for key in (
        "ballistic_contact_residual_rad",
        "ballistic_contact_policy_frame",
        "ballistic_contact_lead_duration_sec",
        "ballistic_contact_trail_duration_sec",
        "ballistic_skill_memory_hash",
        "ballistic_skill_id",
        "schema_version",
    ):
        flow.pop(key, None)
    sonic = dict(evidence.get("sonic_runup_config", {}))
    sonic.pop("schema_version", None)
    return canonical_hash(
        {
            "flow_config_without_contact_controls": flow,
            "sonic_runup_config": sonic,
            "runup_config": evidence.get("runup_config"),
            "goal_spec": evidence.get("goal_spec"),
            "approach_strike_candidate_hash": evidence.get(
                "approach_strike_candidate_hash"
            ),
        }
    )


def _probe_from_evidence(
    evidence: dict[str, Any], path: Path, evidence_hash: str
) -> G1BallisticContactIslandProbe:
    controls = _controls_from_flow(dict(evidence.get("flow_config", {})))
    result = dict(evidence.get("result", {}))
    error = _optional_non_negative(result.get("goal_plane_target_error_m"))
    crossing = result.get("goal_crossing_xyz_m")
    crossing_height = (
        _optional_non_negative(crossing[2])
        if isinstance(crossing, list) and len(crossing) == 3
        else None
    )
    hard_safe = bool(
        result.get("finite_state") is True
        and result.get("post_kick_fall") is False
        and result.get("joint_limit_violation") is False
        and result.get("torque_limit_violation") is False
    )
    continuity = result.get("perceptual_continuity_passed") is True
    qualified = bool(
        hard_safe
        and continuity
        and result.get("kick_contact_observed") is True
        and result.get("goal_crossed") is True
        and error is not None
        and error <= _MAX_SKILL_ERROR_M
        and crossing_height is not None
        and crossing_height >= _MIN_SKILL_CROSSING_HEIGHT_M
    )
    return G1BallisticContactIslandProbe(
        controls=controls,
        qualified_contact_island=qualified,
        hard_safe=hard_safe,
        perceptual_continuity_passed=continuity,
        goal_plane_target_error_m=error,
        goal_crossing_height_m=crossing_height,
        evidence_path=str(path),
        evidence_hash=evidence_hash,
        trajectory_hash=str(evidence["trajectory_hash"]),
    )


def _controls_from_flow(flow: dict[str, Any]) -> tuple[float, ...]:
    raw_action = flow.get("ballistic_contact_residual_rad")
    frame = flow.get("ballistic_contact_policy_frame")
    lead = flow.get("ballistic_contact_lead_duration_sec", 0.16)
    trail = flow.get("ballistic_contact_trail_duration_sec", 0.08)
    if (
        not isinstance(raw_action, list)
        or len(raw_action) != 6
        or not isinstance(frame, int)
        or isinstance(frame, bool)
    ):
        raise ValueError("ballistic contact island controls are missing")
    controls = tuple(float(value) for value in (*raw_action, frame, lead, trail))
    _validate_controls(controls)
    return controls


def _validate_controls(controls: tuple[float, ...]) -> None:
    if len(controls) != len(G1_BALLISTIC_CONTACT_ISLAND_FEATURE_NAMES) or not all(
        math.isfinite(value) for value in controls
    ):
        raise ValueError("ballistic contact island controls must be finite")
    frame = controls[6]
    if frame != round(frame):
        raise ValueError("ballistic contact island frame must be integral")
    G1BallisticContactResidualConfig(
        right_leg_residual_rad=tuple(controls[:6]),
        contact_policy_frame=int(frame),
        lead_duration_sec=controls[7],
        trail_duration_sec=controls[8],
    )


def _verify_contact_trace(evidence: dict[str, Any], trajectory: Path) -> None:
    with np.load(trajectory, allow_pickle=False) as archive:
        required = {
            "right_foot_linear_velocity",
            "ball_contact_force_peak_n",
            "ball_contact_normal",
            "ball_contact_force_world",
        }
        if not required.issubset(archive.files):
            raise ValueError("ballistic contact island trace lacks contact arrays")
        velocity = np.asarray(archive["right_foot_linear_velocity"], dtype=np.float64)
        peak = np.asarray(archive["ball_contact_force_peak_n"], dtype=np.float64)
        normal = np.asarray(archive["ball_contact_normal"], dtype=np.float64)
        force = np.asarray(archive["ball_contact_force_world"], dtype=np.float64)
    if (
        velocity.ndim != 2
        or velocity.shape[1:] != (3,)
        or peak.shape != (len(velocity),)
        or normal.shape != velocity.shape
        or force.shape != velocity.shape
        or not all(np.all(np.isfinite(item)) for item in (velocity, peak, normal, force))
    ):
        raise ValueError("ballistic contact island trace arrays are invalid")
    measured_peak = _optional_non_negative(
        dict(evidence.get("result", {})).get("kick_contact_peak_force_n")
    )
    if measured_peak is None or not math.isclose(
        float(np.max(peak)), measured_peak, rel_tol=1e-7, abs_tol=1e-6
    ):
        raise ValueError("ballistic contact island peak force binding is invalid")


def _optional_non_negative(value: Any) -> float | None:
    if value is None:
        return None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError("ballistic contact island metric is invalid")
    return float(value)


def _gate_from_dict(
    value: dict[str, Any], gate_hash: str
) -> G1BallisticContactIslandGate:
    if (
        value.get("schema_version")
        != "rosclaw.growth.g1_ballistic_contact_island_gate.v3"
        or tuple(value.get("feature_names", ()))
        != G1_BALLISTIC_CONTACT_ISLAND_FEATURE_NAMES
        or value.get("evidence_domain") != "SIM_ONLY_DEVELOPMENT"
        or value.get("model_role") != "PRE_CONTACT_DISCONTINUOUS_EVENT_GATE"
        or value.get("sealed_generalization_evidence") is not False
        or value.get("direct_torque_output") is not False
        or value.get("online_hot_swap_allowed") is not False
        or value.get("activation_authorized") is not False
        or value.get("promotion_authorized") is not False
        or value.get("hardware_authorized") is not False
    ):
        raise ValueError("ballistic contact island gate safety boundary is invalid")
    probes = tuple(_probe_from_dict(item) for item in value["probes"])
    source_hashes = tuple(str(item) for item in value["source_evidence_hashes"])
    support_min = tuple(float(item) for item in value["qualified_support_min"])
    support_max = tuple(float(item) for item in value["qualified_support_max"])
    padding = tuple(float(item) for item in value["support_padding"])
    failures = tuple(str(item) for item in value["failure_codes"])
    training_ready = value["training_ready"]
    positive_count = int(value["positive_count"])
    negative_count = int(value["negative_count"])
    loo = float(value["leave_one_out_balanced_accuracy"])
    loo_qualified_recall = float(value["leave_one_out_qualified_recall"])
    loo_rejected_recall = float(value["leave_one_out_rejected_recall"])
    expected_failures: list[str] = []
    if positive_count < _MIN_CLASS_COUNT:
        expected_failures.append("INSUFFICIENT_QUALIFIED_ISLAND_SAMPLES")
    if negative_count < _MIN_CLASS_COUNT:
        expected_failures.append("INSUFFICIENT_REJECTED_ISLAND_SAMPLES")
    if float(value["training_balanced_accuracy"]) < 1.0:
        expected_failures.append("EVENT_ATLAS_TRAINING_ERROR")
    if loo < _MIN_LOO_BALANCED_ACCURACY:
        expected_failures.append("LEAVE_ONE_OUT_ISLAND_ERROR_TOO_HIGH")
    if loo_rejected_recall < _MIN_LOO_REJECTED_RECALL:
        expected_failures.append("LEAVE_ONE_OUT_REJECTED_RECALL_TOO_LOW")
    feature_index = int(value["event_feature_index"])
    event_tolerance = float(value["event_tolerance"])
    qualified_event_values = tuple(
        float(item) for item in value["qualified_event_values"]
    )
    rejected_event_values = tuple(
        float(item) for item in value["rejected_event_values"]
    )
    controls = np.asarray([probe.controls for probe in probes], dtype=np.float64)
    labels = np.asarray(
        [probe.qualified_contact_island for probe in probes], dtype=np.bool_
    )
    expected_axis = _fit_event_axis(controls, labels)
    qualified_controls = controls[labels]
    expected_support_min = tuple(np.min(qualified_controls, axis=0).tolist())
    expected_support_max = tuple(np.max(qualified_controls, axis=0).tolist())
    hashes = (
        str(value["body_hash"]),
        str(value["implementation_hash"]),
        str(value["experiment_context_hash"]),
        gate_hash,
        *source_hashes,
    )
    if (
        not isinstance(training_ready, bool)
        or len(probes) != int(value["sample_count"])
        or len(probes) < _MIN_SAMPLES
        or len(source_hashes) != len(probes)
        or len(set(source_hashes)) != len(source_hashes)
        or source_hashes != tuple(probe.evidence_hash for probe in probes)
        or len(support_min) != len(G1_BALLISTIC_CONTACT_ISLAND_FEATURE_NAMES)
        or len(support_max) != len(support_min)
        or padding != _SUPPORT_PADDING
        or not all(math.isfinite(item) for item in (*support_min, *support_max))
        or any(low > high for low, high in zip(support_min, support_max, strict=True))
        or feature_index not in _EVENT_FEATURE_INDICES
        or feature_index != expected_axis.feature_index
        or value["event_feature_name"]
        != G1_BALLISTIC_CONTACT_ISLAND_FEATURE_NAMES[feature_index]
        or not math.isfinite(event_tolerance)
        or event_tolerance <= 0.0
        or not math.isclose(
            event_tolerance,
            expected_axis.tolerance,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or qualified_event_values != expected_axis.qualified_values
        or rejected_event_values != expected_axis.rejected_values
        or not 0.0 <= float(value["training_balanced_accuracy"]) <= 1.0
        or not 0.0 <= loo <= 1.0
        or not 0.0 <= loo_qualified_recall <= 1.0
        or not 0.0 <= loo_rejected_recall <= 1.0
        or not math.isclose(
            float(value["training_balanced_accuracy"]),
            expected_axis.training_balanced_accuracy,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            loo,
            expected_axis.leave_one_out_balanced_accuracy,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            loo_qualified_recall,
            expected_axis.leave_one_out_qualified_recall,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            loo_rejected_recall,
            expected_axis.leave_one_out_rejected_recall,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or support_min != expected_support_min
        or support_max != expected_support_max
        or value.get("minimum_leave_one_out_balanced_accuracy")
        != _MIN_LOO_BALANCED_ACCURACY
        or value.get("minimum_leave_one_out_rejected_recall")
        != _MIN_LOO_REJECTED_RECALL
        or positive_count != sum(probe.qualified_contact_island for probe in probes)
        or negative_count != len(probes) - positive_count
        or failures != tuple(expected_failures)
        or training_ready is not (not failures)
        or value.get("island_conditioned_actor_training_allowed") is not training_ready
        or any(
            probe.qualified_contact_island
            and (
                not probe.hard_safe
                or not probe.perceptual_continuity_passed
                or probe.goal_plane_target_error_m is None
                or probe.goal_plane_target_error_m > _MAX_SKILL_ERROR_M
                or probe.goal_crossing_height_m is None
                or probe.goal_crossing_height_m < _MIN_SKILL_CROSSING_HEIGHT_M
            )
            for probe in probes
        )
        or not all(_is_sha256(item) for item in hashes)
    ):
        raise ValueError("ballistic contact island gate geometry is invalid")
    return G1BallisticContactIslandGate(
        event_feature_index=feature_index,
        event_feature_name=str(value["event_feature_name"]),
        event_tolerance=event_tolerance,
        qualified_event_values=qualified_event_values,
        rejected_event_values=rejected_event_values,
        qualified_support_min=support_min,
        qualified_support_max=support_max,
        support_padding=padding,
        training_balanced_accuracy=float(value["training_balanced_accuracy"]),
        leave_one_out_balanced_accuracy=loo,
        leave_one_out_qualified_recall=loo_qualified_recall,
        leave_one_out_rejected_recall=loo_rejected_recall,
        positive_count=positive_count,
        negative_count=negative_count,
        probes=probes,
        body_hash=str(value["body_hash"]),
        implementation_hash=str(value["implementation_hash"]),
        experiment_context_hash=str(value["experiment_context_hash"]),
        source_evidence_hashes=source_hashes,
        training_ready=training_ready,
        failure_codes=failures,
        gate_hash=gate_hash,
    )


def _probe_from_dict(value: dict[str, Any]) -> G1BallisticContactIslandProbe:
    qualified = value["qualified_contact_island"]
    hard_safe = value["hard_safe"]
    continuity = value["perceptual_continuity_passed"]
    if not all(isinstance(item, bool) for item in (qualified, hard_safe, continuity)):
        raise ValueError("ballistic contact island probe decisions are invalid")
    return G1BallisticContactIslandProbe(
        controls=tuple(float(item) for item in value["controls"]),
        qualified_contact_island=qualified,
        hard_safe=hard_safe,
        perceptual_continuity_passed=continuity,
        goal_plane_target_error_m=_optional_non_negative(
            value["goal_plane_target_error_m"]
        ),
        goal_crossing_height_m=_optional_non_negative(
            value["goal_crossing_height_m"]
        ),
        evidence_path=str(value["evidence_path"]),
        evidence_hash=str(value["evidence_hash"]),
        trajectory_hash=str(value["trajectory_hash"]),
    )


__all__ = [
    "G1_BALLISTIC_CONTACT_ISLAND_FEATURE_NAMES",
    "G1BallisticContactIslandDecision",
    "G1BallisticContactIslandGate",
    "G1BallisticContactIslandProbe",
    "derive_g1_ballistic_contact_island_gate",
    "load_g1_ballistic_contact_island_gate",
]
