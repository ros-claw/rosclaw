"""Full-state SIM_ONLY skill-island memory for G1 ballistic contact.

The memory binds a deterministic SONIC plan, its measured handoff state, and
one bounded contact action.  Runtime selection is nearest-prototype with an
explicit support radius and ambiguity margin.  An unknown handoff state is an
abstention, never an invitation to broadcast a successful action out of
distribution.
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
from rosclaw.growth.ballistic_contact_residual import G1BallisticContactResidualConfig

_MAX_TRAJECTORY_BYTES = 2 * 1024 * 1024 * 1024
_MAX_SKILL_ERROR_M = 0.75
_MIN_SKILL_CROSSING_HEIGHT_M = 0.65

G1_BALLISTIC_HANDOFF_GROUP_NAMES = (
    "pelvis_pose_xyz_rpy",
    "joint_position",
    "pelvis_velocity_linear_angular",
    "joint_velocity",
)
_PELVIS_POSE_SCALE = np.asarray((0.15, 0.15, 0.08, 0.15, 0.15, 0.15))
_JOINT_POSITION_SCALE = np.full(29, 0.25, dtype=np.float64)
_PELVIS_VELOCITY_SCALE = np.asarray((0.30, 0.30, 0.30, 0.50, 0.50, 0.50))
_JOINT_VELOCITY_SCALE = np.full(29, 0.80, dtype=np.float64)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 71 and value.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in value[7:]
    )


def ballistic_skill_experiment_context_hash(
    *,
    flow_config: dict[str, Any],
    sonic_runup_config: dict[str, Any],
    runup_config: dict[str, Any],
    goal_spec: dict[str, Any],
    approach_strike_candidate_hash: str | None,
) -> str:
    """Hash every task input except the selected skill and deterministic seed."""

    flow = dict(flow_config)
    sonic = dict(sonic_runup_config)
    for key in (
        "ballistic_contact_residual_rad",
        "ballistic_contact_policy_frame",
        "post_contact_damping_scale",
        "ballistic_skill_memory_hash",
        "ballistic_skill_id",
        "schema_version",
    ):
        flow.pop(key, None)
    for key in ("planner_seed", "schema_version"):
        sonic.pop(key, None)
    return canonical_hash(
        {
            "flow_config_without_skill": flow,
            "sonic_runup_config_without_seed": sonic,
            "runup_config": runup_config,
            "goal_spec": goal_spec,
            "approach_strike_candidate_hash": approach_strike_candidate_hash,
        }
    )


@dataclass(frozen=True)
class G1BallisticHandoffState:
    pelvis_pose_xyz_rpy: tuple[float, ...]
    joint_position: tuple[float, ...]
    pelvis_velocity_linear_angular: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    schema_version: str = "rosclaw.growth.g1_ballistic_handoff_state.v1"

    def __post_init__(self) -> None:
        groups = self.groups
        if tuple(len(group) for group in groups) != (6, 29, 6, 29):
            raise ValueError("ballistic handoff state has invalid group dimensions")
        if not all(math.isfinite(value) for group in groups for value in group):
            raise ValueError("ballistic handoff state must be finite")

    @property
    def groups(self) -> tuple[tuple[float, ...], ...]:
        return (
            self.pelvis_pose_xyz_rpy,
            self.joint_position,
            self.pelvis_velocity_linear_angular,
            self.joint_velocity,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class G1BallisticSkillPrototype:
    skill_id: str
    planner_seed: int
    state: G1BallisticHandoffState
    action_rad: tuple[float, ...]
    contact_policy_frame: int
    post_contact_damping_scale: float
    goal_plane_target_error_m: float
    goal_crossing_height_m: float
    evidence_path: str
    evidence_hash: str
    trajectory_hash: str

    def __post_init__(self) -> None:
        if not self.skill_id or self.planner_seed < 0:
            raise ValueError("ballistic skill identity is invalid")
        G1BallisticContactResidualConfig(
            right_leg_residual_rad=self.action_rad,
            contact_policy_frame=self.contact_policy_frame,
        )
        if not 1.0 <= self.post_contact_damping_scale <= 2.5:
            raise ValueError("ballistic skill damping is outside the SIM envelope")
        metrics = (self.goal_plane_target_error_m, self.goal_crossing_height_m)
        if not all(math.isfinite(value) and value >= 0.0 for value in metrics):
            raise ValueError("ballistic skill metrics must be finite and non-negative")
        if not _is_sha256(self.evidence_hash) or not _is_sha256(self.trajectory_hash):
            raise ValueError("ballistic skill provenance hashes are invalid")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["state"] = self.state.to_dict()
        return value


@dataclass(frozen=True)
class G1BallisticSkillSelection:
    selected_skill_id: str | None
    selected_planner_seed: int | None
    abstained: bool
    failure_code: str | None
    nearest_distance: float
    distance_margin: float


@dataclass(frozen=True)
class G1BallisticSkillMemory:
    prototypes: tuple[G1BallisticSkillPrototype, ...]
    maximum_support_distance: float
    minimum_distance_margin: float
    rejected_seeds: tuple[int, ...]
    rejected_nearest_distances: tuple[float, ...]
    source_evidence_hashes: tuple[str, ...]
    rejected_evidence_hashes: tuple[str, ...]
    body_hash: str
    implementation_hash: str
    experiment_context_hash: str
    minimum_rejected_distance: float
    accepted: bool
    failure_codes: tuple[str, ...]
    memory_hash: str
    schema_version: str = "rosclaw.growth.g1_ballistic_skill_memory.v1"

    def select(self, state: G1BallisticHandoffState) -> G1BallisticSkillSelection:
        distances = np.asarray(
            [ballistic_handoff_distance(state, item.state) for item in self.prototypes],
            dtype=np.float64,
        )
        order = np.argsort(distances)
        nearest = float(distances[order[0]])
        margin = float(distances[order[1]] - nearest)
        if nearest > self.maximum_support_distance:
            return G1BallisticSkillSelection(
                selected_skill_id=None,
                selected_planner_seed=None,
                abstained=True,
                failure_code="OUT_OF_DISTRIBUTION_HANDOFF",
                nearest_distance=nearest,
                distance_margin=margin,
            )
        if margin < self.minimum_distance_margin:
            return G1BallisticSkillSelection(
                selected_skill_id=None,
                selected_planner_seed=None,
                abstained=True,
                failure_code="AMBIGUOUS_SKILL_ISLAND",
                nearest_distance=nearest,
                distance_margin=margin,
            )
        prototype = self.prototypes[int(order[0])]
        return G1BallisticSkillSelection(
            selected_skill_id=prototype.skill_id,
            selected_planner_seed=prototype.planner_seed,
            abstained=False,
            failure_code=None,
            nearest_distance=nearest,
            distance_margin=margin,
        )

    @property
    def best_prototype(self) -> G1BallisticSkillPrototype:
        return min(
            self.prototypes,
            key=lambda item: (item.goal_plane_target_error_m, item.planner_seed),
        )

    def prototype(self, skill_id: str) -> G1BallisticSkillPrototype:
        matches = [item for item in self.prototypes if item.skill_id == skill_id]
        if len(matches) != 1:
            raise ValueError(f"ballistic skill is not registered exactly once: {skill_id}")
        return matches[0]

    def prototype_for_seed(self, planner_seed: int) -> G1BallisticSkillPrototype:
        matches = [item for item in self.prototypes if item.planner_seed == planner_seed]
        if len(matches) != 1:
            raise ValueError(
                f"planner seed is not registered exactly once in skill memory: {planner_seed}"
            )
        return matches[0]

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "prototypes": [item.to_dict() for item in self.prototypes],
            "feature_group_names": list(G1_BALLISTIC_HANDOFF_GROUP_NAMES),
            "feature_group_scales": [
                _PELVIS_POSE_SCALE.tolist(),
                _JOINT_POSITION_SCALE.tolist(),
                _PELVIS_VELOCITY_SCALE.tolist(),
                _JOINT_VELOCITY_SCALE.tolist(),
            ],
            "distance_metric": "equal_group_weighted_normalized_rms",
            "maximum_support_distance": self.maximum_support_distance,
            "minimum_distance_margin": self.minimum_distance_margin,
            "rejected_seeds": list(self.rejected_seeds),
            "rejected_nearest_distances": list(self.rejected_nearest_distances),
            "source_evidence_hashes": list(self.source_evidence_hashes),
            "rejected_evidence_hashes": list(self.rejected_evidence_hashes),
            "body_hash": self.body_hash,
            "implementation_hash": self.implementation_hash,
            "experiment_context_hash": self.experiment_context_hash,
            "minimum_rejected_distance": self.minimum_rejected_distance,
            "accepted": self.accepted,
            "failure_codes": list(self.failure_codes),
            "evidence_domain": "SIM_ONLY_DEVELOPMENT",
            "sealed_generalization_evidence": False,
            "direct_torque_output": False,
            "online_hot_swap_allowed": False,
            "promotion_authorized": False,
            "hardware_authorized": False,
        }
        if include_hash:
            value["memory_hash"] = self.memory_hash
        return value


def ballistic_handoff_state(
    *,
    pelvis_pose_xyz_wxyz: np.ndarray,
    joint_position: np.ndarray,
    pelvis_velocity_linear_angular: np.ndarray,
    joint_velocity: np.ndarray,
) -> G1BallisticHandoffState:
    pose = np.asarray(pelvis_pose_xyz_wxyz, dtype=np.float64)
    position = np.asarray(joint_position, dtype=np.float64)
    pelvis_velocity = np.asarray(pelvis_velocity_linear_angular, dtype=np.float64)
    velocity = np.asarray(joint_velocity, dtype=np.float64)
    if pose.shape != (7,) or position.shape != (29,):
        raise ValueError("ballistic handoff pose/position dimensions are invalid")
    if pelvis_velocity.shape != (6,) or velocity.shape != (29,):
        raise ValueError("ballistic handoff velocity dimensions are invalid")
    if not all(
        np.all(np.isfinite(value))
        for value in (pose, position, pelvis_velocity, velocity)
    ):
        raise ValueError("ballistic handoff arrays must be finite")
    w, x, y, z = (float(item) for item in pose[3:7])
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("ballistic handoff quaternion norm is zero")
    w, x, y, z = (item / norm for item in (w, x, y, z))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return G1BallisticHandoffState(
        pelvis_pose_xyz_rpy=tuple(float(item) for item in (*pose[:3], roll, pitch, yaw)),
        joint_position=tuple(float(item) for item in position),
        pelvis_velocity_linear_angular=tuple(float(item) for item in pelvis_velocity),
        joint_velocity=tuple(float(item) for item in velocity),
    )


def ballistic_handoff_distance(
    left: G1BallisticHandoffState,
    right: G1BallisticHandoffState,
) -> float:
    scales = (
        _PELVIS_POSE_SCALE,
        _JOINT_POSITION_SCALE,
        _PELVIS_VELOCITY_SCALE,
        _JOINT_VELOCITY_SCALE,
    )
    group_distances = [
        float(
            np.sqrt(
                np.mean(
                    np.square(
                        (
                            np.asarray(left_group, dtype=np.float64)
                            - np.asarray(right_group, dtype=np.float64)
                        )
                        / scale
                    )
                )
            )
        )
        for left_group, right_group, scale in zip(
            left.groups, right.groups, scales, strict=True
        )
    ]
    return float(np.sqrt(np.mean(np.square(group_distances))))


def derive_g1_ballistic_skill_memory(
    *,
    skill_evidence_paths: tuple[Path, ...],
    rejected_evidence_paths: tuple[Path, ...],
    output_path: Path,
    source_checkout: Path,
    maximum_support_distance: float = 0.35,
    minimum_distance_margin: float = 0.05,
) -> G1BallisticSkillMemory:
    if len(skill_evidence_paths) < 2:
        raise ValueError("ballistic skill memory requires at least two skill islands")
    if len(rejected_evidence_paths) < 4:
        raise ValueError("ballistic skill memory requires at least four rejected states")
    if not 0.05 <= maximum_support_distance <= 0.75:
        raise ValueError("ballistic skill support distance must be in [0.05, 0.75]")
    if not 0.0 <= minimum_distance_margin <= 0.30:
        raise ValueError("ballistic skill distance margin must be in [0, 0.30]")
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("ballistic skill memory evidence must be outside the source checkout")
    if output.exists():
        raise FileExistsError("ballistic skill memory output already exists")

    prototypes: list[G1BallisticSkillPrototype] = []
    rejected: list[tuple[int, G1BallisticHandoffState, str]] = []
    body_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    context_hashes: set[str] = set()
    seen_seeds: set[int] = set()
    for path_value, expected_skill in (
        *((path, True) for path in skill_evidence_paths),
        *((path, False) for path in rejected_evidence_paths),
    ):
        path = path_value.expanduser().resolve()
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if evidence.get("strict_replay") is not True:
            raise ValueError("ballistic skill memory requires strict replay evidence")
        trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
        if (
            not trajectory.is_file()
            or not 1 <= trajectory.stat().st_size <= _MAX_TRAJECTORY_BYTES
            or evidence.get("trajectory_hash") != _file_hash(trajectory)
        ):
            raise ValueError("ballistic skill memory trajectory binding is invalid")
        body_hashes.add(str(evidence.get("body_hash", "")))
        implementation_hashes.add(str(evidence.get("implementation_hash", "")))
        flow = dict(evidence.get("flow_config", {}))
        sonic = dict(evidence.get("sonic_runup_config", {}))
        seed = int(sonic.get("planner_seed", -1))
        if seed < 0 or seed in seen_seeds:
            raise ValueError("ballistic skill memory planner seeds must be unique")
        seen_seeds.add(seed)
        action_value = flow.get("ballistic_contact_residual_rad")
        contact_frame = int(flow.get("ballistic_contact_policy_frame", -1))
        damping = float(flow.get("post_contact_damping_scale", 1.0))
        context_hashes.add(
            ballistic_skill_experiment_context_hash(
                flow_config=flow,
                sonic_runup_config=sonic,
                runup_config=dict(evidence.get("runup_config", {})),
                goal_spec=dict(evidence.get("goal_spec", {})),
                approach_strike_candidate_hash=evidence.get(
                    "approach_strike_candidate_hash"
                ),
            )
        )
        state = _trajectory_handoff_state(trajectory)
        result = dict(evidence.get("result", {}))
        hard_safe = bool(
            result.get("finite_state") is True
            and result.get("post_kick_fall") is False
            and result.get("joint_limit_violation") is False
            and result.get("torque_limit_violation") is False
        )
        crossing = result.get("goal_crossing_xyz_m")
        crossed = result.get("goal_crossed") is True and isinstance(crossing, list)
        raw_error = result.get("goal_plane_target_error_m")
        error = (
            float(raw_error)
            if isinstance(raw_error, (int, float)) and math.isfinite(float(raw_error))
            else math.inf
        )
        height = (
            float(crossing[2])
            if crossed
            and len(crossing) == 3
            and isinstance(crossing[2], (int, float))
            and math.isfinite(float(crossing[2]))
            else 0.0
        )
        qualified = bool(
            hard_safe
            and result.get("perceptual_continuity_passed") is True
            and crossed
            and error <= _MAX_SKILL_ERROR_M
            and height >= _MIN_SKILL_CROSSING_HEIGHT_M
        )
        evidence_hash = _file_hash(path)
        if expected_skill:
            if not qualified or not isinstance(action_value, list):
                raise ValueError("ballistic skill evidence is not a qualified skill island")
            action = tuple(float(item) for item in action_value)
            prototypes.append(
                G1BallisticSkillPrototype(
                    skill_id=f"sonic-seed-{seed}",
                    planner_seed=seed,
                    state=state,
                    action_rad=action,
                    contact_policy_frame=contact_frame,
                    post_contact_damping_scale=damping,
                    goal_plane_target_error_m=error,
                    goal_crossing_height_m=height,
                    evidence_path=str(path),
                    evidence_hash=evidence_hash,
                    trajectory_hash=str(evidence["trajectory_hash"]),
                )
            )
        else:
            if qualified:
                raise ValueError("rejected ballistic state unexpectedly meets skill criteria")
            rejected.append((seed, state, evidence_hash))

    if len(body_hashes) != 1 or not _is_sha256(next(iter(body_hashes), "")):
        raise ValueError("ballistic skill memory Body hashes disagree")
    if len(implementation_hashes) != 1 or not _is_sha256(
        next(iter(implementation_hashes), "")
    ):
        raise ValueError("ballistic skill memory implementation hashes disagree")
    if len(context_hashes) != 1:
        raise ValueError("ballistic skill memory experiment contexts disagree")
    prototypes.sort(key=lambda item: item.planner_seed)
    if len({item.skill_id for item in prototypes}) != len(prototypes):
        raise ValueError("ballistic skill ids must be unique")
    if min(
        ballistic_handoff_distance(left.state, right.state)
        for index, left in enumerate(prototypes)
        for right in prototypes[index + 1 :]
    ) <= minimum_distance_margin:
        raise ValueError("ballistic skill prototypes are not distinguishable")
    rejected_distances = tuple(
        min(ballistic_handoff_distance(state, item.state) for item in prototypes)
        for _, state, _ in rejected
    )
    minimum_rejected = min(rejected_distances)
    failures: list[str] = []
    if minimum_rejected <= maximum_support_distance:
        failures.append("REJECTED_STATE_INSIDE_SUPPORT_RADIUS")
    accepted = not failures
    unsigned = {
        "schema_version": "rosclaw.growth.g1_ballistic_skill_memory.v1",
        "prototypes": [item.to_dict() for item in prototypes],
        "feature_group_names": list(G1_BALLISTIC_HANDOFF_GROUP_NAMES),
        "feature_group_scales": [
            _PELVIS_POSE_SCALE.tolist(),
            _JOINT_POSITION_SCALE.tolist(),
            _PELVIS_VELOCITY_SCALE.tolist(),
            _JOINT_VELOCITY_SCALE.tolist(),
        ],
        "distance_metric": "equal_group_weighted_normalized_rms",
        "maximum_support_distance": maximum_support_distance,
        "minimum_distance_margin": minimum_distance_margin,
        "rejected_seeds": [seed for seed, _, _ in rejected],
        "rejected_nearest_distances": list(rejected_distances),
        "source_evidence_hashes": [item.evidence_hash for item in prototypes],
        "rejected_evidence_hashes": [item[2] for item in rejected],
        "body_hash": next(iter(body_hashes)),
        "implementation_hash": next(iter(implementation_hashes)),
        "experiment_context_hash": next(iter(context_hashes)),
        "minimum_rejected_distance": minimum_rejected,
        "accepted": accepted,
        "failure_codes": failures,
        "evidence_domain": "SIM_ONLY_DEVELOPMENT",
        "sealed_generalization_evidence": False,
        "direct_torque_output": False,
        "online_hot_swap_allowed": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
    memory = _memory_from_dict(unsigned, canonical_hash(unsigned))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(memory.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return memory


def load_g1_ballistic_skill_memory(path: Path) -> G1BallisticSkillMemory:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    claimed = str(value.pop("memory_hash", ""))
    if claimed != canonical_hash(value):
        raise ValueError("ballistic skill memory hash mismatch")
    memory = _memory_from_dict(value, claimed)
    if not memory.accepted or memory.failure_codes:
        raise ValueError("ballistic skill memory was not development-accepted")
    return memory


def _memory_from_dict(value: dict[str, Any], memory_hash: str) -> G1BallisticSkillMemory:
    expected_scales = [
        _PELVIS_POSE_SCALE.tolist(),
        _JOINT_POSITION_SCALE.tolist(),
        _PELVIS_VELOCITY_SCALE.tolist(),
        _JOINT_VELOCITY_SCALE.tolist(),
    ]
    if (
        value.get("schema_version") != "rosclaw.growth.g1_ballistic_skill_memory.v1"
        or tuple(value.get("feature_group_names", ()))
        != G1_BALLISTIC_HANDOFF_GROUP_NAMES
        or value.get("feature_group_scales") != expected_scales
        or value.get("distance_metric") != "equal_group_weighted_normalized_rms"
        or value.get("evidence_domain") != "SIM_ONLY_DEVELOPMENT"
        or value.get("sealed_generalization_evidence") is not False
        or value.get("direct_torque_output") is not False
        or value.get("online_hot_swap_allowed") is not False
        or value.get("promotion_authorized") is not False
        or value.get("hardware_authorized") is not False
    ):
        raise ValueError("ballistic skill memory safety boundary is invalid")
    prototypes = tuple(_prototype_from_dict(item) for item in value["prototypes"])
    rejected_seeds = tuple(int(item) for item in value["rejected_seeds"])
    rejected_distances = tuple(
        float(item) for item in value["rejected_nearest_distances"]
    )
    source_hashes = tuple(str(item) for item in value["source_evidence_hashes"])
    rejected_hashes = tuple(str(item) for item in value["rejected_evidence_hashes"])
    raw_failures = value["failure_codes"]
    raw_accepted = value["accepted"]
    if not isinstance(raw_failures, list) or not isinstance(raw_accepted, bool):
        raise ValueError("ballistic skill memory decision types are invalid")
    failures = tuple(str(item) for item in raw_failures)
    hashes = (
        str(value["body_hash"]),
        str(value["implementation_hash"]),
        str(value["experiment_context_hash"]),
        memory_hash,
        *source_hashes,
        *rejected_hashes,
    )
    maximum_distance = float(value["maximum_support_distance"])
    minimum_margin = float(value["minimum_distance_margin"])
    minimum_rejected = float(value["minimum_rejected_distance"])
    accepted = raw_accepted
    if (
        len(prototypes) < 2
        or len({item.skill_id for item in prototypes}) != len(prototypes)
        or len({item.planner_seed for item in prototypes}) != len(prototypes)
        or len(rejected_seeds) < 4
        or len(set(rejected_seeds)) != len(rejected_seeds)
        or len(rejected_distances) != len(rejected_seeds)
        or len(source_hashes) != len(prototypes)
        or len(rejected_hashes) != len(rejected_seeds)
        or not all(_is_sha256(item) for item in hashes)
        or not 0.05 <= maximum_distance <= 0.75
        or not 0.0 <= minimum_margin <= 0.30
        or not math.isfinite(minimum_rejected)
        or any(not math.isfinite(item) or item < 0.0 for item in rejected_distances)
        or accepted == bool(failures)
    ):
        raise ValueError("ballistic skill memory geometry is invalid")
    return G1BallisticSkillMemory(
        prototypes=prototypes,
        maximum_support_distance=maximum_distance,
        minimum_distance_margin=minimum_margin,
        rejected_seeds=rejected_seeds,
        rejected_nearest_distances=rejected_distances,
        source_evidence_hashes=source_hashes,
        rejected_evidence_hashes=rejected_hashes,
        body_hash=str(value["body_hash"]),
        implementation_hash=str(value["implementation_hash"]),
        experiment_context_hash=str(value["experiment_context_hash"]),
        minimum_rejected_distance=minimum_rejected,
        accepted=accepted,
        failure_codes=failures,
        memory_hash=memory_hash,
    )


def _prototype_from_dict(value: dict[str, Any]) -> G1BallisticSkillPrototype:
    state_value = dict(value["state"])
    state = G1BallisticHandoffState(
        pelvis_pose_xyz_rpy=tuple(float(item) for item in state_value["pelvis_pose_xyz_rpy"]),
        joint_position=tuple(float(item) for item in state_value["joint_position"]),
        pelvis_velocity_linear_angular=tuple(
            float(item) for item in state_value["pelvis_velocity_linear_angular"]
        ),
        joint_velocity=tuple(float(item) for item in state_value["joint_velocity"]),
    )
    return G1BallisticSkillPrototype(
        skill_id=str(value["skill_id"]),
        planner_seed=int(value["planner_seed"]),
        state=state,
        action_rad=tuple(float(item) for item in value["action_rad"]),
        contact_policy_frame=int(value["contact_policy_frame"]),
        post_contact_damping_scale=float(value["post_contact_damping_scale"]),
        goal_plane_target_error_m=float(value["goal_plane_target_error_m"]),
        goal_crossing_height_m=float(value["goal_crossing_height_m"]),
        evidence_path=str(value["evidence_path"]),
        evidence_hash=str(value["evidence_hash"]),
        trajectory_hash=str(value["trajectory_hash"]),
    )


def _trajectory_handoff_state(path: Path) -> G1BallisticHandoffState:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "controller_mode",
            "pelvis_pose",
            "joint_position",
            "pelvis_velocity",
            "joint_velocity",
        }
        if not required.issubset(archive.files):
            raise ValueError("ballistic skill trajectory lacks full handoff state")
        mode = np.asarray(archive["controller_mode"])
        pose = np.asarray(archive["pelvis_pose"], dtype=np.float64)
        position = np.asarray(archive["joint_position"], dtype=np.float64)
        pelvis_velocity = np.asarray(archive["pelvis_velocity"], dtype=np.float64)
        velocity = np.asarray(archive["joint_velocity"], dtype=np.float64)
    if (
        mode.ndim != 1
        or pose.shape != (len(mode), 7)
        or position.shape != (len(mode), 29)
        or pelvis_velocity.shape != (len(mode), 6)
        or velocity.shape != (len(mode), 29)
    ):
        raise ValueError("ballistic skill handoff arrays have invalid dimensions")
    approach = np.flatnonzero(mode == 5)
    if approach.size == 0:
        raise ValueError("ballistic skill trajectory has no SONIC approach state")
    index = int(approach[-1])
    return ballistic_handoff_state(
        pelvis_pose_xyz_wxyz=pose[index],
        joint_position=position[index],
        pelvis_velocity_linear_angular=pelvis_velocity[index],
        joint_velocity=velocity[index],
    )


__all__ = [
    "G1BallisticHandoffState",
    "G1BallisticSkillMemory",
    "G1BallisticSkillPrototype",
    "G1BallisticSkillSelection",
    "ballistic_handoff_distance",
    "ballistic_handoff_state",
    "ballistic_skill_experiment_context_hash",
    "derive_g1_ballistic_skill_memory",
    "load_g1_ballistic_skill_memory",
]
