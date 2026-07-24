"""ContactPush Failure-to-Success task with replayable MuJoCo evidence."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from rosclaw.simforge.dataset_snapshot import PracticeEpisodeRecord
from rosclaw.simforge.models import Partition
from rosclaw.simforge.seed_ledger import SeedLedger

CONTACT_PUSH_TASK_ID = "contact_push_v3"
CONTACT_PUSH_BODY_ID = "sim_contact_pusher_v3"
_BODY_CONTRACT = {
    "body_id": CONTACT_PUSH_BODY_ID,
    "kinematics": "one_axis_velocity_pusher",
    "end_effector": {"width_m": 0.05, "height_m": 0.10},
    "limits": {"max_velocity_mps": 0.5, "max_duration_sec": 1.5},
}
CONTACT_PUSH_BODY_HASH = (
    "sha256:"
    + hashlib.sha256(
        json.dumps(_BODY_CONTRACT, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
)


class ContactPushStatus(StrEnum):
    SUCCESS = "SUCCESS"
    OVERSHOOT = "OBJECT_OVERSHOT"
    UNDERSHOOT = "OBJECT_UNDERSHOT"
    FORCE_LIMIT = "FORCE_LIMIT_EXCEEDED"
    NO_CONTACT = "NO_CONTACT"
    DEADLINE = "DEADLINE_EXCEEDED"
    NON_FINITE = "NON_FINITE_STATE"


@dataclass(frozen=True)
class ContactPushScenario:
    scenario_id: str
    partition: Partition
    seed: int
    seed_commitment: str
    object_mass_kg: float
    floor_friction: float
    target_distance_m: float
    initial_offset_y_m: float
    control_delay_sec: float
    friction_sensor_noise: float

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("contact-push scenario id is required")
        if self.seed < 0:
            raise ValueError("scenario seed must be non-negative")
        if not self.seed_commitment.startswith("sha256:"):
            raise ValueError("scenario seed commitment is required")
        bounds = {
            "object_mass_kg": (self.object_mass_kg, 0.15, 1.2),
            "floor_friction": (self.floor_friction, 0.1, 1.2),
            "target_distance_m": (self.target_distance_m, 0.08, 0.45),
            "initial_offset_y_m": (self.initial_offset_y_m, -0.035, 0.035),
            "control_delay_sec": (self.control_delay_sec, 0.0, 0.12),
            "friction_sensor_noise": (self.friction_sensor_noise, -0.15, 0.15),
        }
        for name, (value, minimum, maximum) in bounds.items():
            if not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be in [{minimum}, {maximum}]")

    @property
    def observed_friction(self) -> float:
        return min(1.2, max(0.1, self.floor_friction + self.friction_sensor_noise))

    @property
    def scenario_commitment(self) -> str:
        return _hash_json(self.to_dict(reveal_seed=False))

    def feature_dict(self) -> dict[str, float]:
        return {
            "object_mass_kg": self.object_mass_kg,
            "estimated_friction": self.observed_friction,
            "target_distance_m": self.target_distance_m,
            "initial_offset_y_m": self.initial_offset_y_m,
            "control_delay_sec": self.control_delay_sec,
        }

    def to_dict(self, *, reveal_seed: bool | None = None) -> dict[str, Any]:
        if reveal_seed is None:
            reveal_seed = self.partition.candidate_may_view_cases
        value: dict[str, Any] = {
            "scenario_id": self.scenario_id,
            "partition": self.partition.value,
            "seed_commitment": self.seed_commitment,
            "object_mass_kg": self.object_mass_kg,
            "floor_friction": self.floor_friction,
            "target_distance_m": self.target_distance_m,
            "initial_offset_y_m": self.initial_offset_y_m,
            "control_delay_sec": self.control_delay_sec,
            "friction_sensor_noise": self.friction_sensor_noise,
        }
        if reveal_seed:
            value["seed"] = self.seed
        return value

    def to_private_dict(self) -> dict[str, Any]:
        return {**self.to_dict(reveal_seed=True), "scenario_commitment": self.scenario_commitment}

    @classmethod
    def from_private_dict(cls, value: dict[str, Any]) -> ContactPushScenario:
        return cls(
            scenario_id=str(value["scenario_id"]),
            partition=Partition(str(value["partition"])),
            seed=int(value["seed"]),
            seed_commitment=str(value["seed_commitment"]),
            object_mass_kg=float(value["object_mass_kg"]),
            floor_friction=float(value["floor_friction"]),
            target_distance_m=float(value["target_distance_m"]),
            initial_offset_y_m=float(value["initial_offset_y_m"]),
            control_delay_sec=float(value["control_delay_sec"]),
            friction_sensor_noise=float(value["friction_sensor_noise"]),
        )


@dataclass(frozen=True)
class ContactPushPolicy:
    push_velocity_mps: float
    contact_duration_sec: float
    contact_offset_y_m: float
    deceleration_fraction: float
    micro_push: bool
    policy_type: str = "parameter"

    def __post_init__(self) -> None:
        bounds = {
            "push_velocity_mps": (self.push_velocity_mps, 0.08, 0.5),
            "contact_duration_sec": (self.contact_duration_sec, 0.25, 1.5),
            "contact_offset_y_m": (self.contact_offset_y_m, -0.035, 0.035),
            "deceleration_fraction": (self.deceleration_fraction, 0.45, 1.0),
        }
        for name, (value, minimum, maximum) in bounds.items():
            if not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
        if self.policy_type not in {
            "baseline",
            "parameter",
            "trajectory",
            "skill_graph",
            "learned_policy",
        }:
            raise ValueError("unsupported contact-push policy type")

    @classmethod
    def baseline(cls) -> ContactPushPolicy:
        return cls(
            push_velocity_mps=0.40,
            contact_duration_sec=1.20,
            contact_offset_y_m=0.0,
            deceleration_fraction=1.0,
            micro_push=False,
            policy_type="baseline",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def policy_hash(self) -> str:
        return _hash_json(self.to_dict())


@dataclass(frozen=True)
class ContactPushState:
    step: int
    time_sec: float
    command_velocity_mps: float
    pusher_x_m: float
    object_x_m: float
    object_y_m: float
    object_velocity_mps: float
    contact_force_n: float


@dataclass(frozen=True)
class ContactPushResult:
    status: ContactPushStatus
    success: bool
    physics_executed: bool
    contact_observed: bool
    target_x_m: float
    final_object_x_m: float
    final_error_m: float
    peak_contact_force_n: float
    final_object_speed_mps: float
    elapsed_sec: float
    steps: int
    robustness: float
    force_limit_n: float
    target_tolerance_m: float
    trace: tuple[ContactPushState, ...]

    def summary_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "success": self.success,
            "physics_executed": self.physics_executed,
            "contact_observed": self.contact_observed,
            "target_x_m": self.target_x_m,
            "final_object_x_m": self.final_object_x_m,
            "final_error_m": self.final_error_m,
            "peak_contact_force_n": self.peak_contact_force_n,
            "final_object_speed_mps": self.final_object_speed_mps,
            "elapsed_sec": self.elapsed_sec,
            "steps": self.steps,
            "robustness": self.robustness,
            "force_limit_n": self.force_limit_n,
            "target_tolerance_m": self.target_tolerance_m,
        }


@dataclass(frozen=True)
class ContactPushEpisodeEvidence:
    episode_id: str
    practice_id: str
    scenario: ContactPushScenario
    policy: ContactPushPolicy
    result: ContactPushResult
    request_hash: str
    state_hash: str
    receipt_hash: str
    artifact_root: Path
    independently_verified: bool
    strict_replay: bool

    def to_practice_record(self) -> PracticeEpisodeRecord:
        return PracticeEpisodeRecord(
            episode_id=self.episode_id,
            practice_id=self.practice_id,
            scenario_id=self.scenario.scenario_id,
            seed_commitment=self.scenario.seed_commitment,
            body_snapshot_hash=CONTACT_PUSH_BODY_HASH,
            task_id=CONTACT_PUSH_TASK_ID,
            features=tuple(sorted(self.scenario.feature_dict().items())),
            policy=tuple(sorted(self.policy.to_dict().items())),
            labels=tuple(
                sorted(
                    {
                        "failure_type": self.result.status.value,
                        "success": self.result.success,
                        "final_error_m": self.result.final_error_m,
                        "peak_contact_force_n": self.result.peak_contact_force_n,
                        "robustness": self.result.robustness,
                    }.items()
                )
            ),
            artifact_hashes=(self.request_hash, self.state_hash, self.receipt_hash),
            complete=True,
            independently_verified=self.independently_verified,
            strict_replay=self.strict_replay,
        )


class ContactPushPhysics:
    """Deterministic MuJoCo runner and independent strict replay verifier."""

    def __init__(
        self,
        *,
        force_limit_n: float = 30.0,
        target_tolerance_m: float = 0.035,
        deadline_sec: float = 2.5,
        trace_stride: int = 5,
    ) -> None:
        values = (force_limit_n, target_tolerance_m, deadline_sec)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("contact-push task limits must be finite and positive")
        if not 1 <= trace_stride <= 1000:
            raise ValueError("trace_stride must be in [1, 1000]")
        self.force_limit_n = force_limit_n
        self.target_tolerance_m = target_tolerance_m
        self.deadline_sec = deadline_sec
        self.trace_stride = trace_stride

    def run(
        self,
        scenario: ContactPushScenario,
        policy: ContactPushPolicy,
    ) -> ContactPushResult:
        import mujoco
        import numpy as np

        model = mujoco.MjModel.from_xml_string(_model_xml(scenario, policy))
        data = mujoco.MjData(model)
        object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
        object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        pusher_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pusher_geom")
        pusher_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pusher_x")
        pusher_qpos_address = int(model.jnt_qposadr[pusher_joint])
        mujoco.mj_forward(model, data)
        target_x = scenario.target_distance_m
        peak_force = 0.0
        contact_observed = False
        trace: list[ContactPushState] = []
        timestep = float(model.opt.timestep)
        primary_end = scenario.control_delay_sec + policy.contact_duration_sec
        micro_end = min(self.deadline_sec - 0.15, primary_end + 0.50)
        total_steps = math.ceil(self.deadline_sec / timestep)
        stop_stable_steps = 0
        last_step = 0
        for step in range(total_steps):
            now = step * timestep
            object_x_before = float(data.xpos[object_body][0])
            command = _command_velocity(
                now=now,
                scenario=scenario,
                policy=policy,
                object_x=object_x_before,
                target_x=target_x,
                primary_end=primary_end,
                micro_end=micro_end,
                tolerance=self.target_tolerance_m,
            )
            data.ctrl[0] = command
            mujoco.mj_step(model, data)
            step_force = 0.0
            for contact_index in range(int(data.ncon)):
                contact = data.contact[contact_index]
                if {int(contact.geom1), int(contact.geom2)} != {object_geom, pusher_geom}:
                    continue
                contact_observed = True
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, contact_index, force)
                step_force = max(step_force, float(np.linalg.norm(force[:3])))
            peak_force = max(peak_force, step_force)
            object_x = float(data.xpos[object_body][0])
            object_y = float(data.xpos[object_body][1])
            object_velocity = float(data.cvel[object_body][3])
            if step % self.trace_stride == 0 or step == total_steps - 1:
                trace.append(
                    ContactPushState(
                        step=step,
                        time_sec=now + timestep,
                        command_velocity_mps=command,
                        pusher_x_m=float(data.qpos[pusher_qpos_address]) - 0.12,
                        object_x_m=object_x,
                        object_y_m=object_y,
                        object_velocity_mps=object_velocity,
                        contact_force_n=step_force,
                    )
                )
            finite = all(
                math.isfinite(value)
                for value in (
                    object_x,
                    object_y,
                    object_velocity,
                    peak_force,
                    float(data.qpos[pusher_qpos_address]),
                )
            )
            if not finite:
                return ContactPushResult(
                    status=ContactPushStatus.NON_FINITE,
                    success=False,
                    physics_executed=True,
                    contact_observed=contact_observed,
                    target_x_m=target_x,
                    final_object_x_m=object_x,
                    final_error_m=target_x - object_x,
                    peak_contact_force_n=peak_force,
                    final_object_speed_mps=object_velocity,
                    elapsed_sec=now + timestep,
                    steps=step + 1,
                    robustness=-1.0,
                    force_limit_n=self.force_limit_n,
                    target_tolerance_m=self.target_tolerance_m,
                    trace=tuple(trace),
                )
            if command == 0 and now > primary_end:
                stop_stable_steps = stop_stable_steps + 1 if abs(object_velocity) <= 0.01 else 0
                if stop_stable_steps >= 50 and (not policy.micro_push or now >= micro_end):
                    last_step = step
                    break
            last_step = step
        final_x = float(data.xpos[object_body][0])
        final_velocity = float(data.cvel[object_body][3])
        error = target_x - final_x
        elapsed = (last_step + 1) * timestep
        status = _status(
            contact_observed=contact_observed,
            final_error_m=error,
            tolerance_m=self.target_tolerance_m,
            peak_force_n=peak_force,
            force_limit_n=self.force_limit_n,
            elapsed_sec=elapsed,
            deadline_sec=self.deadline_sec,
        )
        success = status is ContactPushStatus.SUCCESS
        positional_margin = self.target_tolerance_m - abs(error)
        force_margin = (self.force_limit_n - peak_force) / max(self.force_limit_n, 1e-9)
        deadline_margin = (self.deadline_sec - elapsed) / self.deadline_sec
        robustness = min(positional_margin, force_margin, deadline_margin)
        return ContactPushResult(
            status=status,
            success=success,
            physics_executed=True,
            contact_observed=contact_observed,
            target_x_m=target_x,
            final_object_x_m=final_x,
            final_error_m=error,
            peak_contact_force_n=peak_force,
            final_object_speed_mps=final_velocity,
            elapsed_sec=elapsed,
            steps=last_step + 1,
            robustness=robustness,
            force_limit_n=self.force_limit_n,
            target_tolerance_m=self.target_tolerance_m,
            trace=tuple(trace),
        )

    def run_and_record(
        self,
        *,
        scenario: ContactPushScenario,
        policy: ContactPushPolicy,
        artifact_root: Path,
        source_checkout: Path,
        practice_id: str,
    ) -> ContactPushEpisodeEvidence:
        root = artifact_root.expanduser().resolve()
        checkout = source_checkout.resolve()
        if root == checkout or checkout in root.parents:
            raise ValueError("raw ContactPush evidence must be outside the source checkout")
        episode_id = _episode_id(scenario, policy)
        episode_root = root / episode_id
        episode_root.mkdir(parents=True, exist_ok=False)
        request = {
            "schema_version": "rosclaw.contact_push_request.v1",
            "episode_id": episode_id,
            "practice_id": practice_id,
            "task_id": CONTACT_PUSH_TASK_ID,
            "body_id": CONTACT_PUSH_BODY_ID,
            "body_snapshot_hash": CONTACT_PUSH_BODY_HASH,
            "scenario": scenario.to_private_dict(),
            "policy": policy.to_dict(),
            "policy_hash": policy.policy_hash,
            "task_limits": {
                "force_limit_n": self.force_limit_n,
                "target_tolerance_m": self.target_tolerance_m,
                "deadline_sec": self.deadline_sec,
                "trace_stride": self.trace_stride,
            },
        }
        request_path = episode_root / "trajectory_request.json"
        _write_json(request_path, request)
        result = self.run(scenario, policy)
        states_path = episode_root / "trajectory_states.json"
        _write_json(
            states_path,
            {
                "schema_version": "rosclaw.contact_push_states.v1",
                "episode_id": episode_id,
                "states": [asdict(item) for item in result.trace],
                "result": result.summary_dict(),
            },
        )
        request_hash = _hash_bytes(request_path.read_bytes())
        state_hash = _hash_bytes(states_path.read_bytes())
        replay = self.run(scenario, policy)
        strict_replay = _results_equal(result, replay)
        receipt = {
            "schema_version": "rosclaw.contact_push_receipt.v1",
            "episode_id": episode_id,
            "practice_id": practice_id,
            "task_id": CONTACT_PUSH_TASK_ID,
            "body_snapshot_hash": CONTACT_PUSH_BODY_HASH,
            "scenario_commitment": scenario.scenario_commitment,
            "seed_commitment": scenario.seed_commitment,
            "policy_hash": policy.policy_hash,
            "result": result.summary_dict(),
            "artifact_hashes": {
                "trajectory_request": request_hash,
                "trajectory_states": state_hash,
            },
            "independent_verifier": {
                "runner": "contact_push_mujoco_v3",
                "physics_reexecuted": True,
                "strict_replay": strict_replay,
                "replay_result_hash": _hash_json(replay.summary_dict()),
            },
            "evidence_domain": "SHADOW",
        }
        receipt_path = episode_root / "simulation_receipt.json"
        _write_json(receipt_path, receipt)
        receipt_hash = _hash_bytes(receipt_path.read_bytes())
        return ContactPushEpisodeEvidence(
            episode_id=episode_id,
            practice_id=practice_id,
            scenario=scenario,
            policy=policy,
            result=result,
            request_hash=request_hash,
            state_hash=state_hash,
            receipt_hash=receipt_hash,
            artifact_root=episode_root,
            independently_verified=True,
            strict_replay=strict_replay,
        )


def generate_contact_push_scenarios(
    *,
    ledger: SeedLedger,
    partition: Partition,
    count: int,
    root_seed: int,
) -> tuple[ContactPushScenario, ...]:
    if not 1 <= count <= 100_000:
        raise ValueError("contact-push scenario count must be in [1, 100000]")
    scenarios = []
    for index in range(count):
        record = ledger.derive(partition, index)
        rng = random.Random(record.seed ^ root_seed)
        friction = rng.uniform(0.14, 0.85)
        scenarios.append(
            ContactPushScenario(
                scenario_id=(
                    f"contact_{partition.value}_{index:05d}_"
                    f"{record.commitment.removeprefix('sha256:')[:8]}"
                ),
                partition=partition,
                seed=record.seed,
                seed_commitment=record.commitment,
                object_mass_kg=rng.uniform(0.22, 0.75),
                floor_friction=friction,
                target_distance_m=rng.uniform(0.13, 0.34),
                initial_offset_y_m=rng.uniform(-0.018, 0.018),
                control_delay_sec=rng.uniform(0.0, 0.08),
                friction_sensor_noise=rng.uniform(-0.06, 0.06),
            )
        )
    return tuple(scenarios)


def _command_velocity(
    *,
    now: float,
    scenario: ContactPushScenario,
    policy: ContactPushPolicy,
    object_x: float,
    target_x: float,
    primary_end: float,
    micro_end: float,
    tolerance: float,
) -> float:
    if now < scenario.control_delay_sec:
        return 0.0
    elapsed = now - scenario.control_delay_sec
    if now < primary_end:
        deceleration_start = policy.contact_duration_sec * policy.deceleration_fraction
        if elapsed <= deceleration_start or policy.deceleration_fraction >= 1.0:
            return policy.push_velocity_mps
        tail = policy.contact_duration_sec - deceleration_start
        progress = min(1.0, max(0.0, (elapsed - deceleration_start) / max(tail, 1e-9)))
        return policy.push_velocity_mps * max(0.12, 1.0 - progress)
    if policy.micro_push and now < micro_end and object_x < target_x - tolerance * 0.45:
        return min(0.14, policy.push_velocity_mps * 0.45)
    return 0.0


def _status(
    *,
    contact_observed: bool,
    final_error_m: float,
    tolerance_m: float,
    peak_force_n: float,
    force_limit_n: float,
    elapsed_sec: float,
    deadline_sec: float,
) -> ContactPushStatus:
    if not contact_observed:
        return ContactPushStatus.NO_CONTACT
    if peak_force_n > force_limit_n:
        return ContactPushStatus.FORCE_LIMIT
    if elapsed_sec > deadline_sec + 1e-9:
        return ContactPushStatus.DEADLINE
    if final_error_m < -tolerance_m:
        return ContactPushStatus.OVERSHOOT
    if final_error_m > tolerance_m:
        return ContactPushStatus.UNDERSHOOT
    return ContactPushStatus.SUCCESS


def _model_xml(scenario: ContactPushScenario, policy: ContactPushPolicy) -> str:
    return f"""
<mujoco model="contact_push_v3">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.05"
          friction="{scenario.floor_friction:.12g} 0.01 0.001"/>
    <body name="object" pos="0 {scenario.initial_offset_y_m:.12g} 0.04">
      <freejoint/>
      <geom name="object_geom" type="box" size="0.04 0.04 0.04"
            mass="{scenario.object_mass_kg:.12g}"
            friction="{scenario.floor_friction:.12g} 0.01 0.001"/>
    </body>
    <body name="pusher" pos="-0.12 {policy.contact_offset_y_m:.12g} 0.05">
      <joint name="pusher_x" type="slide" axis="1 0 0" range="-0.2 0.8" damping="1"/>
      <geom name="pusher_geom" type="box" size="0.025 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="push_velocity" joint="pusher_x" kv="150" ctrlrange="0 0.5"/>
  </actuator>
</mujoco>
"""


def build_contact_push_model_xml(
    scenario: ContactPushScenario,
    policy: ContactPushPolicy,
) -> str:
    """Return the canonical model for differential simulation backends."""

    return _model_xml(scenario, policy)


def _results_equal(left: ContactPushResult, right: ContactPushResult) -> bool:
    if left.status is not right.status or left.success is not right.success:
        return False
    numeric = (
        "target_x_m",
        "final_object_x_m",
        "final_error_m",
        "peak_contact_force_n",
        "final_object_speed_mps",
        "elapsed_sec",
        "robustness",
    )
    return (
        all(
            math.isclose(getattr(left, name), getattr(right, name), rel_tol=0.0, abs_tol=1e-12)
            for name in numeric
        )
        and left.steps == right.steps
    )


def _episode_id(scenario: ContactPushScenario, policy: ContactPushPolicy) -> str:
    identity = f"{scenario.scenario_commitment}\0{scenario.seed_commitment}\0{policy.policy_hash}"
    return "episode_" + hashlib.sha256(identity.encode()).hexdigest()[:24]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _hash_json(value: dict[str, Any]) -> str:
    return _hash_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )


def _hash_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


__all__ = [
    "CONTACT_PUSH_BODY_HASH",
    "CONTACT_PUSH_BODY_ID",
    "CONTACT_PUSH_TASK_ID",
    "ContactPushEpisodeEvidence",
    "ContactPushPhysics",
    "ContactPushPolicy",
    "ContactPushResult",
    "ContactPushScenario",
    "ContactPushState",
    "ContactPushStatus",
    "build_contact_push_model_xml",
    "generate_contact_push_scenarios",
]
