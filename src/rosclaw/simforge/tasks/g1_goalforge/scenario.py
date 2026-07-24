"""Partition-safe scenarios for G1 GoalForge."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any

from rosclaw.simforge.models import Partition
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.g1_goalforge.concepts import GOALFORGE_TASK_ID, hash_json


@dataclass(frozen=True)
class GoalForgeScenario:
    scenario_id: str
    partition: Partition
    seed: int
    seed_commitment: str
    generation: int
    ball_x_m: float
    ball_y_m: float
    ball_velocity_x_mps: float
    ball_velocity_y_mps: float
    target_y_m: float
    target_z_m: float
    ball_mass_kg: float
    ball_ground_friction: float
    restitution: float
    support_ground_friction: float
    control_latency_ms: float
    observation_noise_m: float
    joint_zero_bias_rad: float
    disturbance_n: float
    reachable: bool = True

    def __post_init__(self) -> None:
        if not self.scenario_id or self.seed < 0:
            raise ValueError("GoalForge scenario identity is invalid")
        if not self.seed_commitment.startswith("sha256:"):
            raise ValueError("GoalForge scenario requires a seed commitment")
        if not 0 <= self.generation <= 10:
            raise ValueError("GoalForge generation must be in [0, 10]")
        bounds = {
            "ball_x_m": (self.ball_x_m, 0.75, 1.25),
            "ball_y_m": (self.ball_y_m, -0.30, 0.30),
            "ball_velocity_x_mps": (self.ball_velocity_x_mps, -0.60, 0.10),
            "ball_velocity_y_mps": (self.ball_velocity_y_mps, -0.20, 0.20),
            "target_y_m": (self.target_y_m, -1.20, 1.20),
            "target_z_m": (self.target_z_m, 0.11, 1.20),
            "ball_mass_kg": (self.ball_mass_kg, 0.36, 0.48),
            "ball_ground_friction": (self.ball_ground_friction, 0.03, 0.35),
            "restitution": (self.restitution, 0.20, 0.90),
            "support_ground_friction": (self.support_ground_friction, 0.45, 1.25),
            "control_latency_ms": (self.control_latency_ms, 0.0, 80.0),
            "observation_noise_m": (self.observation_noise_m, 0.0, 0.08),
            "joint_zero_bias_rad": (self.joint_zero_bias_rad, -0.04, 0.04),
            "disturbance_n": (self.disturbance_n, 0.0, 80.0),
        }
        for name, (value, minimum, maximum) in bounds.items():
            if not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be in [{minimum}, {maximum}]")

    @property
    def scenario_commitment(self) -> str:
        return hash_json(self.to_private_dict(include_commitment=False))

    @property
    def target_zone(self) -> str:
        column = (
            "left" if self.target_y_m > 0.35 else "right" if self.target_y_m < -0.35 else "center"
        )
        row = "high" if self.target_z_m > 0.75 else "middle" if self.target_z_m > 0.35 else "low"
        return f"{column}_{row}"

    def observed_context(self) -> dict[str, float]:
        """Candidate-visible context; hidden physical truth is represented by beliefs."""

        rng = random.Random(self.seed ^ 0x514F414C)
        noise = self.observation_noise_m
        return {
            "ball_x": self.ball_x_m + rng.uniform(-noise, noise),
            "ball_y": self.ball_y_m + rng.uniform(-noise, noise),
            "ball_vx": self.ball_velocity_x_mps,
            "ball_vy": self.ball_velocity_y_mps,
            "target_y": self.target_y_m,
            "target_z": self.target_z_m,
            "support_friction_belief": 0.85,
            "ball_mass_belief": 0.42,
            "control_latency_belief_ms": 20.0,
            "body_calibration_state": abs(self.joint_zero_bias_rad),
        }

    def to_dict(self, *, reveal_hidden: bool | None = None) -> dict[str, Any]:
        if reveal_hidden is None:
            reveal_hidden = self.partition.candidate_may_view_cases
        result: dict[str, Any] = {
            "scenario_id": self.scenario_id,
            "partition": self.partition.value,
            "seed_commitment": self.seed_commitment,
            "generation": self.generation,
            "target_zone": self.target_zone,
            "observed_context": self.observed_context(),
            "reachable": self.reachable,
        }
        if reveal_hidden:
            result.update(asdict(self))
            result["partition"] = self.partition.value
        return result

    def to_private_dict(self, *, include_commitment: bool = True) -> dict[str, Any]:
        value = asdict(self)
        value["partition"] = self.partition.value
        if include_commitment:
            value["scenario_commitment"] = hash_json(
                {key: item for key, item in value.items() if key != "scenario_commitment"}
            )
        return value


def generate_goalforge_scenarios(
    *,
    ledger: SeedLedger,
    partition: Partition,
    count: int,
    generation: int,
) -> tuple[GoalForgeScenario, ...]:
    records = ledger.allocate(partition, count)
    scenarios = []
    for index, record in enumerate(records):
        rng = random.Random(record.seed)
        moving = generation >= 9
        scenarios.append(
            GoalForgeScenario(
                scenario_id=f"g1_goalforge_{partition.value}_{index:05d}",
                partition=partition,
                seed=record.seed,
                seed_commitment=record.commitment,
                generation=generation,
                ball_x_m=(1.0 if generation == 0 else rng.uniform(0.86, 1.14)),
                ball_y_m=(0.0 if generation == 0 else rng.uniform(-0.20, 0.20)),
                ball_velocity_x_mps=(rng.uniform(-0.45, -0.08) if moving else 0.0),
                ball_velocity_y_mps=(rng.uniform(-0.12, 0.12) if moving else 0.0),
                target_y_m=(0.0 if generation < 2 else rng.choice((-0.75, 0.0, 0.75))),
                target_z_m=(0.20 if generation < 2 else rng.choice((0.20, 0.55, 0.90))),
                ball_mass_kg=(0.41 if generation < 3 else rng.uniform(0.37, 0.47)),
                ball_ground_friction=(0.05 if generation < 3 else rng.uniform(0.03, 0.30)),
                restitution=(0.55 if generation < 3 else rng.uniform(0.30, 0.85)),
                support_ground_friction=(1.0 if generation < 4 else rng.uniform(0.50, 1.20)),
                control_latency_ms=(0.0 if generation < 6 else rng.uniform(0.0, 70.0)),
                observation_noise_m=(0.0 if generation < 8 else rng.uniform(0.0, 0.06)),
                joint_zero_bias_rad=(0.0 if generation < 5 else rng.uniform(-0.03, 0.03)),
                disturbance_n=(0.0 if generation < 7 else rng.uniform(0.0, 65.0)),
            )
        )
    ledger.assert_disjoint()
    return tuple(scenarios)


def default_goalforge_ledger(secret: bytes) -> SeedLedger:
    return SeedLedger(task_id=GOALFORGE_TASK_ID, secret=secret)


__all__ = [
    "GoalForgeScenario",
    "default_goalforge_ledger",
    "generate_goalforge_scenarios",
]
