"""Backend-neutral contracts for Sandbox 2.0."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    physics: bool
    batched: bool = False
    replay: bool = False
    contacts: bool = False
    actuator_forces: bool = False
    supported_tasks: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    robot_id: str
    world_id: str
    body_snapshot_hash: str
    model_hash: str
    seed: int = 0
    schema_version: str = "rosclaw.scenario.v1"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompiledScenario:
    spec: ScenarioSpec
    backend_fingerprint: str
    world_asset_hash: str


@dataclass(frozen=True)
class RolloutRequest:
    scenario: ScenarioSpec
    trajectory: list[list[float]]
    control_dt_sec: float | None = None
    max_joint_delta_rad: float = 0.005
    max_joint_velocity_radps: float = 3.15
    max_final_tracking_error_rad: float = 0.25
    settle_steps: int = 100
    artifact_dir: Path | None = None


@dataclass
class TrajectorySimulationReceipt:
    scenario_id: str
    backend: dict[str, Any]
    seed: int
    body_snapshot_hash: str
    model_hash: str
    world_asset_hash: str
    action_hash: str
    scenario_hash: str
    is_safe: bool
    physics_executed: bool
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    collision_pairs: list[list[str]] = field(default_factory=list)
    final_qpos: list[float] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    request: dict[str, Any] = field(default_factory=dict)
    evidence_domain: str = "SIMULATION"
    schema_version: str = "rosclaw.simulation_receipt.v1"

    @property
    def valid_for_promotion(self) -> bool:
        return bool(
            self.physics_executed
            and self.evidence_domain == "SIMULATION"
            and self.body_snapshot_hash
            and self.model_hash
            and self.action_hash
            and self.artifact_hashes
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["valid_for_promotion"] = self.valid_for_promotion
        return value


@dataclass(frozen=True)
class ReplayReport:
    verified: bool
    environment_match: bool
    hashes_verified: bool
    deterministic_label: bool
    final_qpos_max_abs_error: float | None
    reason: str
    mismatches: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class SandboxBackend(Protocol):
    def capabilities(self) -> BackendCapabilities: ...

    def compile(self, scenario: ScenarioSpec) -> CompiledScenario: ...

    def rollout(self, request: RolloutRequest) -> TrajectorySimulationReceipt: ...

    def replay(
        self, receipt: TrajectorySimulationReceipt | dict[str, Any], *, strict: bool = True
    ) -> ReplayReport: ...

    def close(self) -> None: ...


__all__ = [
    "BackendCapabilities",
    "CompiledScenario",
    "ReplayReport",
    "RolloutRequest",
    "SandboxBackend",
    "ScenarioSpec",
    "TrajectorySimulationReceipt",
]
