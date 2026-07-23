"""Explicit fixture backend; never produces promotion evidence."""

from __future__ import annotations

from typing import Any

from rosclaw.sandbox.backends.base import (
    BackendCapabilities,
    CompiledScenario,
    ReplayReport,
    RolloutRequest,
    ScenarioSpec,
    TrajectorySimulationReceipt,
)
from rosclaw.sandbox.backends.fingerprints import canonical_hash


class FixtureBackend:
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name="fixture", physics=False, supported_tasks=())

    def compile(self, scenario: ScenarioSpec) -> CompiledScenario:
        return CompiledScenario(
            spec=scenario,
            backend_fingerprint=canonical_hash({"backend": "fixture"}),
            world_asset_hash="",
        )

    def rollout(self, request: RolloutRequest) -> TrajectorySimulationReceipt:
        return TrajectorySimulationReceipt(
            scenario_id=request.scenario.scenario_id,
            backend={"name": "fixture"},
            seed=request.scenario.seed,
            body_snapshot_hash=request.scenario.body_snapshot_hash,
            model_hash=request.scenario.model_hash,
            world_asset_hash="",
            action_hash=canonical_hash(request.trajectory),
            scenario_hash=canonical_hash(request.scenario.__dict__),
            is_safe=False,
            physics_executed=False,
            reason="fixture_backend_does_not_execute_physics",
            violations=["PHYSICS_NOT_EXECUTED"],
            evidence_domain="FIXTURE",
        )

    def replay(
        self, receipt: TrajectorySimulationReceipt | dict[str, Any], *, strict: bool = True
    ) -> ReplayReport:
        del receipt, strict
        return ReplayReport(
            verified=False,
            environment_match=False,
            hashes_verified=False,
            deterministic_label=False,
            final_qpos_max_abs_error=None,
            reason="fixture_receipts_are_not_physics_replay_evidence",
        )

    def close(self) -> None:
        return None


__all__ = ["FixtureBackend"]
