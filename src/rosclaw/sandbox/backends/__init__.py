"""Truthful, pluggable sandbox simulation backends."""

from rosclaw.sandbox.backends.base import (
    BackendCapabilities,
    CompiledScenario,
    ReplayReport,
    RolloutRequest,
    ScenarioSpec,
    TrajectorySimulationReceipt,
)
from rosclaw.sandbox.backends.fixture import FixtureBackend
from rosclaw.sandbox.backends.mujoco_cpu import MujocoCpuBackend
from rosclaw.sandbox.backends.registry import SandboxBackendRegistry

__all__ = [
    "BackendCapabilities",
    "CompiledScenario",
    "FixtureBackend",
    "MujocoCpuBackend",
    "ReplayReport",
    "RolloutRequest",
    "SandboxBackendRegistry",
    "ScenarioSpec",
    "TrajectorySimulationReceipt",
]
