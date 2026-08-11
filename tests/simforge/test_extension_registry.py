from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from rosclaw import extension_discovery
from rosclaw.sandbox.backends.base import (
    BackendCapabilities,
    CompiledScenario,
    ReplayReport,
    RolloutRequest,
    ScenarioSpec,
    TrajectorySimulationReceipt,
)
from rosclaw.simforge import (
    SIMFORGE_BACKEND_GROUP,
    SIMFORGE_TASK_GROUP,
    SimForgeExtensionRegistry,
    SimForgeTaskSpec,
)


@dataclass
class _EntryPoint:
    name: str
    group: str
    value: Any
    error: Exception | None = None

    def load(self) -> Any:
        if self.error is not None:
            raise self.error
        return self.value


class _EntryPoints(list[_EntryPoint]):
    def select(self, *, group: str) -> _EntryPoints:
        return _EntryPoints(item for item in self if item.group == group)


def _task(task_id: str, body_id: str, capability: str) -> SimForgeTaskSpec:
    return SimForgeTaskSpec(
        task_id=task_id,
        suite_id="cross_domain.v1",
        body_id=body_id,
        required_capabilities=(capability,),
        discovery_backends=("mujoco.cpu",),
        evaluation_backends=("mujoco.cpu",),
        differential_backends=(),
        scenario_distribution_ref=f"fixture://{task_id}",
        success_spec=(("task.progress", 1.0),),
        safety_spec=(("safety.violation", 0.0),),
        candidate_allowed_paths=("/controller/gain",),
    )


@dataclass
class _TaskProvider:
    provider_id: str
    specs: dict[str, SimForgeTaskSpec]

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(self.specs)

    def task_spec(self, task_id: str) -> SimForgeTaskSpec:
        return self.specs[task_id]


class _Backend:
    def __init__(self, backend_id: str) -> None:
        self.backend_id = backend_id
        self.closed = False

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name=self.backend_id, physics=True, replay=True)

    def compile(self, _scenario: ScenarioSpec) -> CompiledScenario:
        raise NotImplementedError

    def rollout(self, _request: RolloutRequest) -> TrajectorySimulationReceipt:
        raise NotImplementedError

    def replay(
        self,
        _receipt: TrajectorySimulationReceipt | dict[str, Any],
        *,
        strict: bool = True,
    ) -> ReplayReport:
        raise NotImplementedError

    def close(self) -> None:
        self.closed = True


@dataclass
class _BackendFactory:
    backend_id: str
    created: bool = False

    def create(self) -> _Backend:
        self.created = True
        return _Backend(self.backend_id)


def test_task_and_backend_discovery_is_lazy_and_cross_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    navigation = _TaskProvider(
        "navigation.tasks",
        {"navigation.path_follow": _task("navigation.path_follow", "mobile_base", "base.velocity")},
    )
    manipulation = _TaskProvider(
        "manipulation.tasks",
        {
            "manipulation.soft_grip": _task(
                "manipulation.soft_grip", "dexterous_hand", "finger.force"
            )
        },
    )
    backend_factory = _BackendFactory("mujoco.cpu")
    entries = _EntryPoints(
        [
            _EntryPoint("navigation", SIMFORGE_TASK_GROUP, navigation),
            _EntryPoint("manipulation", SIMFORGE_TASK_GROUP, manipulation),
            _EntryPoint("mujoco", SIMFORGE_BACKEND_GROUP, backend_factory),
        ]
    )
    monkeypatch.setattr(extension_discovery.importlib.metadata, "entry_points", lambda: entries)
    registry = SimForgeExtensionRegistry()

    report = registry.discover()

    assert registry.tasks.task_ids == (
        "manipulation.soft_grip",
        "navigation.path_follow",
    )
    assert registry.backends.backend_ids == ("mujoco.cpu",)
    assert backend_factory.created is False
    assert report.tasks.errors == ()
    assert registry.tasks.task_spec("manipulation.soft_grip").body_id == "dexterous_hand"

    backend = registry.backends.create("mujoco.cpu")
    assert backend_factory.created is True
    assert backend.capabilities().physics is True
    backend.close()


def test_duplicate_task_from_optional_package_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _TaskProvider(
        "first.tasks",
        {"navigation.path_follow": _task("navigation.path_follow", "base", "base.velocity")},
    )
    second = _TaskProvider(
        "second.tasks",
        {"navigation.path_follow": _task("navigation.path_follow", "base", "base.velocity")},
    )
    entries = _EntryPoints(
        [
            _EntryPoint("first", SIMFORGE_TASK_GROUP, first),
            _EntryPoint("second", SIMFORGE_TASK_GROUP, second),
        ]
    )
    monkeypatch.setattr(extension_discovery.importlib.metadata, "entry_points", lambda: entries)

    report = SimForgeExtensionRegistry().discover()

    assert report.tasks.loaded == ("first.tasks",)
    assert report.tasks.errors == (
        "second: duplicate SimForge task ids: ['navigation.path_follow']",
    )


def test_provider_cannot_return_spec_for_another_task() -> None:
    provider = _TaskProvider(
        "navigation.tasks",
        {"navigation.path_follow": _task("navigation.path_follow", "base", "base.velocity")},
    )
    registry = SimForgeExtensionRegistry()
    registry.tasks.register(provider)
    provider.specs["navigation.path_follow"] = _task(
        "navigation.other_task", "base", "base.velocity"
    )

    with pytest.raises(ValueError, match="different task_id"):
        registry.tasks.task_spec("navigation.path_follow")


def test_core_contracts_remain_available_when_no_plugins_are_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        extension_discovery.importlib.metadata,
        "entry_points",
        lambda: _EntryPoints(),
    )
    registry = SimForgeExtensionRegistry()

    report = registry.discover()
    spec = _task("core.contract_smoke", "generic_body", "sandbox.rollout")

    assert report.tasks.loaded == ()
    assert report.backends.loaded == ()
    assert registry.tasks.task_ids == ()
    assert spec.task_id == "core.contract_smoke"


def test_simforge_discovery_has_operator_recovery_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROSCLAW_DISABLE_SIMFORGE_EXTENSIONS", "1")
    monkeypatch.setattr(
        extension_discovery.importlib.metadata,
        "entry_points",
        lambda: pytest.fail("disabled discovery must not inspect packages"),
    )

    report = SimForgeExtensionRegistry().discover()

    assert report.tasks.disabled is True
    assert report.backends.disabled is True
