from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from rosclaw import extension_discovery
from rosclaw.growth import (
    GROWTH_ADAPTER_GROUP,
    GROWTH_LEARNER_GROUP,
    GrowthExtensionRegistry,
    LearnerDescriptor,
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


@dataclass
class _Adapter:
    adapter_id: str
    skill_ids: tuple[str, ...]

    def normalize_experience(self, _payload: Any) -> Any:
        raise NotImplementedError

    def diagnose(self, _segment: Any) -> Any:
        raise NotImplementedError


def _learner(learner_id: str) -> LearnerDescriptor:
    return LearnerDescriptor(
        learner_id=learner_id,
        required_field_ids=("state.self", "action.executed", "outcome.cost"),
        artifact_kinds=("residual.policy",),
        online_rollout_required=False,
        consumes_cost_vector=True,
    )


def test_registry_discovers_two_domains_without_runtime_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "navigation",
                GROWTH_ADAPTER_GROUP,
                _Adapter("navigation.path_follow", ("navigation.follow_path",)),
            ),
            _EntryPoint(
                "manipulation",
                GROWTH_ADAPTER_GROUP,
                _Adapter("manipulation.force_control", ("manipulation.soft_grip",)),
            ),
            _EntryPoint("system-identification", GROWTH_LEARNER_GROUP, _learner("sysid")),
        ]
    )
    monkeypatch.setattr(extension_discovery.importlib.metadata, "entry_points", lambda: entries)
    registry = GrowthExtensionRegistry()

    report = registry.discover()

    assert registry.adapter_ids == (
        "manipulation.force_control",
        "navigation.path_follow",
    )
    assert registry.learner_ids == ("sysid",)
    assert report.adapters.errors == ()
    assert report.learners.errors == ()


def test_broken_or_duplicate_extension_isolated_from_healthy_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "broken",
                GROWTH_ADAPTER_GROUP,
                None,
                RuntimeError("broken package"),
            ),
            _EntryPoint(
                "first",
                GROWTH_ADAPTER_GROUP,
                _Adapter("navigation.path_follow", ("navigation.follow_path",)),
            ),
            _EntryPoint(
                "second",
                GROWTH_ADAPTER_GROUP,
                _Adapter("navigation.path_follow", ("navigation.follow_path",)),
            ),
        ]
    )
    monkeypatch.setattr(extension_discovery.importlib.metadata, "entry_points", lambda: entries)
    registry = GrowthExtensionRegistry()

    report = registry.discover()

    assert registry.adapter_ids == ("navigation.path_follow",)
    assert report.adapters.loaded == ("navigation.path_follow",)
    assert report.adapters.errors == (
        "broken: broken package",
        "second: duplicate growth adapter: navigation.path_follow",
    )


def test_growth_extension_discovery_has_operator_recovery_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROSCLAW_DISABLE_GROWTH_EXTENSIONS", "true")
    monkeypatch.setattr(
        extension_discovery.importlib.metadata,
        "entry_points",
        lambda: pytest.fail("disabled discovery must not inspect packages"),
    )

    report = GrowthExtensionRegistry().discover()

    assert report.adapters.disabled is True
    assert report.learners.disabled is True


def test_registry_rejects_objects_that_do_not_satisfy_contracts() -> None:
    registry = GrowthExtensionRegistry()

    with pytest.raises(TypeError, match="GrowthAdapter protocol"):
        registry.register_adapter(object())
    with pytest.raises(TypeError, match="LearnerDescriptor"):
        registry.register_learner(object())
