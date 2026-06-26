"""Contract tests: skill manager consumes EffectiveBody via BodyResolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.body.resolver import BodyResolver
from rosclaw.body.service import BodyInstanceService
from rosclaw.core.event_bus import EventBus
from rosclaw.skill_manager.executor import SkillExecutor
from rosclaw.skill_manager.registry import SkillEntry, SkillRegistry


@pytest.fixture
def linked_workspace(tmp_path: Path, monkeypatch) -> Path:
    workspace = tmp_path / ".rosclaw"
    monkeypatch.setenv("HOME", str(tmp_path))
    BodyInstanceService().create_or_init(
        robot="unitree-g1", name="g1-skill", mode="registry", update_registry=True, switch_active=True
    )
    return workspace


def _write_manifest(workspace: Path, skill_id: str, capability: str) -> None:
    skills_dir = workspace / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "skill_id": skill_id,
        "skill_version": "1.0.0",
        "schema_version": "rosclaw.skill.v1",
        "requires": {
            "capabilities": {"all_of": [capability]},
        },
    }
    (skills_dir / f"{skill_id}.skill.yaml").write_text(
        __import__("yaml").safe_dump(manifest), encoding="utf-8"
    )


def test_skill_executor_uses_effective_body_hash(linked_workspace: Path):
    resolver = BodyResolver()
    _ = resolver.resolve("rosclaw://body/current/effective")

    registry = SkillRegistry(event_bus=EventBus())
    registry.register(SkillEntry(name="walk", description="walk forward", skill_type="programmed"))

    executor = SkillExecutor(
        event_bus=EventBus(),
        registry=registry,
        body_resolver=resolver,
    )

    # Compatibility is unknown because no manifest exists -> fail-closed blocked.
    result = executor.execute("walk")
    assert result["status"] == "blocked"

    # The executor reached the same effective body through BodyResolver.
    assert executor._body_resolver is resolver  # noqa: SLF001


def test_skill_executor_blocks_unknown_compatibility(linked_workspace: Path):
    resolver = BodyResolver()

    registry = SkillRegistry(event_bus=EventBus())
    registry.register(SkillEntry(name="fly", description="fly", skill_type="programmed"))

    executor = SkillExecutor(
        event_bus=EventBus(),
        registry=registry,
        body_resolver=resolver,
    )

    result = executor.execute("fly")
    assert result["status"] == "blocked"
    assert "unknown" in result["message"].lower()


def test_skill_executor_allows_compatible_skill(linked_workspace: Path):
    _write_manifest(linked_workspace, "walk", "walk")
    resolver = BodyResolver()
    body = resolver.resolve("rosclaw://body/current/effective")

    registry = SkillRegistry(event_bus=EventBus())
    registry.register(SkillEntry(name="walk", description="walk forward", skill_type="programmed"))

    executor = SkillExecutor(
        event_bus=EventBus(),
        registry=registry,
        body_resolver=resolver,
    )

    result = executor.execute("walk")
    assert result["status"] in ("executed", "dispatched")
    assert result.get("body_check", {}).get("status") == "ok"

    # Skill compatibility result checked against the same effective body hash.
    body_check = result.get("body_check", {}).get("result", {})
    assert body_check.get("checked_against", {}).get("body_hash") == body.effective_body_hash
