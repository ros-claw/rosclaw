"""Tests for the multi-body registry and body-aware resolver."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from rosclaw.body.registry import BodyRegistryError, BodyRegistryManager
from rosclaw.body.resolver import BodyResolver
from rosclaw.body.schema import BodyRegistry as BodyRegistrySchema
from rosclaw.body.schema import BodyRegistryEntry
from rosclaw.cli import main as rosclaw_main


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


def _run(*argv: str) -> int:
    with patch.object(sys, "argv", ["rosclaw", *argv]):
        return rosclaw_main()


def test_legacy_single_body_resolver(tmp_path: Path) -> None:
    """A pre-existing flat body/ directory is used as the implicit default body."""
    body_dir = tmp_path / "body"
    body_dir.mkdir()
    (body_dir / "body.yaml").write_text(
        yaml.safe_dump(
            {
                "body_instance": {"id": "legacy-01"},
                "model_ref": {"profile_id": "unitree-g1"},
            }
        ),
        encoding="utf-8",
    )

    resolver = BodyResolver(tmp_path)
    assert resolver.body_id == "default"
    assert resolver.body_dir == tmp_path / "body"
    assert resolver.is_legacy_single_body


def test_legacy_migration_persists_registry(tmp_path: Path) -> None:
    """Migrating a legacy workspace writes body_registry.yaml."""
    body_dir = tmp_path / "body"
    body_dir.mkdir()
    (body_dir / "body.yaml").write_text(
        yaml.safe_dump(
            {
                "body_instance": {"id": "legacy-01"},
                "model_ref": {"profile_id": "unitree-g1"},
            }
        ),
        encoding="utf-8",
    )

    registry = BodyRegistryManager(tmp_path)
    entry = registry.migrate_legacy_body()
    assert entry is not None
    assert entry.body_id == "default"
    assert entry.path == "body"
    assert (tmp_path / "body_registry.yaml").exists()


def test_create_body_via_cli() -> None:
    """body create registers a new body under bodies/<id>/."""
    assert _run("body", "create", "--robot", "unitree-g1", "--name", "g1-sim") == 0

    ws = Path.home() / ".rosclaw"
    assert (ws / "bodies" / "g1-sim" / "body.yaml").exists()
    assert (ws / "bodies" / "g1-sim" / "EMBODIMENT.md").exists()

    registry = BodyRegistryManager(ws)
    assert registry.has_body("g1-sim")
    entry = registry.get_body("g1-sim")
    assert entry.profile_id == "unitree-g1"
    assert entry.path == "bodies/g1-sim"


def test_list_and_switch_bodies() -> None:
    """list shows all bodies; switch updates current_body_id."""
    assert _run("body", "create", "--robot", "unitree-g1", "--name", "g1-sim") == 0
    assert _run("body", "create", "--robot", "unitree-g1", "--name", "g1-real") == 0

    ws = Path.home() / ".rosclaw"
    registry = BodyRegistryManager(ws)
    bodies = registry.list_bodies()
    assert {b.body_id for b in bodies} == {"g1-sim", "g1-real"}
    # After creation the most recently created body is current.
    assert registry.get_current_body_id() == "g1-real"

    assert _run("body", "switch", "g1-sim") == 0
    assert BodyRegistryManager(ws).get_current_body_id() == "g1-sim"


def test_per_body_isolation() -> None:
    """State changes on one body do not leak to another body."""
    assert _run("body", "create", "--robot", "unitree-g1", "--name", "g1-sim") == 0
    assert _run("body", "create", "--robot", "unitree-g1", "--name", "g1-real") == 0

    ws = Path.home() / ".rosclaw"

    # Add a fault to g1-sim only.
    assert (
        _run(
            "body",
            "fault",
            "add",
            "--body",
            "g1-sim",
            "--component",
            "left_knee",
            "--severity",
            "high",
            "--summary",
            "overheating",
        )
        == 0
    )

    sim_resolver = BodyResolver(ws, body_id="g1-sim")
    real_resolver = BodyResolver(ws, body_id="g1-real")

    sim_body = sim_resolver.get_effective_body()
    real_body = real_resolver.get_effective_body()

    sim_faults = [f for f in sim_body.known_faults if f.get("status") == "open"]
    real_faults = [f for f in real_body.known_faults if f.get("status") == "open"]

    assert len(sim_faults) == 1
    assert len(real_faults) == 0


def test_remove_body_archives_data() -> None:
    """remove --archive moves body data to bodies/_archive/."""
    assert _run("body", "create", "--robot", "unitree-g1", "--name", "g1-sim") == 0
    assert _run("body", "create", "--robot", "unitree-g1", "--name", "g1-real") == 0

    ws = Path.home() / ".rosclaw"
    assert (ws / "bodies" / "g1-sim").exists()

    assert _run("body", "remove", "g1-sim", "--archive") == 0

    registry = BodyRegistryManager(ws)
    assert not registry.has_body("g1-sim")
    assert not (ws / "bodies" / "g1-sim").exists()
    archive_dirs = list((ws / "bodies" / "_archive").iterdir())
    assert len(archive_dirs) == 1
    assert archive_dirs[0].name.startswith("g1-sim-")


def test_remove_body_without_archive() -> None:
    """remove without --archive deletes body data."""
    assert _run("body", "create", "--robot", "unitree-g1", "--name", "g1-sim") == 0

    ws = Path.home() / ".rosclaw"
    assert _run("body", "remove", "g1-sim") == 0

    assert not BodyRegistryManager(ws).has_body("g1-sim")
    assert not (ws / "bodies" / "g1-sim").exists()
    assert not (ws / "bodies" / "_archive").exists()


def test_registry_schema_roundtrip() -> None:
    """Registry YAML can be saved and loaded with defaults for missing fields."""
    entry = BodyRegistryEntry(
        body_id="g1-real",
        nickname="G1 Real",
        profile_id="unitree-g1",
        path="bodies/g1-real",
    )
    schema = BodyRegistrySchema(current_body_id="g1-real", bodies={"g1-real": entry})

    data = schema.to_dict()
    loaded = BodyRegistrySchema.from_dict(data)

    assert loaded.current_body_id == "g1-real"
    assert "g1-real" in loaded.bodies
    assert loaded.bodies["g1-real"].nickname == "G1 Real"
    assert loaded.schema == "rosclaw.body_registry.v1"


def test_duplicate_body_id_is_rejected() -> None:
    """Creating a body with a duplicate ID fails unless --force is used."""
    assert _run("body", "create", "--robot", "unitree-g1", "--name", "g1-sim") == 0
    assert _run("body", "create", "--robot", "unitree-g1", "--name", "g1-sim") == 1
    assert _run("body", "create", "--robot", "unitree-g1", "--name", "g1-sim", "--force") == 0


def test_switch_to_unknown_body_fails() -> None:
    """Switching to a non-existent body returns an error."""
    assert _run("body", "switch", "no-such-body") == 1


def test_list_workspace_bodies_classmethod() -> None:
    """BodyResolver.list_workspace_bodies returns registry entries."""
    assert _run("body", "create", "--robot", "unitree-g1", "--name", "alpha") == 0
    assert _run("body", "create", "--robot", "unitree-g1", "--name", "beta") == 0

    ws = Path.home() / ".rosclaw"
    entries = BodyResolver.list_workspace_bodies(ws)
    assert {e.body_id for e in entries} == {"alpha", "beta"}


def test_remove_last_body_resets_current() -> None:
    """Removing the only body resets current_body_id to the safe default."""
    assert _run("body", "create", "--robot", "unitree-g1", "--name", "g1-sim") == 0

    ws = Path.home() / ".rosclaw"
    registry = BodyRegistryManager(ws)
    assert registry.get_current_body_id() == "g1-sim"

    assert _run("body", "remove", "g1-sim") == 0

    registry = BodyRegistryManager(ws)
    assert registry.get_current_body_id() == "default"
    assert registry.list_bodies() == []


def test_archive_collision_counter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Removing the same body ID twice with the same timestamp appends a counter."""
    registry = BodyRegistryManager(tmp_path)
    registry.create_body("g1", "unitree-g1")

    archive_root = tmp_path / "bodies" / "_archive"
    fixed = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

    with patch("rosclaw.body.registry.datetime") as mock_dt:
        mock_dt.now.return_value = fixed
        mock_dt.UTC = UTC
        registry.remove_body("g1", archive=True)
        registry.create_body("g1", "unitree-g1", force=True)
        registry.remove_body("g1", archive=True)

    archive_dirs = sorted([d.name for d in archive_root.iterdir() if d.is_dir()])
    assert archive_dirs == ["g1-20240101-120000", "g1-20240101-120000_1"]


def test_invalid_body_id_rejected_cli() -> None:
    """Creating a body with an invalid ID fails at the CLI."""
    assert _run("body", "create", "--robot", "unitree-g1", "--name", "bad id!") == 1


def test_invalid_body_id_rejected_direct(tmp_path: Path) -> None:
    """Creating a body with an invalid ID raises BodyRegistryError directly."""
    registry = BodyRegistryManager(tmp_path)
    with pytest.raises(BodyRegistryError):
        registry.create_body(body_id="bad id!", profile_id="unitree-g1")


def test_remove_unknown_body_fails_direct(tmp_path: Path) -> None:
    """Removing a body that does not exist raises BodyRegistryError."""
    registry = BodyRegistryManager(tmp_path)
    with pytest.raises(BodyRegistryError, match="Body not found"):
        registry.remove_body("no-such-body")
