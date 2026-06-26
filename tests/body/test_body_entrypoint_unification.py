"""Tests that rosclaw body init / create / link-eurdf share a unified service."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.body.registry import BodyRegistryManager
from rosclaw.body.service import BodyInstanceService
from rosclaw.cli import main as rosclaw_main


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    return tmp_path / "rosclaw_workspace"


def test_service_create_or_init_generates_core_files(tmp_workspace: Path):
    service = BodyInstanceService(workspace=tmp_workspace)
    result = service.create_or_init(
        robot="unitree-g1",
        name="test-g1",
        mode="single",
        update_registry=False,
        switch_active=False,
    )

    assert result.body_id == "test-g1"
    assert result.effective_body_hash is not None
    assert (result.body_dir / "body.yaml").exists()
    assert (result.body_dir / "calibration.yaml").exists()
    assert (result.body_dir / "maintenance.log").exists()
    assert (result.body_dir / "refs" / "eurdf.lock").exists()
    assert (result.body_dir / "refs" / "effective_body.json").exists()
    assert (result.body_dir / "EMBODIMENT.md").exists()
    assert (result.body_dir / "BODY.md").exists()


def test_service_registry_mode_creates_under_bodies_dir(tmp_workspace: Path):
    service = BodyInstanceService(workspace=tmp_workspace)
    result = service.create_or_init(
        robot="unitree-g1",
        name="g1-sim",
        mode="registry",
        update_registry=True,
        switch_active=True,
    )

    assert result.body_dir == tmp_workspace / "bodies" / "g1-sim"
    assert (tmp_workspace / "body_registry.yaml").exists()
    manager = BodyRegistryManager(tmp_workspace)
    registry = manager.load()
    assert registry.current_body_id == "g1-sim"


def test_service_refuses_overwrite_without_force(tmp_workspace: Path):
    service = BodyInstanceService(workspace=tmp_workspace)
    service.create_or_init(robot="unitree-g1", name="existing", mode="single")

    with pytest.raises((ValueError, RuntimeError)):
        service.create_or_init(robot="unitree-g1", name="existing", mode="single", force=False)

    result = service.create_or_init(robot="unitree-g1", name="existing", mode="single", force=True)
    assert result.body_id == "existing"


def test_init_and_create_generate_same_core_files(tmp_workspace: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_workspace.parent))

    # init single mode
    with monkeypatch.context() as m:
        m.setattr("sys.argv", ["rosclaw", "body", "init", "--robot", "unitree-g1", "--name", "init-g1", "--workspace", str(tmp_workspace)])
        assert rosclaw_main() == 0

    init_body_dir = tmp_workspace / "body"
    assert (init_body_dir / "body.yaml").exists()
    assert (init_body_dir / "EMBODIMENT.md").exists()

    # create registry mode
    with monkeypatch.context() as m:
        m.setattr("sys.argv", ["rosclaw", "body", "create", "--robot", "unitree-g1", "--name", "create-g1", "--workspace", str(tmp_workspace)])
        assert rosclaw_main() == 0

    create_body_dir = tmp_workspace / "bodies" / "create-g1"
    assert (create_body_dir / "body.yaml").exists()
    assert (create_body_dir / "EMBODIMENT.md").exists()
    assert (create_body_dir / "calibration.yaml").exists()
    assert (create_body_dir / "refs" / "effective_body.json").exists()


def test_link_eurdf_uses_service_and_generates_files(tmp_workspace: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_workspace.parent))

    with monkeypatch.context() as m:
        m.setattr("sys.argv", ["rosclaw", "body", "link-eurdf", "unitree-g1", "--workspace", str(tmp_workspace)])
        assert rosclaw_main() == 0

    body_dir = tmp_workspace / "body"
    assert (body_dir / "body.yaml").exists()
    assert (body_dir / "EMBODIMENT.md").exists()
    assert (body_dir / "refs" / "effective_body.json").exists()


def test_legacy_body_path_resolves_to_active_body(tmp_workspace: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_workspace.parent))

    with monkeypatch.context() as m:
        m.setattr("sys.argv", ["rosclaw", "body", "init", "--robot", "unitree-g1", "--name", "legacy-g1", "--workspace", str(tmp_workspace)])
        assert rosclaw_main() == 0

    # In single mode the legacy ~/.rosclaw/body/ layout is used.
    assert (tmp_workspace / "body" / "body.yaml").exists()


def test_force_required_for_existing_body_overwrite(tmp_workspace: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_workspace.parent))

    with monkeypatch.context() as m:
        m.setattr("sys.argv", ["rosclaw", "body", "init", "--robot", "unitree-g1", "--name", "dup", "--workspace", str(tmp_workspace)])
        assert rosclaw_main() == 0

    # Without force the same init should fail.
    with monkeypatch.context() as m:
        m.setattr("sys.argv", ["rosclaw", "body", "init", "--robot", "unitree-g1", "--name", "dup", "--workspace", str(tmp_workspace)])
        assert rosclaw_main() == 1

    # With force it should succeed.
    with monkeypatch.context() as m:
        m.setattr("sys.argv", ["rosclaw", "body", "init", "--robot", "unitree-g1", "--name", "dup", "--workspace", str(tmp_workspace), "--force"])
        assert rosclaw_main() == 0
