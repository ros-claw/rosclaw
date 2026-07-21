from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from rosclaw.robot_pack import instance as instance_module
from rosclaw.robot_pack.catalog import RobotPackCatalog
from rosclaw.robot_pack.schema import RobotPackManifest
from rosclaw.robot_pack.store import RobotPackStore


@pytest.fixture(autouse=True)
def synthetic_adapter_source_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model a locked adapter for unit records without weakening production code."""

    verify_real_source = instance_module._adapter_source_matches_lock

    def verify(
        record: Any,
        expected_source: str,
        expected_revision: str | None,
        home: Path,
    ) -> bool:
        if record.artifact_type == "test":
            return bool(
                expected_revision
                and record.extra.get("repo_commit") == expected_revision
            )
        return verify_real_source(record, expected_source, expected_revision, home)

    monkeypatch.setattr(instance_module, "_adapter_source_matches_lock", verify)


@pytest.fixture
def builtin_pack_root() -> Path:
    return RobotPackCatalog().resolve("realsense").root


@pytest.fixture
def copied_pack(tmp_path: Path, builtin_pack_root: Path) -> Path:
    destination = tmp_path / "pack"
    shutil.copytree(builtin_pack_root, destination)
    return destination


@pytest.fixture
def installed_pack(tmp_path: Path) -> tuple[Path, RobotPackStore, RobotPackManifest]:
    home = tmp_path / "home"
    store = RobotPackStore(home)
    record = store.install("realsense")
    manifest = RobotPackManifest.from_path(Path(record.path) / "robot-pack.yaml")
    return home, store, manifest
