"""Tests for initializing a ROSClaw body from a manifest-driven zoo asset."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.body.service import BodyInstanceService
from rosclaw.eurdf.zoo_client import E_URDF_ZOO_AVAILABLE

pytestmark = pytest.mark.skipif(
    not E_URDF_ZOO_AVAILABLE,
    reason="e_urdf_zoo package is not installed",
)


@pytest.fixture
def zoo_path() -> Path:
    return Path(__file__).parent.parent.parent / "e-urdf-zoo" / "robots"


@pytest.fixture
def service(tmp_path: Path, zoo_path: Path) -> BodyInstanceService:
    return BodyInstanceService(workspace=tmp_path)


def test_body_init_from_zoo_asset(service: BodyInstanceService, zoo_path: Path) -> None:
    result = service.create_or_init(
        robot="dexhands/inspire_hand/right",
        name="inspire-right-test",
        mode="single",
        update_registry=True,
        switch_active=True,
        render_agent_view=True,
        zoo_path=zoo_path,
    )
    assert result.body_id == "inspire-right-test"
    assert result.profile_id == "dexhands/inspire_hand/right"
    assert result.eurdf_uri == "rosclaw://eurdf/dexhands/inspire_hand/right@1.0.0"
    assert result.effective_body_hash is not None
    assert result.checksum

    # Verify artifacts were written.
    assert (result.workspace / "body" / "refs" / "eurdf.lock").exists()
    assert (result.workspace / "body" / "refs" / "eurdf.profile.yaml").exists()
    assert (result.workspace / "body" / "EMBODIMENT.md").exists()


def test_body_init_from_zoo_detects_slash(service: BodyInstanceService, zoo_path: Path) -> None:
    # ``from_zoo`` is auto-detected because the ID contains a slash.
    result = service.create_or_init(
        robot="dexhands/inspire_hand/right",
        mode="single",
        update_registry=False,
        switch_active=False,
        render_agent_view=False,
        zoo_path=zoo_path,
    )
    assert result.profile_id == "dexhands/inspire_hand/right"


def test_body_init_zoo_source_lock(service: BodyInstanceService, zoo_path: Path) -> None:
    service.create_or_init(
        robot="dexhands/inspire_hand/right",
        name="lock-test",
        mode="single",
        update_registry=False,
        switch_active=False,
        render_agent_view=False,
        zoo_path=zoo_path,
    )
    lock_path = service.workspace / "body" / "refs" / "eurdf.lock"
    import yaml

    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    assert lock["source"] == "zoo"
    assert lock.get("zoo_source") is True
