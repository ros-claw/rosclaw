"""Tests for MCP permission store and effective permission computation."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.mcp.onboarding.permissions import PermissionState, PermissionStore
from rosclaw.mcp.onboarding.schema import PermissionDecl, Permissions


@pytest.fixture
def store(fake_home: Path) -> PermissionStore:
    return PermissionStore(home=fake_home)


@pytest.fixture
def sample_permissions() -> Permissions:
    return Permissions(
        required=[
            PermissionDecl(id="mcp:tools:read", level="safe"),
            PermissionDecl(id="mcp:prompts:read", level="guarded"),
            PermissionDecl(id="mcp:roots:list", level="dangerous"),
            PermissionDecl(id="mcp:experimental", level="forbidden_by_default"),
        ],
        optional=[
            PermissionDecl(id="mcp:resources:read", level="safe"),
        ],
    )


def test_grant_and_deny(store: PermissionStore) -> None:
    store.grant("unitree-g1", "mcp:tools:read")
    assert store.is_granted("unitree-g1", "mcp:tools:read")

    store.deny("unitree-g1", "mcp:tools:read")
    assert not store.is_granted("unitree-g1", "mcp:tools:read")
    assert store.is_denied("unitree-g1", "mcp:tools:read")


def test_compute_effective_auto_grants_safe(
    store: PermissionStore,
    sample_permissions: Permissions,
) -> None:
    state = store.compute_effective("unitree-g1", sample_permissions)
    assert "mcp:tools:read" in state.granted
    assert "mcp:resources:read" in state.granted
    assert "mcp:prompts:read" in state.pending
    assert "mcp:roots:list" in state.pending
    assert "mcp:experimental" in state.denied


def test_compute_effective_allow_dangerous(
    store: PermissionStore,
    sample_permissions: Permissions,
) -> None:
    state = store.compute_effective("unitree-g1", sample_permissions, allow_dangerous=True)
    assert "mcp:roots:list" in state.granted


def test_apply_effective_persists(
    store: PermissionStore,
    sample_permissions: Permissions,
) -> None:
    state = store.apply_effective("unitree-g1", sample_permissions, allow_dangerous=False)
    loaded = store.get("unitree-g1")
    assert loaded.granted == state.granted
    assert loaded.denied == state.denied
    assert loaded.pending == state.pending


def test_stored_grants_preserved(
    store: PermissionStore,
    sample_permissions: Permissions,
) -> None:
    store.grant("unitree-g1", "mcp:prompts:read")
    state = store.compute_effective("unitree-g1", sample_permissions)
    assert "mcp:prompts:read" in state.granted
    assert "mcp:tools:read" in state.granted


def test_list_required_ungranted(
    store: PermissionStore,
    sample_permissions: Permissions,
) -> None:
    missing = store.list_required_ungranted("unitree-g1", sample_permissions)
    assert "mcp:tools:read" not in missing
    assert "mcp:prompts:read" in missing


def test_permission_state_from_dict() -> None:
    state = PermissionState.from_dict({"granted": ["a"], "denied": ["b"], "pending": ["c"]})
    assert state.granted == ["a"]
    assert state.denied == ["b"]
    assert state.pending == ["c"]


def test_load_missing_returns_empty(store: PermissionStore) -> None:
    assert store.load() == {}
