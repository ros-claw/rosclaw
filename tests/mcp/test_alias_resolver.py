"""Tests for Hardware MCP alias resolution."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from rosclaw.mcp.onboarding.errors import AliasResolutionError
from rosclaw.mcp.onboarding.hub_client import HubClient
from rosclaw.mcp.onboarding.resolver import CANONICAL_PREFIX, AliasResolver


def test_canonical_id_passed_through() -> None:
    resolver = AliasResolver()
    canonical = "io.rosclaw.hardware.unitree-g1"
    assert resolver.resolve(canonical) == canonical


@pytest.mark.parametrize(
    "alias, expected",
    [
        ("unitree-g1", "io.rosclaw.hardware.unitree-g1"),
        ("g1", "io.rosclaw.hardware.unitree-g1"),
        ("unitreeg1", "io.rosclaw.hardware.unitree-g1"),
        ("realsense-d455", "io.rosclaw.hardware.realsense-d455"),
        ("realsense", "io.rosclaw.hardware.realsense-d455"),
        ("d455", "io.rosclaw.hardware.realsense-d455"),
    ],
)
def test_builtin_aliases_resolve(alias: str, expected: str) -> None:
    resolver = AliasResolver()
    assert resolver.resolve(alias) == expected


def test_unknown_short_name_canonicalized() -> None:
    resolver = AliasResolver()
    assert resolver.resolve("my-widget") == f"{CANONICAL_PREFIX}my-widget"


def test_unresolvable_alias_raises() -> None:
    resolver = AliasResolver()
    with pytest.raises(AliasResolutionError):
        resolver.resolve("not a valid alias!")


def test_remote_owner_repo_resolves_via_manifest_probe() -> None:
    hub = Mock()
    hub.fetch_index.return_value = {}
    hub.fetch_manifest.return_value = SimpleNamespace(id="io.rosclaw.hub.ros-claw.g1-mcp")
    resolver = AliasResolver(hub=hub)

    assert resolver.resolve("ros-claw/g1-mcp") == "io.rosclaw.hub.ros-claw.g1-mcp"
    hub.fetch_manifest.assert_called_once_with("ros-claw/g1-mcp")


def test_hub_index_alias_match(fake_home: Any) -> None:
    hub = HubClient(home=fake_home)
    resolver = AliasResolver(hub=hub)
    # Index includes built-ins; exact id match returns canonical id.
    assert resolver.resolve("io.rosclaw.hardware.unitree-g1") == "io.rosclaw.hardware.unitree-g1"


def test_resolve_or_canonical() -> None:
    resolver = AliasResolver()
    assert resolver.resolve_or_canonical("unitree-g1") == "io.rosclaw.hardware.unitree-g1"
    assert (
        resolver.resolve_or_canonical("io.rosclaw.hardware.unitree-g1")
        == "io.rosclaw.hardware.unitree-g1"
    )


def test_case_insensitive_alias_resolution() -> None:
    resolver = AliasResolver()
    assert resolver.resolve("Unitree-G1") == "io.rosclaw.hardware.unitree-g1"
    assert resolver.resolve("D455") == "io.rosclaw.hardware.realsense-d455"


def test_whitespace_trimmed() -> None:
    resolver = AliasResolver()
    assert resolver.resolve("  g1  ") == "io.rosclaw.hardware.unitree-g1"
