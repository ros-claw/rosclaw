"""Tests for EffectiveBodyCompiler and BodyResolver."""

import sys
from unittest.mock import patch

import pytest

from rosclaw.body.resolver import BodyResolver
from rosclaw.body.schema import BodyYaml
from rosclaw.cli import main as rosclaw_main


@pytest.fixture
def linked_body(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with patch.object(sys, "argv", ["rosclaw", "body", "link-eurdf", "unitree-g1"]):
        assert rosclaw_main() == 0
    yield tmp_path


def test_compiler_produces_hash(linked_body):
    resolver = BodyResolver()
    effective = resolver.get_effective_body()
    assert effective.effective_body_hash
    assert effective.body_instance_id
    assert "head_camera" in effective.sensors


def test_resolver_uri_resolution(linked_body):
    resolver = BodyResolver()
    body = resolver.resolve("rosclaw://body/current")
    assert isinstance(body, BodyYaml)
    effective = resolver.resolve("rosclaw://body/current/effective")
    assert effective.effective_body_hash


def test_capability_derivation_with_disabled_camera(linked_body):
    resolver = BodyResolver()
    with patch.object(sys, "argv", [
        "rosclaw", "body", "update-state",
        "--set", "installed_components.sensors.head_camera.status=unavailable",
        "--reason", "test",
    ]):
        assert rosclaw_main() == 0
    effective = resolver.get_effective_body()
    assert "visual_navigation" in effective.capabilities["blocked"]


def test_cross_module_api(linked_body):
    resolver = BodyResolver()
    assert resolver.has_sensor("head_camera")
    sensor = resolver.get_sensor("head_camera")
    assert sensor["status"] == "available"
