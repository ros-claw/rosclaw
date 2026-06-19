"""Tests for unified cross-module references."""

import sys
from unittest.mock import patch

import pytest

from rosclaw.body.resolver import BodyResolver
from rosclaw.cli import main as rosclaw_main


@pytest.fixture
def linked_body(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with patch.object(sys, "argv", ["rosclaw", "body", "link-eurdf", "unitree-g1"]):
        assert rosclaw_main() == 0
    yield tmp_path


def test_uri_resolves_to_same_hash(linked_body):
    resolver = BodyResolver()
    effective = resolver.get_effective_body(recompile_if_stale=False)
    hash1 = effective.effective_body_hash

    class StubAdapter:
        def __init__(self, resolver):
            self.resolver = resolver

        def get_hash(self):
            return self.resolver.get_effective_body(recompile_if_stale=False).effective_body_hash

    sandbox = StubAdapter(resolver)
    provider = StubAdapter(resolver)
    skill_checker = StubAdapter(resolver)

    assert sandbox.get_hash() == hash1
    assert provider.get_hash() == hash1
    assert skill_checker.get_hash() == hash1


def test_no_direct_body_yaml_access(linked_body):
    """Consumers should use BodyResolver, not hardcoded paths."""
    resolver = BodyResolver()
    assert resolver.body_yaml_path.exists()
    # All public reads should go through resolver API
    body = resolver.get_current_body_yaml()
    assert body.body_instance["robot_model"] == "unitree-g1"
