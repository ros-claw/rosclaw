"""Contract tests: provider binder consumes EffectiveBody, not body.yaml."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.body.resolver import BodyResolver
from rosclaw.body.service import BodyInstanceService
from rosclaw.provider.body_binder import ProviderBodyBinder


@pytest.fixture
def linked_workspace(tmp_path: Path, monkeypatch) -> Path:
    workspace = tmp_path / ".rosclaw"
    monkeypatch.setenv("HOME", str(tmp_path))
    BodyInstanceService().create_or_init(
        robot="unitree-g1", name="g1-provider", mode="registry", update_registry=True, switch_active=True
    )
    return workspace


def test_provider_binder_uses_effective_body(linked_workspace: Path):
    resolver = BodyResolver()
    body = resolver.resolve("rosclaw://body/current/effective")
    binder = ProviderBodyBinder.from_effective_body(body)

    assert binder.effective_body_hash == body.effective_body_hash
    assert binder.eurdf_uri == body.eurdf_uri
    assert binder.body_instance_id == body.body_instance_id


def test_provider_diagnosis_hash_matches_body(linked_workspace: Path):
    resolver = BodyResolver()
    body = resolver.resolve("rosclaw://body/current/effective")
    binder = ProviderBodyBinder.from_effective_body(body)

    diagnosis = binder.diagnose()
    assert diagnosis.body_instance_id == body.body_instance_id
    assert diagnosis.effective_body_hash == body.effective_body_hash


def test_provider_diagnosis_required_vs_optional(linked_workspace: Path):
    resolver = BodyResolver()
    body = resolver.resolve("rosclaw://body/current/effective")
    binder = ProviderBodyBinder.from_effective_body(body)

    required = {i.name for i in binder.required_interfaces()}
    optional = {i.name for i in binder.optional_interfaces()}
    assert required.isdisjoint(optional)

    diagnosis = binder.diagnose(available={"joint_states"})
    if "joint_states" in required:
        assert diagnosis.interfaces["joint_states"]["status"] == "available"

    for iface_name in optional:
        assert diagnosis.interfaces[iface_name]["required"] is False
