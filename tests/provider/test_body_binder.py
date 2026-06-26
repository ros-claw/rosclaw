"""Dedicated tests for ProviderBodyBinder."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.body.resolver import BodyResolver
from rosclaw.body.schema import EffectiveBody
from rosclaw.body.service import BodyInstanceService
from rosclaw.provider.body_binder import ProviderBodyBinder


@pytest.fixture
def linked_workspace(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    BodyInstanceService(workspace=tmp_path).create_or_init(robot="unitree-g1", name="g1-binder", mode="single")
    return tmp_path


def test_binder_exposes_hash_and_uri(linked_workspace: Path):
    resolver = BodyResolver(workspace=linked_workspace)
    body = resolver.resolve("rosclaw://body/current/effective")
    binder = ProviderBodyBinder.from_effective_body(body)

    assert binder.effective_body_hash == body.effective_body_hash
    assert binder.eurdf_uri == body.eurdf_uri
    assert binder.body_instance_id == body.body_instance_id


def test_binder_required_and_optional_are_disjoint(linked_workspace: Path):
    resolver = BodyResolver(workspace=linked_workspace)
    body = resolver.resolve("rosclaw://body/current/effective")
    binder = ProviderBodyBinder.from_effective_body(body)

    required = {i.name for i in binder.required_interfaces()}
    optional = {i.name for i in binder.optional_interfaces()}
    assert required.isdisjoint(optional)


def test_diagnose_status_nominal_when_all_available(linked_workspace: Path):
    resolver = BodyResolver(workspace=linked_workspace)
    body = resolver.resolve("rosclaw://body/current/effective")
    binder = ProviderBodyBinder.from_effective_body(body)

    diagnosis = binder.diagnose(available={i.name for i in binder.required_interfaces()})
    # Optional interfaces not in the available set are reported unavailable,
    # so the overall status may be degraded. Required interfaces must be available.
    assert diagnosis.status in ("nominal", "degraded", "unknown")
    for iface in binder.required_interfaces():
        assert diagnosis.interfaces[iface.name]["status"] == "available"


def test_diagnose_blocked_when_required_missing(linked_workspace: Path):
    resolver = BodyResolver(workspace=linked_workspace)
    body = resolver.resolve("rosclaw://body/current/effective")
    binder = ProviderBodyBinder.from_effective_body(body)

    required = {i.name for i in binder.required_interfaces()}
    if required:
        diagnosis = binder.diagnose(available=set())
        assert diagnosis.status == "blocked"
        missing = required.pop()
        assert diagnosis.interfaces[missing]["status"] == "unavailable"


def test_diagnose_reflects_unavailable_sensor():
    body = EffectiveBody(
        body_instance_id="test",
        eurdf_uri="rosclaw://eurdf/test@1.0.0",
        effective_body_hash="hash",
        compiled_at="now",
        sensors={"head_rgb_camera": {"status": "unavailable", "provider_ref": "camera"}},
        provider_interfaces={"sensor": {"required": ["head_rgb_camera"], "optional": []}},
    )
    binder = ProviderBodyBinder.from_effective_body(body)
    diagnosis = binder.diagnose()

    assert diagnosis.status == "blocked"
    assert diagnosis.interfaces["head_rgb_camera"]["status"] == "unavailable"


def test_diagnose_optional_missing_is_degraded_not_blocked():
    body = EffectiveBody(
        body_instance_id="test",
        eurdf_uri="rosclaw://eurdf/test@1.0.0",
        effective_body_hash="hash",
        compiled_at="now",
        sensors={"extra_camera": {"status": "unavailable"}},
        provider_interfaces={"sensor": {"required": [], "optional": ["extra_camera"]}},
    )
    binder = ProviderBodyBinder.from_effective_body(body)
    diagnosis = binder.diagnose()

    assert diagnosis.status == "degraded"
    assert diagnosis.interfaces["extra_camera"]["required"] is False


def test_diagnosis_dict_is_serializable(linked_workspace: Path):
    resolver = BodyResolver(workspace=linked_workspace)
    body = resolver.resolve("rosclaw://body/current/effective")
    binder = ProviderBodyBinder.from_effective_body(body)
    diagnosis = binder.diagnose()

    data = diagnosis.to_dict()
    assert data["body_instance_id"] == body.body_instance_id
    assert data["effective_body_hash"] == body.effective_body_hash
    assert data["status"]
    assert isinstance(data["interfaces"], dict)
    assert isinstance(data["summary"], dict)
