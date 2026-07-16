"""P5 §13.2 execution permit tests."""

from __future__ import annotations

import time

import pytest

from rosclaw.integrations.lerobot.execution import PermitError, PermitManager

HASHES = {
    "policy_contract_hash": "sha256:policy",
    "body_hash": "sha256:body",
    "calibration_hash": "sha256:calib",
    "mapping_hash": "sha256:mapping",
    "transport_profile_hash": "sha256:transport",
}


def _issue(pm: PermitManager, **overrides):
    params = {
        "body_id": "rh56_mock",
        **HASHES,
        "operator_armed": True,
        "physical_estop_confirmed": True,
    }
    params.update(overrides)
    return pm.issue(**params)


def _validate(pm: PermitManager, permit_id: str, **overrides):
    params = {
        "body_id": "rh56_mock",
        **HASHES,
        "representation": "joint_position",
        "units": "raw_device_unit",
    }
    params.update(overrides)
    return pm.validate(permit_id, **params)


def test_permit_requires_valid_calibration() -> None:
    pm = PermitManager()
    with pytest.raises(PermitError, match="calibration_not_validated"):
        _issue(pm, calibration_status="uncalibrated")
    permit = _issue(pm, calibration_status="validated")
    assert permit.permit_id.startswith("permit_")


def test_permit_requires_operator_arm_and_estop() -> None:
    pm = PermitManager()
    with pytest.raises(PermitError, match="permit_not_armed"):
        _issue(pm, operator_armed=False)
    with pytest.raises(PermitError, match="permit_estop_unconfirmed"):
        _issue(pm, physical_estop_confirmed=False)


def test_permit_expires() -> None:
    pm = PermitManager()
    permit = _issue(pm, expires_in_sec=0.05)
    time.sleep(0.1)
    with pytest.raises(PermitError, match="permit_expired|permit_revoked"):
        _validate(pm, permit.permit_id)


def test_body_hash_change_revokes_permit() -> None:
    pm = PermitManager()
    permit = _issue(pm)
    with pytest.raises(PermitError, match="permit_hash_mismatch"):
        _validate(pm, permit.permit_id, body_hash="sha256:tampered")
    assert pm.is_revoked(permit.permit_id)


def test_policy_hash_change_revokes_permit() -> None:
    pm = PermitManager()
    permit = _issue(pm)
    with pytest.raises(PermitError, match="permit_hash_mismatch"):
        _validate(pm, permit.permit_id, policy_contract_hash="sha256:tampered")
    assert pm.is_revoked(permit.permit_id)


def test_mapping_hash_change_revokes_permit() -> None:
    pm = PermitManager()
    permit = _issue(pm)
    with pytest.raises(PermitError, match="permit_hash_mismatch"):
        _validate(pm, permit.permit_id, mapping_hash="sha256:tampered")
    assert pm.is_revoked(permit.permit_id)


def test_transport_hash_change_revokes_permit() -> None:
    pm = PermitManager()
    permit = _issue(pm)
    with pytest.raises(PermitError, match="permit_hash_mismatch"):
        _validate(pm, permit.permit_id, transport_profile_hash="sha256:tampered")
    assert pm.is_revoked(permit.permit_id)


def test_worker_restart_revokes_permit() -> None:
    pm = PermitManager()
    permit = _issue(pm)
    assert pm.on_worker_restart() == 1
    with pytest.raises(PermitError, match="permit_revoked"):
        _validate(pm, permit.permit_id)


def test_representation_and_unit_mismatch_blocked() -> None:
    pm = PermitManager()
    permit = _issue(pm)
    with pytest.raises(PermitError, match="permit_representation_mismatch"):
        _validate(pm, permit.permit_id, representation="cartesian_pose")
    with pytest.raises(PermitError, match="permit_unit_mismatch"):
        _validate(pm, permit.permit_id, units="radian")


def test_valid_permit_roundtrip() -> None:
    pm = PermitManager()
    permit = _issue(pm, calibration_status="validated")
    validated = _validate(pm, permit.permit_id)
    assert validated.permit_id == permit.permit_id
    assert validated.max_step_delta_raw == 30.0
