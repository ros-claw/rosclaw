"""P5 §13.1 transport profile tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.body.rh56.calibration import load_rh56_calibration
from rosclaw.body.rh56.transport import (
    MockModbusTransport,
    SerialModbusTransport,
    TransportIOError,
    TransportUnavailableError,
)
from rosclaw.body.rh56.transport_profile import (
    TransportBindingError,
    load_transport_profile,
    validate_transport_binding,
)

CONFIGS = Path(__file__).resolve().parents[3] / "configs"
RS485_PROFILE = CONFIGS / "rh56_right_rs485_v1.yaml"
CAN_PROFILE = CONFIGS / "rh56_can_v1.yaml"
CALIBRATION = CONFIGS / "rh56_right_01_calibration.yaml"


def test_rs485_profile_roundtrip(tmp_path: Path) -> None:
    profile = load_transport_profile(RS485_PROFILE)
    assert profile.id == "rh56_right_rs485_v1"
    assert profile.transport.type == "serial_modbus_rtu"
    assert profile.command.actuator_count == 6
    assert profile.position_range == (0, 1000)
    assert profile.position_open() == 1000
    assert profile.position_closed() == 0
    assert profile.action_order == [
        "little",
        "ring",
        "middle",
        "index",
        "thumb",
        "thumb_rot",
    ]
    # Roundtrip via dict preserves the content hash.
    clone = type(profile).from_dict(profile.to_dict())
    assert clone.content_hash() == profile.content_hash()


def test_can_and_rs485_scale_not_interchangeable() -> None:
    rs485 = load_transport_profile(RS485_PROFILE)
    can = load_transport_profile(CAN_PROFILE)
    assert can.command.actuator_count == 11
    assert can.position_range == (0, 65535)

    # CAN action space must not bind to the RS485 profile.
    with pytest.raises(TransportBindingError, match="actuator_count_mismatch"):
        validate_transport_binding(rs485, action_dim=11)
    with pytest.raises(TransportBindingError, match="command_scale_mismatch"):
        validate_transport_binding(rs485, position_range=(0, 65535))
    # RS485 provider must not bind to the CAN profile and vice versa.
    with pytest.raises(TransportBindingError, match="provider_transport_mismatch"):
        validate_transport_binding(rs485, provider_ref="inspire_rh56_can")
    with pytest.raises(TransportBindingError, match="provider_transport_mismatch"):
        validate_transport_binding(can, provider_ref="inspire_rh56_serial")


def test_actuator_count_mismatch_blocked() -> None:
    profile = load_transport_profile(RS485_PROFILE)
    with pytest.raises(TransportBindingError, match="actuator_count_mismatch"):
        validate_transport_binding(profile, action_dim=5)
    with pytest.raises(TransportBindingError, match="actuator_count_mismatch"):
        validate_transport_binding(profile, action_names=["a", "b", "c", "d", "e", "f"])


def test_device_path_missing() -> None:
    profile = load_transport_profile(RS485_PROFILE)
    # Deterministic: point the profile at a device that never exists.
    profile.transport.device = "/dev/definitely_not_a_real_rh56"
    with pytest.raises(TransportUnavailableError, match="device_path_disappeared"):
        SerialModbusTransport(profile)


def test_serial_timeout() -> None:
    profile = load_transport_profile(RS485_PROFILE)
    transport = MockModbusTransport(profile)
    with pytest.raises(TransportIOError, match="serial_timeout"):
        transport.read_state()
    transport.connect()
    feedback = transport.read_state()
    assert len(feedback.position) == 6
    transport.fail_next_read()
    with pytest.raises(TransportIOError, match="io_error"):
        transport.read_state()


def test_serial_reenumeration_revokes_permit() -> None:
    from rosclaw.integrations.lerobot.execution import PermitManager

    profile = load_transport_profile(RS485_PROFILE)
    pm = PermitManager()
    permit = pm.issue(
        body_id="rh56_mock",
        policy_contract_hash="sha256:p",
        body_hash="sha256:b",
        calibration_hash="sha256:c",
        mapping_hash="sha256:m",
        transport_profile_hash=profile.content_hash(),
        operator_armed=True,
        physical_estop_confirmed=True,
    )
    # Simulate USB re-enumeration: transport drops, permits revoked.
    transport = MockModbusTransport(profile)
    transport.connect()
    transport.drop_connection()
    with pytest.raises(TransportIOError, match="device_path_disappeared"):
        transport.read_state()
    revoked = pm.on_serial_reconnect()
    assert revoked == 1
    assert pm.get(permit.permit_id) is None
    assert pm.is_revoked(permit.permit_id)


def test_calibration_schema_and_gate() -> None:
    calib = load_rh56_calibration(CALIBRATION)
    profile = load_transport_profile(RS485_PROFILE)
    calib.validate_against_profile(profile)
    # The shipped config is marked validated after on-hardware acceptance
    # (HAND-01). Exercise the gate-denial path with an explicit uncalibrated
    # copy so the test does not depend on the shipped config's mutable status.
    import dataclasses

    from rosclaw.body.rh56.calibration import (
        CalibrationError,
        CalibrationValidation,
        RH56CalibrationGate,
    )

    uncalibrated = dataclasses.replace(calib, validation=CalibrationValidation())
    assert uncalibrated.status == "uncalibrated"
    gate = RH56CalibrationGate(uncalibrated, profile)
    with pytest.raises(CalibrationError, match="calibration_not_validated"):
        gate.check()
    validated = gate.mark_validated(rounds=5, body_hash="sha256:b")
    assert validated.status == "validated"
    RH56CalibrationGate(validated, profile).check()
