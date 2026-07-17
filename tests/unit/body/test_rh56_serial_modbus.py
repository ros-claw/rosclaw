"""Experiment 0 regression: Modbus frame layer + SerialModbusTransport.

Two layers:

* golden fixtures from the real audit (tests/fixtures/rh56_modbus/) —
  frame parsing regresses offline;
* a loopback fake serial implementing a mini device model — the full
  transport (read_state/write_position/delivery semantics) without hardware.
"""

from __future__ import annotations

import struct
import time
from pathlib import Path

import pytest

from rosclaw.body.rh56 import modbus
from rosclaw.body.rh56.transport import (
    CommandDelivery,
    SerialModbusTransport,
    TransportIOError,
)
from rosclaw.body.rh56.transport_profile import (
    CommandConfig,
    TransportConfig,
    TransportProfile,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "rh56_modbus"


def _profile(device: str = "/dev/null") -> TransportProfile:
    return TransportProfile(
        id="test_profile",
        transport=TransportConfig(
            type="serial_modbus_rtu", device=device, baudrate=115200, slave_id=1
        ),
        command=CommandConfig(
            actuator_count=6,
            position_range=[0, 1000],
            position_convention={"closed": 0, "open": 1000},
        ),
        action_order=["little", "ring", "middle", "index", "thumb", "thumb_rot"],
    )


# ---------------------------------------------------------------------------
# Golden fixtures
# ---------------------------------------------------------------------------


def test_fixture_read_request_frame() -> None:
    req = (FIXTURES / "read_positions_request.bin").read_bytes()
    rebuilt = modbus.build_read_holding_registers(1, modbus.Register.ANGLE_ACT, 6)
    assert req == rebuilt, "fixture and builder disagree on the 0x03 wire format"


def test_fixture_set_request_frame() -> None:
    req = (FIXTURES / "set_positions_request.bin").read_bytes()
    rebuilt = modbus.build_write_multiple_registers(1, modbus.Register.ANGLE_SET, [500] * 6)
    assert req == rebuilt, "fixture and builder disagree on the 0x10 wire format"


def test_fixture_real_response_parses() -> None:
    resp = (FIXTURES / "read_positions_response.bin").read_bytes()
    values = modbus.parse_read_response(resp, 1)
    assert len(values) == 6
    assert all(0 <= v <= 2000 for v in values)


def test_fixture_exception_frame_detected() -> None:
    frame = (FIXTURES / "exception_response.bin").read_bytes()
    with pytest.raises(modbus.ModbusExceptionError) as err:
        modbus.check_response_header(frame, 1, 0x03)
    assert err.value.exception_code == 0x02


def test_fixture_crc_failure_detected() -> None:
    frame = (FIXTURES / "crc_failure.bin").read_bytes()
    with pytest.raises(RuntimeError, match="crc"):
        modbus.parse_read_response(frame, 1)


def test_crc16_known_vector() -> None:
    # CRC16/MODBUS of an empty body is the 0xFFFF init value; the canonical
    # "123456789" vector is 0x4B37.
    assert modbus.crc16_modbus(b"") == 0xFFFF
    assert modbus.crc16_modbus(b"123456789") == 0x4B37


# ---------------------------------------------------------------------------
# Loopback fake serial (mini device model)
# ---------------------------------------------------------------------------


class _FakeDevice:
    """Answers Modbus requests like a real RH56 (mini register file)."""

    def __init__(self, slave_id: int = 1):
        self.slave_id = slave_id
        self.registers = {modbus.Register.HAND_ID: [slave_id], modbus.Register.VLTAGE: [608]}
        for base in (
            modbus.Register.ANGLE_ACT,
            modbus.Register.FORCE_ACT,
            modbus.Register.CURRENT,
            modbus.Register.STATUS,
            modbus.Register.TEMP,
            modbus.Register.ANGLE_SET,
            modbus.Register.SPEED_SET,
            modbus.Register.FORCE_SET,
        ):
            self.registers[base] = [1000] * 6
        self.registers[modbus.Register.TEMP] = [35] * 6
        self.registers[modbus.Register.FORCE_ACT] = [0] * 6
        self.registers[modbus.Register.CURRENT] = [0] * 6
        self.registers[modbus.Register.STATUS] = [2] * 6
        self.drop_next_write = False  # simulate a lost write ACK

    def handle(self, request: bytes) -> bytes:
        slave, fc = request[0], request[1]
        assert slave == self.slave_id
        if fc == 0x03:
            start, qty = struct.unpack(">HH", request[2:6])
            values = self.registers.get(start, [0] * qty)[:qty]
            body = bytes([slave, 0x03, qty * 2]) + b"".join(
                struct.pack(">H", v & 0xFFFF) for v in values
            )
        elif fc == 0x10:
            start, qty = struct.unpack(">HH", request[2:6])
            data = request[7 : 7 + qty * 2]
            self.registers[start] = [
                struct.unpack(">H", data[i : i + 2])[0] for i in range(0, qty * 2, 2)
            ]
            if self.drop_next_write:
                self.drop_next_write = False
                return b""  # ACK lost on the wire
            body = bytes([slave, 0x10]) + request[2:6]
        else:
            body = bytes([slave, fc | 0x80, 0x02])
        return body + struct.pack("<H", modbus.crc16_modbus(body))


class _FakeSerial:
    """pyserial-compatible loopback around :class:`_FakeDevice`."""

    def __init__(self, device: _FakeDevice):
        self._device = device
        self.is_open = True
        self.timeout = 0.05
        self._rx = bytearray()
        self.written: list[bytes] = []

    def write(self, data: bytes) -> int:
        self.written.append(bytes(data))
        self._rx += self._device.handle(bytes(data))
        return len(data)

    def flush(self) -> None:
        pass

    def read(self, n: int) -> bytes:
        if not self._rx:
            time.sleep(0.001)
            return b""
        out = bytes(self._rx[:n])
        del self._rx[:n]
        return out

    def reset_input_buffer(self) -> None:
        self._rx.clear()

    def close(self) -> None:
        self.is_open = False

    def open(self) -> None:
        self.is_open = True


@pytest.fixture
def transport() -> SerialModbusTransport:
    device = _FakeDevice()
    fake = _FakeSerial(device)
    transport = SerialModbusTransport(_profile(), existing_serial=fake)
    transport.connect()
    return transport


def test_read_state_fields(transport: SerialModbusTransport) -> None:
    state = transport.read_state()
    assert state.position == [1000] * 6
    assert state.temperature_c == [35.0] * 6
    assert state.status_bits == [2] * 6
    assert state.force_g == [0.0] * 6


def test_write_position_acknowledged(transport: SerialModbusTransport) -> None:
    ok = transport.write_position([500] * 6, speed=200, force_limit=300)
    assert ok
    assert transport.last_command_delivery == CommandDelivery.ACKNOWLEDGED
    state = transport.read_state()
    # The fake device's ANGLE_SET register was written, not ANGLE_ACT.
    device_set = transport._read_registers(modbus.Register.ANGLE_SET, 6)
    assert device_set == [500] * 6
    assert state.position == [1000] * 6  # actual unchanged (fake doesn't simulate motion)


def test_write_position_rejected_on_setpoint_mismatch(transport: SerialModbusTransport) -> None:
    # Make the device ignore writes to ANGLE_SET (setpoint won't match).
    device = transport._ser._device
    original = device.handle

    def _sabotage(request: bytes) -> bytes:
        if request[1] == 0x10 and struct.unpack(">H", request[2:4])[0] == modbus.Register.ANGLE_SET:
            return (
                bytes([request[0], 0x10])
                + request[2:6]
                + struct.pack("<H", modbus.crc16_modbus(bytes([request[0], 0x10]) + request[2:6]))
            )
        return original(request)

    device.handle = _sabotage  # type: ignore[method-assign]
    ok = transport.write_position([400] * 6, speed=200, force_limit=300)
    assert not ok
    assert transport.last_command_delivery == CommandDelivery.REJECTED


def test_uncertain_write_needs_reread(transport: SerialModbusTransport) -> None:
    # Lose the ACK but let the write land: ANGLE_SET updates, response dropped.
    device = transport._ser._device
    original = device.handle

    def _drop_ack(request: bytes) -> bytes:
        if request[1] == 0x10 and struct.unpack(">H", request[2:4])[0] == modbus.Register.ANGLE_SET:
            start, qty = struct.unpack(">HH", request[2:6])
            data = request[7 : 7 + qty * 2]
            device.registers[start] = [
                struct.unpack(">H", data[i : i + 2])[0] for i in range(0, qty * 2, 2)
            ]
            device.registers[modbus.Register.ANGLE_ACT] = device.registers[start]  # device moved
            return b""  # ACK lost
        return original(request)

    device.handle = _drop_ack  # type: ignore[method-assign]
    ok = transport.write_position([600] * 6, speed=200, force_limit=300)
    # Read-back finds setpoints (and actuals) already at target → acknowledged via re-read.
    assert ok
    assert transport.last_command_delivery == CommandDelivery.ACKNOWLEDGED


def test_uncertain_write_stays_fail_closed(transport: SerialModbusTransport) -> None:
    # Drop the ACK AND don't move: classification must be uncertain, no resend.
    device = transport._ser._device
    original = device.handle

    def _drop_ack_no_move(request: bytes) -> bytes:
        if request[1] == 0x10 and struct.unpack(">H", request[2:4])[0] == modbus.Register.ANGLE_SET:
            return b""
        return original(request)

    device.handle = _drop_ack_no_move  # type: ignore[method-assign]
    writes_before = len(transport._ser.written)
    ok = transport.write_position([300] * 6, speed=200, force_limit=300)
    assert not ok
    assert transport.last_command_delivery == CommandDelivery.UNCERTAIN
    # No blind resend: only speed/force/angle writes + read-back reads happened.
    angle_writes = [
        w
        for w in transport._ser.written[writes_before:]
        if w[1] == 0x10 and struct.unpack(">H", w[2:4])[0] == modbus.Register.ANGLE_SET
    ]
    assert len(angle_writes) == 1


def test_emergency_stop_freezes_at_actual(transport: SerialModbusTransport) -> None:
    assert transport.emergency_stop()
    assert transport._read_registers(modbus.Register.SPEED_SET, 6) == [0] * 6
    assert transport._read_registers(modbus.Register.ANGLE_SET, 6) == [1000] * 6


def test_not_connected_raises() -> None:
    transport = SerialModbusTransport(_profile(), existing_serial=_FakeSerial(_FakeDevice()))
    with pytest.raises(TransportIOError, match="serial_timeout"):
        transport.read_state()
