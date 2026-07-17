"""Inspire RH56 Modbus-RTU frame layer (Experiment 0, vendored and audited).

Frame builders/parsers for function codes 0x03 / 0x06 / 0x10, CRC16, and the
Inspire register map.  This is a self-contained copy of the implementation
proven across the 35h 7×24 RH56 stress runs (rosclaw_rh56 runtime package),
vendored so the published ``rosclaw`` package has no unpublished dependency.

Wire format (audited on physical hands, slave ids 1/2, 115200 8N1):

* request:  ``[slave][fc][addr_hi addr_lo][qty_hi qty_lo][crc_lo crc_hi]``
* 0x10 request: ``[slave][0x10][addr][qty][byte_count][data...][crc]``
* CRC16/MODBUS poly 0xA001, little-endian trailing pair.
"""

from __future__ import annotations

import struct


# Register addresses (holding registers, 16-bit each) — RH56 Series User Manual V1.0.9.
class Register:
    HAND_ID = 0x03E8
    BAUD_RATE = 0x03EA
    CLEAR_ERROR = 0x03EC
    SAVE = 0x03ED
    RESET_PARA = 0x03EE
    GESTURE_FORCE_CLB = 0x03F1

    CURRENT_LIMIT = 0x03FC  # 6 shorts
    DEFAULT_SPEED_SET = 0x0408  # 6 shorts
    DEFAULT_FORCE_SET = 0x0414  # 6 shorts

    VLTAGE = 0x05C0
    POS_SET = 0x05C2  # 6 shorts
    ANGLE_SET = 0x05CE  # 6 shorts
    FORCE_SET = 0x05DA  # 6 shorts
    SPEED_SET = 0x05F2  # 6 shorts

    POS_ACT = 0x05FE  # 6 shorts
    ANGLE_ACT = 0x060A  # 6 shorts
    FORCE_ACT = 0x062E  # 6 shorts
    CURRENT = 0x063A  # 6 shorts
    ERROR = 0x0646  # 6 registers (firmware returns 6, manual says 3)
    STATUS = 0x064C  # 6 registers
    TEMP = 0x0652  # 6 registers


DOF_NAMES = ["little", "ring", "middle", "index", "thumb", "thumb_rot"]

POS_MIN, POS_MAX = 0, 2000
ANGLE_MIN, ANGLE_MAX = 0, 2000
SPEED_MAX = 1000
FORCE_MAX = 1000

# Modbus exception codes (response fc = request | 0x80).
EXCEPTION_CODES = {
    0x01: "illegal_function",
    0x02: "illegal_data_address",
    0x03: "illegal_data_value",
    0x04: "slave_device_failure",
    0x05: "acknowledge",
    0x06: "slave_device_busy",
    0x08: "memory_parity_error",
    0x0A: "gateway_path_unavailable",
    0x0B: "gateway_target_no_response",
}


class ModbusExceptionError(RuntimeError):
    """Raised when the device answers with a Modbus exception frame."""

    def __init__(self, function_code: int, exception_code: int):
        self.function_code = function_code
        self.exception_code = exception_code
        meaning = EXCEPTION_CODES.get(exception_code, f"unknown_{exception_code:#x}")
        super().__init__(
            f"modbus_exception: fc={function_code:#04x} code={exception_code:#04x} ({meaning})"
        )


def crc16_modbus(data: bytes) -> int:
    """CRC16/MODBUS (poly 0xA001, init 0xFFFF)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def build_read_holding_registers(slave_id: int, start_addr: int, quantity: int) -> bytes:
    """Build a 0x03 request frame."""
    if not (1 <= slave_id <= 247):
        raise ValueError(f"Invalid Modbus slave id: {slave_id}")
    if not (1 <= quantity <= 125):
        raise ValueError(f"Invalid quantity: {quantity}")
    body = bytes([slave_id, 0x03]) + struct.pack(">HH", start_addr, quantity)
    return body + struct.pack("<H", crc16_modbus(body))


def build_write_multiple_registers(slave_id: int, start_addr: int, values: list[int]) -> bytes:
    """Build a 0x10 request frame (values are 16-bit unsigned)."""
    if not (1 <= slave_id <= 247):
        raise ValueError(f"Invalid Modbus slave id: {slave_id}")
    if not (1 <= len(values) <= 123):
        raise ValueError(f"Invalid register count: {len(values)}")
    byte_count = len(values) * 2
    data = b"".join(struct.pack(">H", v & 0xFFFF) for v in values)
    body = (
        bytes([slave_id, 0x10])
        + struct.pack(">HH", start_addr, len(values))
        + bytes([byte_count])
        + data
    )
    return body + struct.pack("<H", crc16_modbus(body))


def build_write_single_register(slave_id: int, addr: int, value: int) -> bytes:
    """Build a 0x06 request frame."""
    if not (1 <= slave_id <= 247):
        raise ValueError(f"Invalid Modbus slave id: {slave_id}")
    body = bytes([slave_id, 0x06]) + struct.pack(">HH", addr, value & 0xFFFF)
    return body + struct.pack("<H", crc16_modbus(body))


def check_response_header(response: bytes, slave_id: int, function_code: int) -> None:
    """Validate header + CRC; raises ModbusExceptionError on exception frames."""
    if len(response) < 5:
        raise RuntimeError(f"modbus_short_frame: {len(response)} bytes")
    if response[0] != slave_id:
        raise RuntimeError(f"modbus_slave_mismatch: expected {slave_id}, got {response[0]}")
    if response[1] == (function_code | 0x80):
        raise ModbusExceptionError(function_code, response[2])
    if response[1] != function_code:
        raise RuntimeError(
            f"modbus_fc_mismatch: expected {function_code:#04x}, got {response[1]:#04x}"
        )
    crc_received = struct.unpack("<H", response[-2:])[0]
    crc_computed = crc16_modbus(response[:-2])
    if crc_received != crc_computed:
        raise RuntimeError(
            f"modbus_crc_failure: received {crc_received:#06x} != computed {crc_computed:#06x}"
        )


def parse_read_response(response: bytes, slave_id: int) -> list[int]:
    """Parse a 0x03 response into 16-bit register values (validated)."""
    check_response_header(response, slave_id, 0x03)
    byte_count = response[2]
    if len(response) != 5 + byte_count:
        raise RuntimeError(
            f"modbus_length_mismatch: header says {byte_count} bytes, frame has {len(response) - 5}"
        )
    return [struct.unpack(">H", response[3 + i : 3 + i + 2])[0] for i in range(0, byte_count, 2)]


def to_signed_16(value: int) -> int:
    return value if value < 32768 else value - 65536
