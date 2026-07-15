"""Abstract and concrete RH56 hand controllers."""

from __future__ import annotations

import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import serial

from rosclaw_rh56.protocol.inspire_protocol import DOF_NAMES, InspireProtocol
from rosclaw_rh56.transport.serial_rs485 import SerialRS485Transport, TransportConfig

from ..gesture_schema import HandTelemetry
from .guardian_transport import GuardianTransport
from .port_scanner import get_open_serial, resolve_port


class HandController(ABC):
    """Interface for a dexterous hand controller."""

    @abstractmethod
    def move_to_gesture(
        self, gesture_name: str, angles: List[int], speed: int, force: int
    ) -> bool: ...

    @abstractmethod
    def read_telemetry(self) -> HandTelemetry: ...

    @abstractmethod
    def safe_open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class MockHandController(HandController):
    """Software-only hand controller for mock/camera-only modes."""

    def __init__(self, initial_angles: Optional[List[int]] = None):
        self._angles = list(initial_angles) if initial_angles else [1000] * 6
        self._last_gesture = "ready"
        self._connected = True

    def move_to_gesture(
        self, gesture_name: str, angles: List[int], speed: int, force: int
    ) -> bool:
        self._angles = list(angles)
        self._last_gesture = gesture_name
        return True

    def read_telemetry(self) -> HandTelemetry:
        return HandTelemetry(
            timestamp=time.time(),
            angle_actual={name: self._angles[i] for i, name in enumerate(DOF_NAMES)},
            angle_set={name: self._angles[i] for i, name in enumerate(DOF_NAMES)},
        )

    def safe_open(self) -> None:
        self._angles = [1000] * 6
        self._last_gesture = "error"

    def close(self) -> None:
        self._connected = False


class _ExistingSerialTransport:
    """Minimal transport wrapper around an already-open pyserial object.

    The port scanner keeps CH340 ports open after discovery to avoid the
    close/reopen cycle that triggers driver errors.  This wrapper lets an
    existing ``serial.Serial`` object be used by ``RH56Controller`` without
    reopening.
    """

    def __init__(self, ser: serial.Serial):
        self._ser = ser

    def open(self) -> None:
        pass

    def close(self) -> None:
        if self._ser.is_open:
            try:
                self._ser.flush()
            except Exception:
                pass
            time.sleep(0.05)
            self._ser.close()

    def is_open(self) -> bool:
        return self._ser.is_open

    def write(self, data: bytes) -> None:
        self._ser.write(data)
        self._ser.flush()

    def flush_input(self) -> None:
        self._ser.reset_input_buffer()

    def read(self, n: int, timeout_s: Optional[float] = None) -> bytes:
        old_timeout = self._ser.timeout
        try:
            if timeout_s is not None:
                self._ser.timeout = timeout_s
            return self._ser.read(n)
        finally:
            self._ser.timeout = old_timeout


class RH56Controller(HandController):
    """RH56 RS485/Modbus RTU hand controller."""

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        device_id: int = 2,
        baudrate: int = 115200,
        timeout_s: float = 0.3,
        default_speed: int = 600,
        default_force: int = 300,
        existing_serial: Optional[serial.Serial] = None,
        transport: Optional[Any] = None,
    ):
        self._device_id = device_id
        self._default_speed = default_speed
        self._default_force = default_force
        self._proto = InspireProtocol(device_id=device_id)
        if transport is not None:
            self._transport = transport
        elif existing_serial is not None:
            self._transport = _ExistingSerialTransport(existing_serial)
        else:
            self._transport = SerialRS485Transport(
                TransportConfig(
                    kind="serial_rs485",
                    port=port,
                    baudrate=baudrate,
                    timeout_s=timeout_s,
                )
            )
        self._last_angles: List[int] = [1000] * 6
        # Serialize all Modbus transactions on this port: the 5 Hz telemetry
        # thread and gesture execution share one serial transport, and FTDI
        # adapters raise SerialException on concurrent reads where CH340
        # silently tolerated the race.
        self._io_lock = threading.RLock()

    def connect(self) -> None:
        with self._io_lock:
            if not self._transport.is_open():
                self._transport.open()
            # Configure default speed/force once at connect.
            self._transport.write(
                self._proto.write_speed_set([self._default_speed] * 6)
            )
            time.sleep(0.05)
            self._transport.write(
                self._proto.write_force_set([self._default_force] * 6)
            )
            time.sleep(0.05)

    def move_to_gesture(
        self, gesture_name: str, angles: List[int], speed: int, force: int
    ) -> bool:
        with self._io_lock:
            if not self._transport.is_open():
                self.connect()
            try:
                self._transport.write(self._proto.write_speed_set([speed] * 6))
                time.sleep(0.02)
                self._transport.write(self._proto.write_force_set([force] * 6))
                time.sleep(0.02)
                self._transport.write(self._proto.write_angle_set(angles))
                self._last_angles = list(angles)
                return True
            except Exception as exc:  # pragma: no cover
                print(f"RH56 move failed: {exc}")
                return False

    def read_telemetry(self) -> HandTelemetry:
        if not self._transport.is_open():
            self.connect()
        now = time.time()
        timeout = 0.2

        def _read_and_decode(read_fn, decode_fn):
            # Lock per Modbus transaction (not per 7-register batch) so
            # gesture execution can interleave between registers — holding
            # the lock across the whole batch blocked the UI/game thread
            # for seconds during retry storms.  Draining stale write-ACKs
            # (gesture commands leave unread 8-byte responses) before each
            # request makes the first attempt almost always succeed.
            #
            # Two-stage read: ``read(64)`` blocks for the FULL timeout when
            # the 17-byte response is shorter than 64 bytes (that alone cost
            # 200 ms per register ≈ 1.4 s per hand).  Read the 3-byte header
            # first (slave, function, byte-count), then exactly the remaining
            # payload + CRC — a transaction finishes in ~10 ms.
            for _ in range(3):
                with self._io_lock:
                    flush = getattr(self._transport, "flush_input", None)
                    if flush is not None:
                        flush()
                    self._transport.write(read_fn())
                    head = self._transport.read(3, timeout_s=timeout)
                    resp = head
                    if len(head) == 3:
                        # Exception frame (func|0x80) is 5 bytes total;
                        # normal frame carries head[2] payload bytes + CRC.
                        remaining = 2 if (head[1] & 0x80) else head[2] + 2
                        resp = head + self._transport.read(remaining, timeout_s=timeout)
                if (
                    resp and self._proto.__class__.__name__
                ):  # keep protocol object alive
                    decoded = decode_fn(resp)
                    if decoded is not None:
                        return decoded
                time.sleep(0.02)
            return None

        angles = _read_and_decode(
            self._proto.read_angle_actual, self._proto.decode_angle_actual
        )
        positions = _read_and_decode(
            self._proto.read_position_actual, self._proto.decode_position_actual
        )
        forces = _read_and_decode(
            self._proto.read_force_actual, self._proto.decode_force_actual
        )
        currents = _read_and_decode(
            self._proto.read_current, self._proto.decode_current
        )
        temps = _read_and_decode(
            self._proto.read_temperature, self._proto.decode_temperature
        )
        errors = _read_and_decode(self._proto.read_error, self._proto.decode_error)
        statuses = _read_and_decode(self._proto.read_status, self._proto.decode_status)

        def _to_dict(values):
            if values is None:
                return {name: None for name in DOF_NAMES}
            return {name: values[i] for i, name in enumerate(DOF_NAMES)}

        return HandTelemetry(
            timestamp=now,
            angle_actual=_to_dict(angles),
            angle_set=_to_dict(positions),
            force_act=_to_dict(forces),
            current_ma=_to_dict(currents),
            temperature_c=_to_dict(temps),
            error=_to_dict(errors),
            status=_to_dict(statuses),
        )

    def safe_open(self) -> None:
        self.move_to_gesture("error", [1000] * 6, speed=400, force=250)

    def close(self) -> None:
        self._transport.close()


def build_hand_controller(config: dict) -> HandController:
    controller_type = config.get("controller", "mock")
    if controller_type == "rh56":
        device_id = int(config.get("device_id", 2))

        # Optional serial guardian: keeps CH340 ports open in a daemon process.
        if os.environ.get("RH56_GUARDIAN"):
            socket_dir = os.environ.get("RH56_GUARDIAN_DIR", "/tmp/rh56_guardian")
            socket_path = os.path.join(socket_dir, f"{device_id}.sock")
            return RH56Controller(
                device_id=device_id,
                baudrate=int(config.get("baudrate", 115200)),
                timeout_s=float(config.get("timeout_s", 0.3)),
                default_speed=int(config.get("default_speed", 600)),
                default_force=int(config.get("default_force", 300)),
                transport=GuardianTransport(
                    socket_path, timeout_s=float(config.get("timeout_s", 0.3))
                ),
            )

        port = config.get("port", "/dev/ttyUSB0")
        resolved_port = resolve_port(device_id, port)
        existing_serial = get_open_serial(resolved_port)
        return RH56Controller(
            port=resolved_port,
            device_id=device_id,
            baudrate=int(config.get("baudrate", 115200)),
            timeout_s=float(config.get("timeout_s", 0.3)),
            default_speed=int(config.get("default_speed", 600)),
            default_force=int(config.get("default_force", 300)),
            existing_serial=existing_serial,
        )
    return MockHandController()
