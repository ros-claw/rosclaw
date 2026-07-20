"""RH56 transport backends.

``RH56Transport`` is the minimal interface every backend must implement:

- :meth:`read_state` — position/force/current/status/temperature per actuator
- :meth:`write_position` — one position command (single-step execution only)
- :meth:`emergency_stop` — best-effort stop command

Two backends are provided:

- :class:`MockModbusTransport` — a fully in-memory simulated device used by
  tests and by the mock shadow gate.  It models first-order actuator motion,
  contact force, current draw, temperature drift and status bits.
- :class:`SerialModbusTransport` — the real RS485/Modbus-RTU backend
  (Experiment 0): vendored frame layer (0x03/0x06/0x10, CRC16), two-stage
  reads, and acknowledged/rejected/uncertain write delivery classification.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import termios
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

from rosclaw.body.rh56.constants import RS485_ACTUATOR_ORDER
from rosclaw.body.rh56.transport_profile import TransportProfile

logger = logging.getLogger("rosclaw.body.rh56.transport")


class TransportUnavailableError(RuntimeError):
    """Raised when a transport backend cannot be used (missing device, etc.)."""


class TransportIOError(RuntimeError):
    """Raised on I/O failure during a transport transaction."""


@dataclass
class RH56Feedback:
    """One feedback snapshot from the hand (raw device units)."""

    position: list[int]
    force_g: list[float]
    current_ma: list[float]
    status_bits: list[int]
    temperature_c: list[float]
    timestamp_monotonic_ns: int = 0
    ok: bool = True
    error: str | None = None

    @classmethod
    def zero(cls, count: int = 6) -> RH56Feedback:
        return cls(
            position=[0] * count,
            force_g=[0.0] * count,
            current_ma=[0.0] * count,
            status_bits=[0] * count,
            temperature_c=[25.0] * count,
            timestamp_monotonic_ns=time.monotonic_ns(),
        )


class RH56Transport(Protocol):
    """Minimal transport contract for RH56 backends."""

    execution_mode: str
    implementation_kind: str

    def connect(self) -> None: ...

    def close(self) -> None: ...

    def is_connected(self) -> bool: ...

    def read_state(self) -> RH56Feedback: ...

    def read_angle_setpoints(self) -> list[int]:
        """Read the transport's currently active actuator setpoints."""

    def write_position(
        self,
        positions: list[int],
        *,
        speed: int,
        force_limit: int,
    ) -> bool:
        """Send one position command.  Returns True when acknowledged."""

    def emergency_stop(self) -> bool: ...


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


@dataclass
class MockActuatorSpec:
    """Simulated actuator dynamics parameters."""

    max_step_per_tick: float = 50.0
    contact_position: int | None = None  # where simulated contact happens
    contact_force_g: float = 120.0
    noise: float = 0.5


class MockModbusTransport:
    """In-memory simulated RH56 RS485 device.

    Simulates, per tick (each ``read_state`` call advances one tick):

    - first-order approach of actual position toward the last commanded target
    - contact force when an actuator pushes past ``contact_position``
    - current draw proportional to motion, near-zero at rest (matching the
      observed RH56 behaviour that CURRENT returns to ~0 during static contact)
    - slow temperature drift upward while moving, cooling at rest
    - status protection bits on over-temperature or over-force
    """

    execution_mode = "FIXTURE"
    implementation_kind = "synthetic_fixture"

    def __init__(
        self,
        profile: TransportProfile,
        *,
        initial_position: int | None = None,
        tick_hz: float = 100.0,
        seed_positions: list[int] | None = None,
    ):
        self.profile = profile
        self._count = profile.command.actuator_count
        self._order = list(profile.action_order or RS485_ACTUATOR_ORDER)
        open_pos = profile.position_open()
        self._position: list[float] = (
            [float(p) for p in seed_positions]
            if seed_positions is not None
            else [float(initial_position if initial_position is not None else open_pos)]
            * self._count
        )
        self._target: list[float] = list(self._position)
        self._speed: int = 100
        self._force_limit: int = 100
        self._force_g: list[float] = [0.0] * self._count
        self._current_ma: list[float] = [0.0] * self._count
        self._status_bits: list[int] = [0] * self._count
        self._temperature_c: list[float] = [32.0] * self._count
        self._connected = False
        self._tick_hz = tick_hz
        self._specs = [MockActuatorSpec() for _ in range(self._count)]
        self._estopped = False
        self._io_fail_next = False  # test hook: next read raises TransportIOError
        self._disconnect_next = False  # test hook: next read reports disconnect

    # -- test hooks -----------------------------------------------------

    def set_contact(
        self, actuator_index: int, contact_position: int, force_g: float = 120.0
    ) -> None:
        self._specs[actuator_index].contact_position = contact_position
        self._specs[actuator_index].contact_force_g = force_g

    def fail_next_read(self) -> None:
        self._io_fail_next = True

    def drop_connection(self) -> None:
        self._disconnect_next = True

    # -- transport contract ----------------------------------------------

    def connect(self) -> None:
        self._connected = True
        self._estopped = False

    def close(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def read_state(self) -> RH56Feedback:
        if not self._connected:
            raise TransportIOError("serial_timeout: transport not connected")
        if self._io_fail_next:
            self._io_fail_next = False
            raise TransportIOError("io_error: simulated read failure")
        if self._disconnect_next:
            self._disconnect_next = False
            self._connected = False
            raise TransportIOError("device_path_disappeared: simulated disconnect")

        dt = 1.0 / self._tick_hz
        for i in range(self._count):
            spec = self._specs[i]
            error = self._target[i] - self._position[i]
            if self._estopped:
                error = 0.0
            step = math.copysign(
                min(abs(error), spec.max_step_per_tick * dt * (self._speed / 100.0)),
                error,
            )
            moving = abs(step) > 1e-6
            self._position[i] += step

            # Contact model: pressing past the contact position builds force.
            if spec.contact_position is not None and self._position[i] <= spec.contact_position:
                overshoot = spec.contact_position - self._position[i]
                self._force_g[i] = min(
                    spec.contact_force_g, spec.contact_force_g * (0.5 + overshoot / 20.0)
                )
                self._position[i] = spec.contact_position  # blocked by contact
            else:
                decay = 0.6 if not moving else 0.9
                self._force_g[i] *= decay

            # Current: draw while moving, decays to ~0 at rest (real RH56 behaviour).
            target_current = 180.0 if moving else 0.0
            self._current_ma[i] += (target_current - self._current_ma[i]) * 0.4

            # Temperature drift.
            if moving:
                self._temperature_c[i] += 0.01
            else:
                self._temperature_c[i] = max(32.0, self._temperature_c[i] - 0.005)

            # Status protection bits.
            bits = 0
            if self._temperature_c[i] >= 60.0:
                bits |= 0x02  # over-temperature
            if self._force_g[i] >= 300.0:
                bits |= 0x01  # over-current/over-force
            self._status_bits[i] = bits

        return RH56Feedback(
            position=[int(round(p)) for p in self._position],
            force_g=[round(f, 2) for f in self._force_g],
            current_ma=[round(c, 2) for c in self._current_ma],
            status_bits=list(self._status_bits),
            temperature_c=[round(t, 2) for t in self._temperature_c],
            timestamp_monotonic_ns=time.monotonic_ns(),
        )

    def write_position(self, positions: list[int], *, speed: int, force_limit: int) -> bool:
        if not self._connected:
            raise TransportIOError("serial_timeout: transport not connected")
        if len(positions) != self._count:
            raise TransportIOError(
                f"actuator_count_mismatch: got {len(positions)} positions, expected {self._count}"
            )
        if self._estopped:
            return False
        lo, hi = self.profile.position_range
        self._target = [float(max(lo, min(hi, int(p)))) for p in positions]
        self._speed = int(speed)
        self._force_limit = int(force_limit)
        return True

    def read_angle_setpoints(self) -> list[int]:
        """Current setpoints (the mock tracks its last commanded target)."""
        if not self._connected:
            raise TransportIOError("serial_timeout: transport not connected")
        return [int(t) for t in self._target]

    def emergency_stop(self) -> bool:
        self._estopped = True
        self._target = list(self._position)
        self._force_g = [0.0] * self._count
        self._current_ma = [0.0] * self._count
        return True


# ---------------------------------------------------------------------------
# Real serial backend (Experiment 0)
# ---------------------------------------------------------------------------


class CommandDelivery:
    """Delivery state of the last write_position call (§doc: acknowledged/rejected/uncertain)."""

    ACKNOWLEDGED = "acknowledged"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


class SerialModbusTransport:
    """RS485/Modbus-RTU backend for the real RH56 (Experiment 0 implementation).

    Frame layer: :mod:`rosclaw.body.rh56.modbus` (0x03/0x06/0x10, CRC16).
    IO discipline matches the hard-won lessons from the 7×24 runs:

    * one Modbus transaction at a time per port (RLock);
    * two-stage read: 3-byte header first, then exactly ``2 + byte_count + 2``
      bytes — a blocking bulk read stalls for the whole timeout when the
      response is shorter than requested;
    * stale write-ACKs are drained before each transaction;
    * write uncertainty is never blindly retried: after a write timeout the
      actual position is re-read and only then is delivery classified as
      ``acknowledged`` / ``rejected`` / ``uncertain`` (fail-closed).
    """

    execution_mode = "REAL"
    implementation_kind = "hardware_serial_modbus_rtu"

    _READ_TIMEOUT_S = 0.2
    _WRITE_ACK_MIN_LEN = 8
    _DELIVERY_TOLERANCE = 40  # raw position units

    def __init__(
        self,
        profile: TransportProfile,
        *,
        existing_serial: Any | None = None,
        trace_hook: Any | None = None,
    ):
        device = profile.transport.device
        if not existing_serial and (not device or not os.path.exists(device)):
            raise TransportUnavailableError(
                f"device_path_disappeared: {device or '<unset>'} not available; "
                "SerialModbusTransport requires the physical RH56 device"
            )
        self.profile = profile
        self._device = device
        self._baudrate = int(profile.transport.baudrate or 115200)
        self._slave_id = int(profile.transport.slave_id)
        self._trace_hook = trace_hook
        self._lock = threading.RLock()
        self._ser: Any | None = existing_serial
        self._owns_serial = existing_serial is None
        self._connected = False
        self._last_command_delivery = CommandDelivery.ACKNOWLEDGED

    # ------------------------------------------------------------------
    # lifecycle

    def connect(self) -> None:
        with self._lock:
            if self._connected and self.is_connected():
                return
            if self._ser is None:
                try:
                    import serial
                except ImportError as exc:
                    raise TransportUnavailableError(
                        "pyserial_missing: pip install pyserial"
                    ) from exc
                try:
                    self._ser = serial.Serial(
                        port=self._device,
                        baudrate=self._baudrate,
                        bytesize=8,
                        parity="N",
                        stopbits=1,
                        timeout=0.05,
                        write_timeout=1.0,
                    )
                except (OSError, termios.error) as exc:
                    raise TransportUnavailableError(
                        f"device_path_disappeared: cannot open {self._device}: {exc}"
                    ) from exc
            elif not self._ser.is_open:
                # Reopen after close() — pyserial objects are re-openable.
                try:
                    self._ser.open()
                except (OSError, termios.error) as exc:
                    raise TransportUnavailableError(
                        f"device_path_disappeared: cannot reopen {self._device}: {exc}"
                    ) from exc
            self._connected = True

    def close(self) -> None:
        with self._lock:
            if self._ser is not None and self._owns_serial:
                with contextlib.suppress(Exception):
                    self._ser.close()
            self._connected = False

    def is_connected(self) -> bool:
        return self._connected and self._ser is not None and bool(self._ser.is_open)

    # ------------------------------------------------------------------
    # frame IO

    @contextlib.contextmanager
    def _io_guard(self, op: str):
        """Convert pyserial/termios failures into :class:`TransportIOError`.

        On Linux a vanished adapter surfaces as ``termios.error`` (EIO) from
        pyserial's read/write/flush paths — and ``termios.error`` is NOT an
        OSError subclass, so a plain ``except OSError`` misses it (found by
        the exp4 USB-unbind fault injection: a raw termios.error escaped as
        EXECUTOR_ERROR instead of escalating COMMUNICATION_LOST).  EIO-class
        failures also mark the transport disconnected so later calls fail
        fast instead of hammering a dead fd.
        """
        try:
            yield
        except (OSError, termios.error) as exc:
            errno_ = getattr(exc, "errno", None)
            if errno_ is None and exc.args and isinstance(exc.args[0], int):
                errno_ = exc.args[0]
            if errno_ in (5, 6, 19):  # EIO, ENXIO, ENODEV — adapter gone
                self._connected = False
            raise TransportIOError(f"io_error: serial {op} failed: {exc}") from exc

    def _emit_trace(self, direction: str, data: bytes) -> None:
        if self._trace_hook is not None:
            with contextlib.suppress(Exception):
                self._trace_hook(direction, data)

    def _write(self, data: bytes) -> None:
        self._emit_trace("tx", data)
        with self._io_guard("write"):
            self._ser.write(data)
            self._ser.flush()

    def _read_exact(self, n: int, timeout_s: float) -> bytes:
        deadline = time.monotonic() + timeout_s
        buf = b""
        old_timeout = self._ser.timeout
        try:
            self._ser.timeout = 0.02
            while len(buf) < n and time.monotonic() < deadline:
                with self._io_guard("read"):
                    chunk = self._ser.read(n - len(buf))
                if chunk:
                    buf += chunk
        finally:
            self._ser.timeout = old_timeout
        if buf:
            self._emit_trace("rx", buf)
        return buf

    def _transact(self, request: bytes, *, is_read: bool, timeout_s: float | None = None) -> bytes:
        """One request/response cycle: flush stale bytes, write, two-stage read."""
        timeout_s = timeout_s or self._READ_TIMEOUT_S
        with self._lock:
            if not self.is_connected():
                raise TransportIOError("serial_timeout: transport not connected")
            with self._io_guard("flush"):
                self._ser.reset_input_buffer()
            self._write(request)
            header = self._read_exact(3, timeout_s)
            if len(header) < 3:
                raise TransportIOError(f"serial_timeout: no response within {timeout_s}s")
            if header[1] & 0x80:
                # Exception frame: [slave][fc|0x80][code][crc_lo crc_hi]
                rest = self._read_exact(2, timeout_s)
                frame = header + rest
                self._emit_trace("rx", b"")
                from rosclaw.body.rh56.modbus import ModbusExceptionError

                raise ModbusExceptionError(header[1] & 0x7F, header[2])
            byte_count = header[2] if is_read else 5
            rest_len = byte_count + 2 if is_read else 5  # payload + crc
            rest = self._read_exact(rest_len, timeout_s)
            frame = header + rest
            if len(frame) < (5 + byte_count if is_read else 8):
                raise TransportIOError(
                    f"serial_timeout: short frame after header ({len(frame)} bytes)"
                )
            return frame

    def _read_registers(self, start_addr: int, quantity: int) -> list[int]:
        from rosclaw.body.rh56 import modbus

        frame = self._transact(
            modbus.build_read_holding_registers(self._slave_id, start_addr, quantity),
            is_read=True,
        )
        return modbus.parse_read_response(frame, self._slave_id)

    def _write_registers(self, start_addr: int, values: list[int]) -> None:
        from rosclaw.body.rh56 import modbus

        frame = self._transact(
            modbus.build_write_multiple_registers(self._slave_id, start_addr, values),
            is_read=False,
        )
        modbus.check_response_header(frame, self._slave_id, 0x10)

    # ------------------------------------------------------------------
    # transport contract

    def read_state(self) -> RH56Feedback:
        from rosclaw.body.rh56.modbus import Register, to_signed_16

        if not self.is_connected():
            raise TransportIOError("serial_timeout: transport not connected")
        positions = self._read_registers(Register.ANGLE_ACT, 6)
        forces = self._read_registers(Register.FORCE_ACT, 6)
        currents = self._read_registers(Register.CURRENT, 6)
        # STATUS/TEMP register width is firmware-specific: some revisions
        # expose 6 per-actuator registers, others group them into 3 (one per
        # actuator pair).  Reading 6 on a grouped device bleeds adjacent
        # blocks into slots 3-5 (temps into status, zeros into temp), which
        # would trip protection checks on garbage.  The profile declares the
        # real width; grouped values are expanded pair-wise
        # [g0, g0, g1, g1, g2, g2] so downstream consumers stay 6-wide.
        statuses = self._read_grouped(Register.STATUS, "status_registers")
        temps = self._read_grouped(Register.TEMP, "temperature_registers")
        return RH56Feedback(
            position=[int(v) for v in positions],
            force_g=[float(to_signed_16(v)) for v in forces],
            current_ma=[float(to_signed_16(v)) for v in currents],
            status_bits=[int(v) for v in statuses],
            temperature_c=[float(v) for v in temps],
            timestamp_monotonic_ns=time.monotonic_ns(),
        )

    def _read_grouped(self, start_addr: int, profile_key: str) -> list[int]:
        """Read STATUS/TEMP honoring the firmware's grouped register width."""
        width = int(self.profile.feedback.get(profile_key, 6) or 6)
        n = self.profile.actuator_count
        if width == n:
            return self._read_registers(start_addr, n)
        if width <= 0 or n % width != 0:
            raise TransportIOError(
                f"feedback_width_invalid: {profile_key}={width} does not divide actuator_count={n}"
            )
        grouped = self._read_registers(start_addr, width)
        repeat = n // width
        return [v for g in grouped for v in [g] * repeat]

    @property
    def last_command_delivery(self) -> str:
        """Delivery classification of the most recent write_position call."""
        return self._last_command_delivery

    def read_angle_setpoints(self) -> list[int]:
        """Read the current ANGLE_SET registers (the active setpoints)."""
        from rosclaw.body.rh56.modbus import Register

        if not self.is_connected():
            raise TransportIOError("serial_timeout: transport not connected")
        return self._read_registers(Register.ANGLE_SET, self.profile.actuator_count)

    def write_position(
        self,
        positions: list[int],
        *,
        speed: int,
        force_limit: int,
    ) -> bool:
        """Send one position command with read-before-retry delivery semantics.

        Returns True only when the device acknowledged the write.  On write
        timeout the actual position is re-read: if it already matches the
        target (within tolerance) delivery is ``acknowledged`` (the command
        went through but the ACK was lost); otherwise delivery is
        ``uncertain`` and the call fails closed WITHOUT re-sending.
        """
        from rosclaw.body.rh56.modbus import (
            ANGLE_MAX,
            ANGLE_MIN,
            FORCE_MAX,
            SPEED_MAX,
            Register,
        )

        if len(positions) != self.profile.actuator_count:
            raise TransportIOError(
                f"actuator_count_mismatch: got {len(positions)} positions, "
                f"expected {self.profile.actuator_count}"
            )
        clamped = [self.profile.clamp_position(p) for p in positions]
        clamped = [max(ANGLE_MIN, min(ANGLE_MAX, p)) for p in clamped]
        speed = max(0, min(SPEED_MAX, int(speed)))
        force_limit = max(0, min(FORCE_MAX, int(force_limit)))

        try:
            self._write_registers(Register.SPEED_SET, [speed] * 6)
            self._write_registers(Register.FORCE_SET, [force_limit] * 6)
            self._write_registers(Register.ANGLE_SET, clamped)
        except (TransportIOError, RuntimeError) as exc:
            logger.warning("write_position uncertain (write path failed): %s", exc)
            return self._classify_uncertain_write(clamped)

        # Read-back verify: ACK received, confirm the setpoint actually landed.
        try:
            setpoints = self._read_registers(Register.ANGLE_SET, 6)
        except (TransportIOError, RuntimeError) as exc:
            logger.warning("write_position read-back failed: %s", exc)
            return self._classify_uncertain_write(clamped)
        if [int(v) for v in setpoints] == clamped:
            self._last_command_delivery = CommandDelivery.ACKNOWLEDGED
            return True
        logger.warning(
            "write_position rejected by device: commanded %s, setpoints read back %s",
            clamped,
            setpoints,
        )
        self._last_command_delivery = CommandDelivery.REJECTED
        return False

    def _classify_uncertain_write(self, target: list[int]) -> bool:
        """Read-before-retry classification (never blindly re-send)."""
        try:
            state = self.read_state()
        except TransportIOError:
            self._last_command_delivery = CommandDelivery.UNCERTAIN
            return False
        if all(
            abs(actual - want) <= self._DELIVERY_TOLERANCE
            for actual, want in zip(state.position, target, strict=True)
        ):
            self._last_command_delivery = CommandDelivery.ACKNOWLEDGED
            return True
        self._last_command_delivery = CommandDelivery.UNCERTAIN
        return False

    def emergency_stop(self) -> bool:
        """Best-effort freeze: re-command the current actual positions."""
        from rosclaw.body.rh56.modbus import Register

        try:
            state = self.read_state()
        except TransportIOError:
            return False
        try:
            self._write_registers(Register.SPEED_SET, [0] * 6)
            self._write_registers(Register.ANGLE_SET, [int(p) for p in state.position])
            return True
        except (TransportIOError, RuntimeError):
            return False


# Keep the original symbol available for tests that assert the stub error.
_SerialModbusTransportStub = SerialModbusTransport
