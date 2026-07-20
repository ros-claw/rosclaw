"""RH56 transport backends.

``RH56Transport`` is the minimal interface every backend must implement:

- :meth:`read_state` — position/force/current/status/temperature per actuator
- :meth:`write_position` — one position command (single-step execution only)
- :meth:`emergency_stop` — best-effort stop command

Two backends are provided:

- :class:`MockModbusTransport` — a fully in-memory simulated device used by
  tests and by the mock shadow gate.  It models first-order actuator motion,
  contact force, current draw, temperature drift and status bits.
- :class:`SerialModbusTransport` — the real RS485/Modbus-RTU backend.  It is
  fail-closed: construction without an existing device path raises
  ``TransportUnavailableError``.  The frame-level implementation lands with
  the real hardware bring-up (Experiment 0).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from rosclaw.body.rh56.constants import RS485_ACTUATOR_ORDER
from rosclaw.body.rh56.transport_profile import TransportProfile


class TransportUnavailableError(RuntimeError):
    """Raised when a transport backend cannot be used (missing device, etc.)."""


class TransportIOError(RuntimeError):
    """Raised on I/O failure during a transport transaction."""


class CommandDelivery(str, Enum):
    """Tri-state delivery outcome for one command (review §1).

    A plain boolean cannot express "the command may have executed but the
    response was lost on the wire".  Drivers must return ``UNCERTAIN`` in
    that case; the executor then re-reads the actual position before
    deciding anything — it never blindly re-sends a position command.
    """

    ACKNOWLEDGED = "acknowledged"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


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
    def zero(cls, count: int = 6) -> "RH56Feedback":
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

    def connect(self) -> None: ...

    def close(self) -> None: ...

    def is_connected(self) -> bool: ...

    def read_state(self) -> RH56Feedback: ...

    def write_position(
        self,
        positions: list[int],
        *,
        speed: int,
        force_limit: int,
    ) -> CommandDelivery:
        """Send one position command and report the delivery outcome.

        ``ACKNOWLEDGED`` — the device confirmed the command.
        ``REJECTED`` — the device explicitly refused it.
        ``UNCERTAIN`` — the response was lost/ambiguous; the caller must
        re-read actual positions before any retry decision.
        """

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
        self._uncertain_next_write = False  # test hook: next write loses its response

    # -- test hooks -----------------------------------------------------

    def set_contact(self, actuator_index: int, contact_position: int, force_g: float = 120.0) -> None:
        self._specs[actuator_index].contact_position = contact_position
        self._specs[actuator_index].contact_force_g = force_g

    def fail_next_read(self) -> None:
        self._io_fail_next = True

    def drop_connection(self) -> None:
        self._disconnect_next = True

    def lose_next_write_response(self) -> None:
        """Simulate a write whose response is lost on the wire (UNCERTAIN)."""
        self._uncertain_next_write = True

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
                self._force_g[i] = min(spec.contact_force_g, spec.contact_force_g * (0.5 + overshoot / 20.0))
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

    def write_position(
        self, positions: list[int], *, speed: int, force_limit: int
    ) -> CommandDelivery:
        if not self._connected:
            raise TransportIOError("serial_timeout: transport not connected")
        if len(positions) != self._count:
            raise TransportIOError(
                f"actuator_count_mismatch: got {len(positions)} positions, expected {self._count}"
            )
        if self._estopped:
            return CommandDelivery.REJECTED
        lo, hi = self.profile.position_range
        self._target = [float(max(lo, min(hi, int(p)))) for p in positions]
        self._speed = int(speed)
        self._force_limit = int(force_limit)
        if self._uncertain_next_write:
            # The command WAS applied to the mock device, but the response is
            # lost — exactly the ambiguous case the tri-state exists for.
            self._uncertain_next_write = False
            return CommandDelivery.UNCERTAIN
        return CommandDelivery.ACKNOWLEDGED

    def emergency_stop(self) -> bool:
        self._estopped = True
        self._target = list(self._position)
        self._force_g = [0.0] * self._count
        self._current_ma = [0.0] * self._count
        return True


# ---------------------------------------------------------------------------
# Real serial backend (fail-closed stub until hardware bring-up)
# ---------------------------------------------------------------------------


class SerialModbusTransport:
    """RS485/Modbus-RTU backend for the real RH56.

    Fail-closed by design: if the configured device path does not exist this
    raises :class:`TransportUnavailableError` at construction.  The Modbus
    frame implementation (function codes 0x03/0x06/0x10 over the Inspire
    register map) is completed during Experiment 0 with the physical hand.
    """

    def __init__(self, profile: TransportProfile):
        self.profile = profile
        device = profile.transport.device
        if not device or not os.path.exists(device):
            raise TransportUnavailableError(
                f"device_path_disappeared: {device or '<unset>'} not available; "
                "SerialModbusTransport requires the physical RH56 device"
            )
        raise TransportUnavailableError(
            "SerialModbusTransport frame implementation is pending hardware bring-up "
            "(Experiment 0); use MockModbusTransport for validation"
        )
