"""Fixture-replay transport: replay recorded RH56 frames through the stack.

Golden fixtures recorded from the real device during Experiment 0 let the
full shadow/execute pipeline run in CI without hardware (review §1).  The
fixture format is JSON, one transaction per line, with synthetic examples
checked in under ``tests/fixtures/rh56_modbus/``::

    {"op": "connect"}
    {"op": "read", "position": [1000, ...], "force_g": [...], ...}
    {"op": "write", "positions": [1000, ...], "delivery": "acknowledged"}
    {"op": "read", "position": [980, ...], ...}

``FixtureModbusTransport`` implements the same ``RH56Transport`` contract as
``MockModbusTransport``; write calls check the recorded delivery tri-state.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rosclaw.body.rh56.transport import (
    CommandDelivery,
    RH56Feedback,
    TransportIOError,
)
from rosclaw.body.rh56.transport_profile import TransportProfile


class FixtureModbusTransport:
    """Replay a recorded sequence of transport transactions."""

    def __init__(self, profile: TransportProfile, fixture_path: str | Path):
        self.profile = profile
        self._count = profile.command.actuator_count
        self._frames = self._load(fixture_path)
        self._cursor = 0
        self._connected = False
        self._last_feedback = RH56Feedback.zero(self._count)

    @staticmethod
    def _load(fixture_path: str | Path) -> list[dict[str, Any]]:
        path = Path(fixture_path)
        if not path.exists():
            raise TransportIOError(f"fixture_missing: {path}")
        frames: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                frames.append(json.loads(line))
        if not frames:
            raise TransportIOError(f"fixture_empty: {path}")
        return frames

    # ------------------------------------------------------------------

    def _next(self, expected_op: str) -> dict[str, Any]:
        if self._cursor >= len(self._frames):
            raise TransportIOError(
                f"fixture_exhausted: no more frames (expected {expected_op})"
            )
        frame = self._frames[self._cursor]
        self._cursor += 1
        op = frame.get("op")
        if op != expected_op:
            raise TransportIOError(
                f"fixture_mismatch: expected op {expected_op!r}, got {op!r} "
                f"at frame {self._cursor - 1}"
            )
        return frame

    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._connected = True

    def close(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def read_state(self) -> RH56Feedback:
        if not self._connected:
            raise TransportIOError("serial_timeout: transport not connected")
        frame = self._next("read")
        if frame.get("io_error"):
            raise TransportIOError(f"io_error: {frame.get('io_error')}")
        self._last_feedback = RH56Feedback(
            position=[int(v) for v in frame.get("position", [0] * self._count)],
            force_g=[float(v) for v in frame.get("force_g", [0.0] * self._count)],
            current_ma=[float(v) for v in frame.get("current_ma", [0.0] * self._count)],
            status_bits=[int(v) for v in frame.get("status_bits", [0] * self._count)],
            temperature_c=[float(v) for v in frame.get("temperature_c", [25.0] * self._count)],
            timestamp_monotonic_ns=time.monotonic_ns(),
        )
        return self._last_feedback

    def write_position(
        self, positions: list[int], *, speed: int, force_limit: int
    ) -> CommandDelivery:
        if not self._connected:
            raise TransportIOError("serial_timeout: transport not connected")
        if len(positions) != self._count:
            raise TransportIOError(
                f"actuator_count_mismatch: got {len(positions)} positions, expected {self._count}"
            )
        frame = self._next("write")
        recorded = frame.get("positions")
        if recorded is not None and [int(v) for v in recorded] != [int(v) for v in positions]:
            raise TransportIOError(
                f"fixture_mismatch: write positions {positions} != recorded {recorded}"
            )
        delivery = frame.get("delivery", "acknowledged")
        return CommandDelivery(delivery)

    def emergency_stop(self) -> bool:
        try:
            self._next("estop")
        except TransportIOError:
            pass
        return True
