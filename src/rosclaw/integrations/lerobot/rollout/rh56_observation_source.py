"""RH56 observation source for shadow/execute rollouts.

Reads position/force/current/status/temperature from an ``RH56Transport``
(mock or serial) and returns LeRobot-style flat observations plus the RH56
feedback channels required by the P5 shadow evidence list (plan §6.3).
"""

from __future__ import annotations

import time
from typing import Any

from rosclaw.body.rh56.transport import RH56Transport, TransportIOError
from rosclaw.body.rh56.transport_profile import TransportProfile
from rosclaw.integrations.lerobot.rollout.observation_source import ObservationSource


class RH56ObservationSource(ObservationSource):
    """Observation source backed by an RH56 transport.

    Each ``get_observation`` call performs one ``read_state`` transaction and
    returns::

        {
          "observation.state": [6 raw positions],
          "observation.force": [6 grams],
          "observation.current": [6 mA],
          "observation.temperature": [6 °C],
          "observation.status": [6 status bitfields],
          "state_names": profile.action_order,
          "task": task,
          "observation_age_ms": age of the feedback snapshot,
        }

    Serial health is tracked for the shadow report: an I/O failure raises
    :class:`TransportIOError` so the rollout loop can record the fault and
    stop; failures and disconnects are counted for the report.
    """

    def __init__(
        self,
        transport: RH56Transport,
        profile: TransportProfile,
        *,
        task: str = "hold_current",
    ):
        self.transport = transport
        self.profile = profile
        self.task = task
        self.read_failure_count = 0
        self.disconnect_count = 0
        self.read_count = 0

    def get_observation(self, step_index: int) -> dict[str, Any] | None:
        try:
            feedback = self.transport.read_state()
        except TransportIOError:
            self.read_failure_count += 1
            if not self.transport.is_connected():
                self.disconnect_count += 1
            raise

        self.read_count += 1
        age_ms = max(0, (time.monotonic_ns() - feedback.timestamp_monotonic_ns) / 1e6)
        return {
            "observation.state": [float(p) for p in feedback.position],
            "observation.force": list(feedback.force_g),
            "observation.current": list(feedback.current_ma),
            "observation.temperature": list(feedback.temperature_c),
            "observation.status": list(feedback.status_bits),
            "state_names": list(self.profile.action_order),
            "task": self.task,
            "observation_age_ms": age_ms,
        }

    def is_exhausted(self, step_index: int) -> bool:
        return False

    def serial_health(self) -> dict[str, Any]:
        return {
            "read_count": self.read_count,
            "read_failure_count": self.read_failure_count,
            "disconnect_count": self.disconnect_count,
            "connected": self.transport.is_connected(),
        }

    def close(self) -> None:
        self.transport.close()
