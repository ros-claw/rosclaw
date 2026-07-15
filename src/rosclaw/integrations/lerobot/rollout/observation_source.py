"""Observation sources for proposal-only and shadow rollouts."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from rosclaw.integrations.lerobot.contracts import ObservationContract
from rosclaw.sense.interface import SenseInterface


class ObservationSource(ABC):
    """Abstract source of observations for a rollout loop."""

    @abstractmethod
    def get_observation(self, step_index: int) -> dict[str, Any] | None:
        """Return the observation dict for the given step, or None if exhausted."""

    @abstractmethod
    def is_exhausted(self, step_index: int) -> bool:
        """Return True when no more observations are available."""

    def close(self) -> None:
        """Release any resources."""


class FixtureObservationSource(ObservationSource):
    """Load observations from a JSON fixture file.

    Supported shapes:
    - A list of observation dicts.
    - A dict with ``observations`` key containing a list.
    - A list of PracticeEventEnvelope dicts; ``observation.state`` events are
      extracted when ``event_type`` starts with ``rollout.observation`` or
      contains ``observation``.
    """

    def __init__(self, fixture_path: str | Path) -> None:
        self.fixture_path = Path(fixture_path)
        raw = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        self.observations: list[dict[str, Any]] = []
        if isinstance(raw, list):
            for item in raw:
                obs = self._extract_observation(item)
                if obs is not None:
                    self.observations.append(obs)
        elif isinstance(raw, dict):
            observations = raw.get("observations", [])
            for item in observations:
                obs = self._extract_observation(item)
                if obs is not None:
                    self.observations.append(obs)
        if not self.observations:
            raise ValueError(f"No observations found in fixture: {self.fixture_path}")

    @staticmethod
    def _extract_observation(item: dict[str, Any]) -> dict[str, Any] | None:
        if "event_type" in item and isinstance(item.get("payload"), dict):
            payload = item["payload"]
            if "snapshot" in payload:
                return payload["snapshot"]
            if "observation" in payload:
                return payload["observation"]
        return item

    def get_observation(self, step_index: int) -> dict[str, Any] | None:
        if step_index < 0 or step_index >= len(self.observations):
            return None
        return self.observations[step_index]

    def is_exhausted(self, step_index: int) -> bool:
        return step_index >= len(self.observations)


class BodyObservationSource(ObservationSource):
    """Read live joint positions from the effective body via SenseInterface.

    The observation is returned as a LeRobot-style flat dict:
    ``{"observation.state": [...], "task": "..."}``.  Joint ordering follows
    the provided ``ObservationContract`` state names, falling back to the body
    joint order.
    """

    def __init__(
        self,
        robot_id: str = "rosclaw_default",
        collector: str = "mock",
        scenario: str = "normal",
        contract: ObservationContract | dict[str, Any] | None = None,
        body_joint_names: list[str] | None = None,
    ):
        self.interface = SenseInterface(
            robot_id=robot_id,
            collector=collector,
            scenario=scenario,
        )
        self.interface.initialize()
        self.contract: ObservationContract | None = None
        if isinstance(contract, dict):
            self.contract = ObservationContract.from_dict(contract)
        else:
            self.contract = contract
        self.body_joint_names = body_joint_names or []

    def get_observation(self, step_index: int) -> dict[str, Any] | None:
        state = self.interface.get_body_state()
        joint_positions: list[float] = []
        names = self.contract.get_state_names() if self.contract else None
        if not names:
            names = self.body_joint_names
        if names:
            for name in names:
                joint = state.joints.get(name, {})
                if isinstance(joint, dict):
                    pos = joint.get("position_rad", 0.0)
                else:
                    pos = getattr(joint, "position_rad", 0.0) or 0.0
                joint_positions.append(float(pos))
        else:
            for joint in state.joints.values():
                if isinstance(joint, dict):
                    pos = joint.get("position_rad", 0.0)
                else:
                    pos = getattr(joint, "position_rad", 0.0) or 0.0
                joint_positions.append(float(pos))
        return {
            "observation.state": joint_positions,
            "task": "",
        }

    def is_exhausted(self, step_index: int) -> bool:
        return False

    def close(self) -> None:
        self.interface.stop()
