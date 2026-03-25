"""Action Parser - Convert VLA model output to robot actions."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ActionSequence:
    """Structured action sequence for robot control."""

    positions: np.ndarray
    """End-effector positions [horizon, 3]."""

    orientations: np.ndarray
    """End-effector orientations [horizon, 3] (roll, pitch, yaw)."""

    gripper_states: np.ndarray
    """Gripper states [horizon] (0=closed, 1=open)."""

    timestamps: np.ndarray
    """Relative timestamps for each action [horizon]."""

    metadata: dict | None = None
    """Additional metadata."""

    def __post_init__(self):
        """Validate shapes."""
        horizon = self.positions.shape[0]
        assert self.orientations.shape[0] == horizon
        assert self.gripper_states.shape[0] == horizon
        assert self.timestamps.shape[0] == horizon


class ActionParser:
    """Parser for VLA model outputs.

    Handles various output formats and converts them to
    standardized robot action sequences.
    """

    # Standard action indices
    ACTION_INDICES = {
        "x": 0,
        "y": 1,
        "z": 2,
        "roll": 3,
        "pitch": 4,
        "yaw": 5,
        "gripper": 6,
    }

    def __init__(
        self,
        action_dim: int = 7,
        action_horizon: int = 16,
        control_mode: Literal["position", "velocity", "effort"] = "position",
    ):
        """Initialize action parser.

        Args:
            action_dim: Dimensionality of action space.
            action_horizon: Number of actions in sequence.
            control_mode: Robot control mode.
        """
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.control_mode = control_mode

    def parse(
        self,
        raw_output: str | np.ndarray,
        horizon: int | None = None,
    ) -> np.ndarray:
        """Parse raw VLA output to action sequence.

        Args:
            raw_output: Raw model output (text or array).
            horizon: Expected action horizon.

        Returns:
            Action array [horizon, action_dim].
        """
        horizon = horizon or self.action_horizon

        if isinstance(raw_output, np.ndarray):
            return self._parse_array(raw_output, horizon)
        elif isinstance(raw_output, str):
            return self._parse_text(raw_output, horizon)
        else:
            raise ValueError(f"Unsupported output type: {type(raw_output)}")

    def _parse_array(
        self,
        array: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """Parse numpy array output."""
        # Ensure correct shape
        if array.ndim == 1:
            # Single action repeated
            actions = np.tile(array, (horizon, 1))
        elif array.ndim == 2:
            if array.shape[0] >= horizon:
                actions = array[:horizon]
            else:
                # Pad with last action
                padding = np.tile(array[-1], (horizon - array.shape[0], 1))
                actions = np.vstack([array, padding])
        else:
            raise ValueError(f"Unexpected array shape: {array.shape}")

        # Ensure correct action dimension
        if actions.shape[1] < self.action_dim:
            padding = np.zeros((actions.shape[0], self.action_dim - actions.shape[1]))
            actions = np.hstack([actions, padding])
        elif actions.shape[1] > self.action_dim:
            actions = actions[:, :self.action_dim]

        return actions

    def _parse_text(self, text: str, horizon: int) -> np.ndarray:
        """Parse text output to extract actions."""
        actions = []

        # Try different parsing strategies

        # 1. Look for JSON array format
        json_actions = self._try_parse_json(text)
        if json_actions is not None:
            actions = json_actions

        # 2. Look for bracket notation [x, y, z, ...]
        if not actions:
            bracket_actions = self._try_parse_brackets(text)
            if bracket_actions is not None:
                actions = bracket_actions

        # 3. Look for space-separated numbers
        if not actions:
            space_actions = self._try_parse_space_separated(text)
            if space_actions is not None:
                actions = space_actions

        # 4. Fallback: extract any numbers
        if not actions:
            numbers = self._extract_all_numbers(text)
            if len(numbers) >= self.action_dim:
                actions = numbers

        if not actions:
            logger.warning(f"Could not parse actions from: {text[:200]}")
            return np.zeros((horizon, self.action_dim))

        # Convert to numpy and reshape
        actions_array = np.array(actions, dtype=np.float32)

        return self._parse_array(actions_array, horizon)

    def _try_parse_json(self, text: str) -> list[float] | None:
        """Try to parse JSON array from text."""
        try:
            import json
            # Find JSON array pattern
            match = re.search(r'\[[\s\d.,\-\[\]]+\]', text)
            if match:
                data = json.loads(match.group())
                # Flatten if nested
                if isinstance(data[0], list):
                    return [item for sublist in data for item in sublist]
                return data
        except (json.JSONDecodeError, IndexError):
            pass
        return None

    def _try_parse_brackets(self, text: str) -> list[float] | None:
        """Try to parse bracket notation [x, y, z, ...]."""
        # Match patterns like [0.1, -0.2, 0.3, ...]
        patterns = [
            r'\[\s*(-?\d+\.?\d*(?:,\s*-?\d+\.?\d*)*)\s*\]',
            r'action[s]?\s*[:=]?\s*\[([^\]]+)\]',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                numbers_str = match.group(1)
                numbers = [float(n) for n in re.findall(r'-?\d+\.?\d*', numbers_str)]
                if numbers:
                    return numbers

        return None

    def _try_parse_space_separated(self, text: str) -> list[float] | None:
        """Try to parse space-separated numbers after action keywords."""
        # Look for lines starting with action-related keywords
        keywords = ['action', 'move', 'position', 'velocity', 'command']
        lines = text.split('\n')

        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in keywords):
                numbers = [float(n) for n in re.findall(r'-?\d+\.?\d*', line)]
                if len(numbers) >= self.action_dim:
                    return numbers

        return None

    def _extract_all_numbers(self, text: str) -> list[float]:
        """Extract all numbers from text as fallback."""
        return [float(n) for n in re.findall(r'-?\d+\.?\d*', text)]

    def to_action_sequence(self, actions: np.ndarray) -> ActionSequence:
        """Convert action array to structured sequence.

        Args:
            actions: Action array [horizon, action_dim].

        Returns:
            Structured ActionSequence.
        """
        horizon = actions.shape[0]

        # Extract components
        positions = actions[:, :3]  # x, y, z

        if actions.shape[1] >= 6:
            orientations = actions[:, 3:6]  # roll, pitch, yaw
        else:
            orientations = np.zeros((horizon, 3))

        if actions.shape[1] >= 7:
            gripper_states = actions[:, 6]  # gripper
        else:
            gripper_states = np.zeros(horizon)

        # Generate timestamps (assume 10Hz control)
        timestamps = np.arange(horizon) * 0.1

        return ActionSequence(
            positions=positions,
            orientations=orientations,
            gripper_states=gripper_states,
            timestamps=timestamps,
        )

    def format_for_robot(self, actions: np.ndarray) -> dict:
        """Format actions for robot control.

        Args:
            actions: Action array [horizon, action_dim].

        Returns:
            Dictionary formatted for robot interface.
        """
        action_seq = self.to_action_sequence(actions)

        return {
            "control_mode": self.control_mode,
            "positions": action_seq.positions.tolist(),
            "orientations": action_seq.orientations.tolist(),
            "gripper_states": action_seq.gripper_states.tolist(),
            "timestamps": action_seq.timestamps.tolist(),
            "action_horizon": actions.shape[0],
        }

    def interpolate_actions(
        self,
        actions: np.ndarray,
        target_horizon: int,
    ) -> np.ndarray:
        """Interpolate actions to different horizon length.

        Args:
            actions: Original actions [horizon, action_dim].
            target_horizon: Desired output horizon.

        Returns:
            Interpolated actions [target_horizon, action_dim].
        """
        from scipy import interpolate

        current_horizon = actions.shape[0]
        x = np.linspace(0, 1, current_horizon)
        x_new = np.linspace(0, 1, target_horizon)

        interpolated = np.zeros((target_horizon, actions.shape[1]))
        for i in range(actions.shape[1]):
            f = interpolate.interp1d(x, actions[:, i], kind='cubic')
            interpolated[:, i] = f(x_new)

        return interpolated
