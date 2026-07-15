"""Adapt ROSClaw observation input into a worker observation dict.

The output follows LeRobot-style flat keys:

- ``task``: optional text task description.
- ``observation.state``: list of floats.
- ``observation.images.<name>``: path to image file on disk.

This module must not import torch or lerobot.

P4 contract semantics:

- State order must be deterministic. If ``state`` is a list, ``state_names`` or
  an explicit ``ObservationContract`` must be supplied to name/order the joints.
- Dict ``state`` is only accepted when an ``ObservationContract`` defines the
  expected feature names and their order.
- Missing names are a fail-closed error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rosclaw.integrations.lerobot.contracts import ObservationContract


def adapt_observation_for_worker(
    input_data: dict[str, Any],
    contract: ObservationContract | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a ROSClaw provider input dict into a worker observation dict.

    Supported ``input_data`` shapes:

    1. LeRobot-style flat dict::

        {
          "task": "pick the red cube",
          "observation.state": [0.1, 0.2, ...],
          "observation.images.front": "/tmp/front.jpg"
        }

    2. ROSClaw nested dict with explicit state names::

        {
          "observation": {
            "task": "pick the red cube",
            "state": [0.1, 0.2, ...],
            "state_names": ["joint_a", "joint_b"],
            "images": {"front": "/tmp/front.jpg"}
          }
        }

    Raises:
        ValueError: if state ordering/naming cannot be determined.
        FileNotFoundError: if a referenced image file does not exist.
    """
    parsed_contract = None
    if contract is not None:
        parsed_contract = (
            ObservationContract.from_dict(contract)
            if isinstance(contract, dict)
            else contract
        )

    observation = input_data.get("observation", input_data)
    if not isinstance(observation, dict):
        raise ValueError(f"Expected observation dict, got {type(observation).__name__}")

    out: dict[str, Any] = {}

    # Task
    task = observation.get("task", input_data.get("task", ""))
    if task:
        out["task"] = str(task)

    # State: support both flat and nested keys.
    state = _extract_state(observation, parsed_contract)
    if state is not None:
        out["observation.state"] = state

    # Images: support flat keys and nested images dict.
    images = _extract_images(observation)
    for name, path in images.items():
        image_path = Path(path)
        if not image_path.exists():
            raise FileNotFoundError(f"Observation image not found: {image_path}")
        out[f"observation.images.{name}"] = str(image_path.resolve())

    return out


def _extract_state(
    observation: dict[str, Any],
    contract: ObservationContract | None,
) -> list[float] | None:
    if "observation.state" in observation:
        values = observation["observation.state"]
        if isinstance(values, list):
            return [float(v) for v in values]
        raise ValueError("observation.state must be a list of floats")

    state = observation.get("state")
    if state is None:
        return None

    if isinstance(state, list):
        names = _resolve_state_names(observation, contract)
        if names is None:
            raise ValueError(
                "observation.state is a list but no state_names or ObservationContract "
                "was provided; joint ordering is ambiguous."
            )
        if len(state) != len(names):
            raise ValueError(
                f"observation.state length ({len(state)}) does not match "
                f"state_names length ({len(names)})."
            )
        return [float(v) for v in state]

    if isinstance(state, dict):
        names = _resolve_state_names(observation, contract)
        if names is None:
            raise ValueError(
                "observation.state is a dict but no ObservationContract was provided; "
                "joint ordering is ambiguous."
            )
        missing = [n for n in names if n not in state]
        if missing:
            raise ValueError(
                f"observation.state keys missing expected joints: {missing}"
            )
        return [float(state[n]) for n in names]

    return [float(state)]


def _resolve_state_names(
    observation: dict[str, Any],
    contract: ObservationContract | None,
) -> list[str] | None:
    """Return an explicit ordering of state joint names if available."""
    if contract is not None:
        names = contract.get_state_names()
        if names:
            return names

    explicit_names = observation.get("state_names")
    if isinstance(explicit_names, list) and explicit_names:
        return [str(n) for n in explicit_names]

    return None


def _extract_images(observation: dict[str, Any]) -> dict[str, str]:
    images: dict[str, str] = {}

    # Flat keys.
    for key, value in observation.items():
        if key.startswith("observation.images."):
            name = key.split(".", 2)[2]
            images[name] = str(value)

    # Nested dict.
    nested = observation.get("images")
    if isinstance(nested, dict):
        for name, path in nested.items():
            if name not in images:
                images[name] = str(path)

    return images
