"""Adapt ROSClaw observation input into a worker observation dict.

The output follows LeRobot-style flat keys:

- ``task``: optional text task description.
- ``observation.state``: list of floats.
- ``observation.images.<name>``: path to image file on disk.

This module must not import torch or lerobot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def adapt_observation_for_worker(input_data: dict[str, Any]) -> dict[str, Any]:
    """Convert a ROSClaw provider input dict into a worker observation dict.

    Supported ``input_data`` shapes:

    1. LeRobot-style flat dict::

        {
          "task": "pick the red cube",
          "observation.state": [0.1, 0.2, ...],
          "observation.images.front": "/tmp/front.jpg"
        }

    2. ROSClaw nested dict::

        {
          "observation": {
            "task": "pick the red cube",
            "state": [0.1, 0.2, ...],
            "images": {"front": "/tmp/front.jpg"}
          }
        }
    """
    observation = input_data.get("observation", input_data)
    if not isinstance(observation, dict):
        raise ValueError(f"Expected observation dict, got {type(observation).__name__}")
    base_dir_raw = input_data.get("_base_dir") or observation.get("_base_dir")
    base_dir = Path(base_dir_raw) if base_dir_raw else None

    out: dict[str, Any] = {}

    # Task
    task = observation.get("task", input_data.get("task", ""))
    if task:
        out["task"] = str(task)

    # State: support both flat and nested keys.
    state = _extract_state(observation)
    if state is not None:
        out["observation.state"] = state

    # Images: support flat keys and nested images dict.
    images = _extract_images(observation)
    for name, path in images.items():
        image_path = Path(path)
        if base_dir is not None and not image_path.is_absolute():
            image_path = base_dir / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Observation image not found: {image_path}")
        out[f"observation.images.{name}"] = str(image_path.resolve())

    return out


def _extract_state(observation: dict[str, Any]) -> list[float] | None:
    if "observation.state" in observation:
        return [float(v) for v in observation["observation.state"]]

    state = observation.get("state")
    if state is None:
        return None
    if isinstance(state, list):
        return [float(v) for v in state]
    if isinstance(state, dict):
        return [float(v) for v in state.values()]
    return [float(state)]


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
