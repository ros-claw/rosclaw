"""ROSClaw ↔ LeRobot feature mapping."""

from __future__ import annotations

from typing import Any

ROSCLAW_TO_LEROBOT_OBS = {
    "camera.front.rgb": "observation.images.front",
    "camera.wrist.rgb": "observation.images.wrist",
    "robot.joint.position": "observation.state",
    "robot.ee.pose": "observation.ee_pose",
    "task.language": "task",
}

ROSCLAW_TO_LEROBOT_ACTION = {
    "action.joint_position": "action",
    "action.ee_delta_pose": "action",
}


def flatten_observation(obs: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested ROSClaw observation into LeRobot-style flat keys.

    Example input:
        {
          "task": "make OK gesture",
          "observation": {
            "state": [0.0, 0.1, 0.2],
            "images": {"front": "examples/lerobot/front.png"}
          }
        }

    Example output:
        {
          "task": "make OK gesture",
          "observation.state": [0.0, 0.1, 0.2],
          "observation.images.front": "examples/lerobot/front.png"
        }
    """
    flat: dict[str, Any] = {}

    task = obs.get("task") or obs.get("agent.task")
    if task is not None:
        flat["task"] = task

    observation = obs.get("observation")
    if isinstance(observation, dict):
        state = observation.get("state")
        if state is not None:
            flat["observation.state"] = state
        ee_pose = observation.get("ee_pose")
        if ee_pose is not None:
            flat["observation.ee_pose"] = ee_pose
        images = observation.get("images")
        if isinstance(images, dict):
            for camera_name, value in images.items():
                flat[f"observation.images.{camera_name}"] = value

    for key in ("robot", "sandbox", "failure", "human"):
        if key in obs:
            flat[key] = obs[key]

    return flat
