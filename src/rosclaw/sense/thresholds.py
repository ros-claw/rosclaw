"""Default thresholds and robot-specific threshold loading for rosclaw.sense."""

from __future__ import annotations

import logging
import os
from typing import Any

import yaml

logger = logging.getLogger("rosclaw.sense.thresholds")

DEFAULT_SENSE_THRESHOLDS: dict[str, Any] = {
    "battery": {
        "low": 25.0,
        "critical": 10.0,
    },
    "joint_temperature_c": {
        "warm": 65.0,
        "hot": 75.0,
        "overheat": 85.0,
    },
    "dds_latency_ms": {
        "degraded": 50.0,
        "bad": 100.0,
    },
    "tracking_error": {
        "degraded": 0.10,
        "bad": 0.25,
    },
    "support_margin": {
        "low": 0.12,
        "ok": 0.18,
    },
    "perception": {
        "target_confidence_min": 0.75,
        "camera_fps_min": 15.0,
    },
    "compute": {
        "cpu_usage_high": 85.0,
        "cpu_usage_critical": 95.0,
        "memory_usage_high": 85.0,
        "memory_usage_critical": 95.0,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base without mutating base."""
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_thresholds(
    robot_family: str | None = None,
    override_path: str | None = None,
) -> dict[str, Any]:
    """Load sense thresholds.

    Priority (highest first):
      1. ``override_path`` YAML file
      2. Robot-specific config file under ``src/rosclaw/sense/configs/``
      3. ``DEFAULT_SENSE_THRESHOLDS``

    Args:
        robot_family: Optional robot family name (e.g. ``unitree_g1``).
        override_path: Optional path to a YAML file with threshold overrides.

    Returns:
        A dictionary of thresholds suitable for estimators.
    """
    thresholds = dict(DEFAULT_SENSE_THRESHOLDS)

    if robot_family:
        package_dir = os.path.dirname(__file__)
        robot_path = os.path.join(package_dir, "configs", f"{robot_family}.yaml")
        if os.path.exists(robot_path):
            try:
                with open(robot_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                robot_thresholds = data.get("thresholds", {})
                thresholds = _deep_merge(thresholds, robot_thresholds)
                logger.debug("Loaded thresholds for robot family %s", robot_family)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to load robot thresholds %s: %s", robot_path, e)

    if override_path and os.path.exists(override_path):
        try:
            with open(override_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            override_thresholds = data.get("thresholds", data)
            thresholds = _deep_merge(thresholds, override_thresholds)
            logger.debug("Loaded threshold override from %s", override_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load threshold override %s: %s", override_path, e)

    return thresholds


def get_capability_requirements(
    robot_family: str | None = None,
    override_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load capability readiness requirements for a robot family.

    Returns a mapping from capability name to requirement dict, e.g.::

        {
            "kick_ball": {
                "battery_percent_min": 40.0,
                "max_leg_joint_temp_c": 72.0,
                ...
            }
        }
    """
    capabilities: dict[str, dict[str, Any]] = {}

    if robot_family:
        package_dir = os.path.dirname(__file__)
        robot_path = os.path.join(package_dir, "configs", f"{robot_family}.yaml")
        if os.path.exists(robot_path):
            try:
                with open(robot_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                capabilities = dict(data.get("capabilities", {}))
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to load capabilities %s: %s", robot_path, e)

    if override_path and os.path.exists(override_path):
        try:
            with open(override_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            override_capabilities = data.get("capabilities", {})
            for name, reqs in override_capabilities.items():
                capabilities.setdefault(name, {})
                capabilities[name].update(reqs)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load capability override %s: %s", override_path, e)

    return capabilities
