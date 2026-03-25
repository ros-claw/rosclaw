"""
ROSClaw - The Universal Operating System for Software-Defined Embodied AI.

This package provides production-ready middleware for connecting LLMs to physical robots.
"""

__version__ = "0.1.0"
__author__ = "ROSClaw Team"

from rosclaw.firewall.decorator import (
    DigitalTwinFirewall,
    SafetyLevel,
    SafetyViolationError,
    mujoco_firewall,
)

__all__ = [
    "DigitalTwinFirewall",
    "SafetyLevel",
    "SafetyViolationError",
    "mujoco_firewall",
]