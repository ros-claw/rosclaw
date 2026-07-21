"""Supervisory runtime health states exposed by rosclawd."""

from enum import StrEnum


class SupervisionState(StrEnum):
    STARTING = "STARTING"
    READY = "READY"
    ARMED = "ARMED"
    DEGRADED = "DEGRADED"
    DISARMED = "DISARMED"
    ESTOPPED = "ESTOPPED"
    FAILED = "FAILED"
    STOPPING = "STOPPING"


__all__ = ["SupervisionState"]
