"""RH56 actuator naming and canonical orders.

The RS485/Modbus-RTU interface drives 6 linear actuators.  The CAN 2.0B
interface exposes 11 joints (thumb 3 DOF + 4 fingers x 2 DOF).  These are
different devices with different value ranges and MUST NOT be interchanged.
"""

from __future__ import annotations

from enum import StrEnum


class RH56Actuator(StrEnum):
    """The 6 actuators of the RS485 RH56 interface."""

    LITTLE = "little"
    RING = "ring"
    MIDDLE = "middle"
    INDEX = "index"
    THUMB = "thumb"
    THUMB_ROT = "thumb_rot"


# Canonical action order for the RS485 0-1000 raw interface.
RS485_ACTUATOR_ORDER: tuple[str, ...] = (
    RH56Actuator.LITTLE.value,
    RH56Actuator.RING.value,
    RH56Actuator.MIDDLE.value,
    RH56Actuator.INDEX.value,
    RH56Actuator.THUMB.value,
    RH56Actuator.THUMB_ROT.value,
)

# Canonical joint order for the CAN 2.0B 0-65535 interface (11 joints).
CAN_JOINT_ORDER: tuple[str, ...] = (
    "thumb_rot",
    "thumb_proximal",
    "thumb_distal",
    "index_proximal",
    "index_distal",
    "middle_proximal",
    "middle_distal",
    "ring_proximal",
    "ring_distal",
    "little_proximal",
    "little_distal",
)

# RS485 raw position convention: 0 = fully closed, 1000 = fully open.
RS485_POSITION_RANGE: tuple[int, int] = (0, 1000)
RS485_SPEED_RANGE: tuple[int, int] = (0, 1000)
RS485_FORCE_RANGE: tuple[int, int] = (0, 1000)

# CAN raw position convention: 0 = fully closed, 65535 = fully open.
CAN_POSITION_RANGE: tuple[int, int] = (0, 65535)
CAN_FORCE_RANGE: tuple[int, int] = (0, 4095)
CAN_CURRENT_RANGE: tuple[int, int] = (0, 4095)
