"""RH56 dexterous hand support: transport profiles, calibration, transports.

This package owns the RS485/Modbus-RTU vs CAN protocol separation required by
the P5 implementation plan.  No hardware I/O happens here unless a concrete
transport backend is explicitly constructed.
"""

from rosclaw.body.rh56.calibration import (
    RH56Calibration,
    RH56CalibrationGate,
    load_rh56_calibration,
)
from rosclaw.body.rh56.constants import (
    CAN_JOINT_ORDER,
    RS485_ACTUATOR_ORDER,
    RH56Actuator,
)
from rosclaw.body.rh56.transport_profile import (
    TransportBindingError,
    TransportProfile,
    load_transport_profile,
    validate_transport_binding,
)

__all__ = [
    "CAN_JOINT_ORDER",
    "RS485_ACTUATOR_ORDER",
    "RH56Actuator",
    "RH56Calibration",
    "RH56CalibrationGate",
    "TransportBindingError",
    "TransportProfile",
    "load_rh56_calibration",
    "load_transport_profile",
    "validate_transport_binding",
]
