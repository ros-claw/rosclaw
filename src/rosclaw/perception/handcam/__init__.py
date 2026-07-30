"""HandCam perception for the Physical Evolution Lab (PR-PE-3)."""

from .calibration import (
    CONTRACT_VERSION,
    CameraIntrinsics,
    CameraPoseContract,
    intrinsics_from_json,
    intrinsics_from_pyrealsense,
)
from .hand_state import (
    STATE_VERSION,
    ClusterQuality,
    StabilityResult,
    VisualHandState,
    contact_distance_m,
    estimate_hand_state,
    measure_visual_stability,
)

__all__ = [
    "CONTRACT_VERSION",
    "STATE_VERSION",
    "CameraIntrinsics",
    "CameraPoseContract",
    "ClusterQuality",
    "StabilityResult",
    "VisualHandState",
    "contact_distance_m",
    "estimate_hand_state",
    "intrinsics_from_json",
    "intrinsics_from_pyrealsense",
    "measure_visual_stability",
]
