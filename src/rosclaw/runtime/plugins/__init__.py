"""Default runtime doctor plugins."""

from __future__ import annotations

from rosclaw.runtime.plugins.dexhand_doctor import DexHandDoctor
from rosclaw.runtime.plugins.realsense_doctor import RealSenseDoctor
from rosclaw.runtime.plugins.unitree_doctor import UnitreeDoctor
from rosclaw.runtime.plugins.ur_doctor import URDoctor

__all__ = ["DexHandDoctor", "RealSenseDoctor", "UnitreeDoctor", "URDoctor"]
