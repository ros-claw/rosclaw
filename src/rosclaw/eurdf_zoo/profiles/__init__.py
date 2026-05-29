"""Robot profile modules for e-URDF-Zoo."""

from rosclaw.eurdf_zoo.profiles.franka_panda import FRANKA_PANDA_PROFILE
from rosclaw.eurdf_zoo.profiles.unitree_go2 import UNITREE_GO2_PROFILE
from rosclaw.eurdf_zoo.profiles.fetch_robot import FETCH_ROBOT_PROFILE

__all__ = [
    "FRANKA_PANDA_PROFILE",
    "UNITREE_GO2_PROFILE",
    "FETCH_ROBOT_PROFILE",
]
