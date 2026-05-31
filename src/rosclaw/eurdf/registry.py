"""e-URDF Registry — In-memory robot profile registry.

Re-exports RobotRegistry from runtime.eurdf_loader
so consumers can import from the unified `rosclaw.eurdf` namespace.
"""

from rosclaw.runtime.eurdf_loader import RobotRegistry

__all__ = ["RobotRegistry"]
