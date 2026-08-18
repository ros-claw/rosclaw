"""EmbodimentCard — robot embodiment prior for Know module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmbodimentCard:
    """Embodiment prior: DOF, sensors, workspace, constraints."""

    robot_id: str = ""
    embodiment_type: str = ""  # manipulator | humanoid | quadruped | mobile
    dof: int = 0
    sensor_suite: list[str] = field(default_factory=list)
    workspace: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    dynamics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "embodiment_type": self.embodiment_type,
            "dof": self.dof,
            "sensor_suite": self.sensor_suite,
            "workspace": self.workspace,
            "constraints": self.constraints,
            "dynamics": self.dynamics,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EmbodimentCard:
        return cls(
            robot_id=d.get("robot_id", ""),
            embodiment_type=d.get("embodiment_type", ""),
            dof=d.get("dof", 0),
            sensor_suite=list(d.get("sensor_suite", [])),
            workspace=dict(d.get("workspace", {})),
            constraints=list(d.get("constraints", [])),
            dynamics=dict(d.get("dynamics", {})),
        )
