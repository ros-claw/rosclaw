"""Explicit known SIM fixtures; dimensions do not authorize REAL execution."""

import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FixtureProfile:
    name: str
    vendor_model: str
    simulation_model: str
    body_id: str
    physical_radius_m: float
    cleaner_half_width_m: float
    recovery_radius_m: float
    coverage_width_m: float
    operation_width_m: float

    def to_dict(self):
        return {
            **asdict(self),
            "evidence_domain": "SIMULATION",
            "cleaner_kind": "simulated_attachment",
        }

    @property
    def cleaning_polygon(self):
        h = self.cleaner_half_width_m
        return [[-h, -h], [h, -h], [h, h], [-h, h]]


PROFILES = {
    "waffle": FixtureProfile(
        "waffle",
        "turtlebot3_waffle",
        "turtlebot3_waffle",
        "ros_expert_base",
        0.25,
        0.275,
        0.3,
        0.5,
        0.45,
    ),
    "burger": FixtureProfile(
        "burger",
        "turtlebot3_burger",
        "expert_robot",
        "ros_expert_burger",
        0.15,
        0.175,
        0.15,
        0.3,
        0.3,
    ),
}


def profile_for_urdf(path, requested=None):
    """Identify vendor identity, rejecting caller/model disagreement before dispatch."""
    robot = ET.fromstring(path.read_bytes())
    candidates = [p for p in PROFILES.values() if p.vendor_model == robot.get("name")]
    if robot.tag != "robot" or len(candidates) != 1:
        raise ValueError("an explicitly supported vendor simulation model is required")
    profile = candidates[0]
    if requested is not None and profile.name != requested:
        raise ValueError("requested fixture profile does not match vendor URDF")
    return profile
