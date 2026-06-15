"""Darwin stress scenarios."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StressScenario:
    """A stress scenario definition."""

    scenario_id: str = ""
    name: str = ""
    task_family: str = ""
    description: str = ""
    perturbations: dict[str, Any] = field(default_factory=dict)
    difficulty: str = "normal"  # easy | normal | hard | extreme

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "task_family": self.task_family,
            "description": self.description,
            "perturbations": self.perturbations,
            "difficulty": self.difficulty,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StressScenario:
        return cls(
            scenario_id=d.get("scenario_id", ""),
            name=d.get("name", ""),
            task_family=d.get("task_family", ""),
            description=d.get("description", ""),
            perturbations=dict(d.get("perturbations", {})),
            difficulty=d.get("difficulty", "normal"),
        )


DEFAULT_SCENARIOS: list[StressScenario] = [
    StressScenario(
        scenario_id="pick_cube_heavy",
        name="PickCube Heavy Object",
        task_family="manipulation",
        description="Object mass increased by 3x, friction reduced.",
        perturbations={"mass_multiplier": 3.0, "friction_reduction": 0.5},
        difficulty="hard",
    ),
    StressScenario(
        scenario_id="pick_cube_occluded",
        name="PickCube Occluded",
        task_family="manipulation",
        description="Object partially occluded by obstacle.",
        perturbations={"occlusion_ratio": 0.4, "lighting_variance": 0.3},
        difficulty="hard",
    ),
    StressScenario(
        scenario_id="valve_tight",
        name="Valve Tight Handle",
        task_family="manipulation",
        description="Valve requires 2x torque to operate.",
        perturbations={"torque_multiplier": 2.0, " backlash_increase": 0.1},
        difficulty="hard",
    ),
    StressScenario(
        scenario_id="button_small",
        name="Press Button Small Target",
        task_family="manipulation",
        description="Button diameter reduced by 50%.",
        perturbations={"button_scale": 0.5, "approach_noise": 0.02},
        difficulty="normal",
    ),
]


def get_scenario(scenario_id: str) -> StressScenario | None:
    for s in DEFAULT_SCENARIOS:
        if s.scenario_id == scenario_id:
            return s
    return None


def list_scenarios(task_family: str | None = None) -> list[StressScenario]:
    if task_family is None:
        return list(DEFAULT_SCENARIOS)
    return [s for s in DEFAULT_SCENARIOS if s.task_family == task_family]
