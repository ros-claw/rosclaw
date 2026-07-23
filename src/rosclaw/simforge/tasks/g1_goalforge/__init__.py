"""G1 GoalForge task contracts."""

from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_DDS_JOINT_NAMES,
    G1_HARD_TORQUE_LIMITS,
    GOALFORGE_TASK_ID,
    GoalForgeResult,
    GoalForgeStatus,
    ShotParameters,
    SimulationReceiptV4,
)
from rosclaw.simforge.tasks.g1_goalforge.scenario import (
    GoalForgeScenario,
    generate_goalforge_scenarios,
)

__all__ = [
    "G1_DDS_JOINT_NAMES",
    "G1_HARD_TORQUE_LIMITS",
    "GOALFORGE_TASK_ID",
    "GoalForgeResult",
    "GoalForgeScenario",
    "GoalForgeStatus",
    "ShotParameters",
    "SimulationReceiptV4",
    "generate_goalforge_scenarios",
]
