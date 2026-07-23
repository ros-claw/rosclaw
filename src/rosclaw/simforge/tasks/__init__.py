"""Built-in CoreSimBench tasks."""

from rosclaw.simforge.tasks.shield_reach import (
    ShieldReachCase,
    ShieldReachPlan,
    compile_automatic_candidate,
    generate_shield_reach_1k,
    generate_shield_reach_cases,
    run_shield_reach_evaluation,
)

__all__ = [
    "ShieldReachCase",
    "ShieldReachPlan",
    "compile_automatic_candidate",
    "generate_shield_reach_1k",
    "generate_shield_reach_cases",
    "run_shield_reach_evaluation",
]
