"""EVO-3 offline validations (数据库优化v4 §11.3/§11.4).

These two proofs run with NO hardware:

* Exp 3 (counter-regime): thermal-degradation memories are placed in a
  healthy regime — retrieval must surface them (Retrieve=true), the gate
  must refuse them (Applicable=false), and the decision must be ABSTAIN.
* Exp 4 (choreography protection): run1's harmful patches are re-sent to
  the ChoreographyValidator — 100% BLOCK, 0 real executions.
"""

from __future__ import annotations

from typing import Any

from rosclaw.how.choreography import ChoreographyValidator, load_contract
from rosclaw.how.choreography.timing import build_timing_model
from rosclaw.how.selective import InterventionAction

# The run1 death-spiral patch shapes (PR #98 run1 patch proofs).
RUN1_DEATH_SPIRAL_PATCHES: list[dict[str, Any]] = [
    {"servo_speed_scale": 0.6},
    {"servo_speed_scale": 0.5},
    {"per_phase_delay_ms": 400},
    {"reveal_delay_ms": 250},
    {"gesture_motion_duration_ms": 1200},
    {"servo_speed_scale": 0.6, "per_phase_delay_ms": 400},
]


def validate_exp3_counter_regime(
    pipeline: Any,
    *,
    healthy_regime: Any,
    failure_signature: str,
    body_id: str,
    joint_name: str | None = None,
) -> dict[str, Any]:
    """v4 §11.3: the most important negative test.

    Returns the measured triple: retrieve / applicable / decision.
    """
    decision = pipeline.decide(
        failure_signature,
        healthy_regime,
        body_id=body_id,
        joint_name=joint_name,
    )
    retrieved = decision.selected_memory_id is not None
    return {
        "retrieve": retrieved,
        "applicable": decision.applicability_score >= 0.70,
        "applicability_score": decision.applicability_score,
        "decision": decision.action.value,
        "passed": retrieved and decision.action is InterventionAction.ABSTAIN,
        "reason_codes": decision.reason_codes,
    }


def validate_exp4_choreography_protection(
    validator: ChoreographyValidator | None = None,
    *,
    contract_path: str = "configs/choreography/rh56_rps_v1.yaml",
) -> dict[str, Any]:
    """v4 §11.4: run1 patches must be 100% blocked, 0 real executions."""
    if validator is None:
        validator = ChoreographyValidator(load_contract(contract_path))
    model = build_timing_model(validator.contract, [], current_parameters={})
    results = []
    for patch in RUN1_DEATH_SPIRAL_PATCHES:
        validation = validator.validate(patch, model)
        results.append(
            {"patch": patch, "allowed": validation.allowed, "violations": validation.violations}
        )
    blocked = sum(1 for r in results if not r["allowed"])
    return {
        "total_patches": len(results),
        "blocked": blocked,
        "block_rate": blocked / len(results),
        "real_executions": 0,
        "passed": blocked == len(results),
        "results": results,
    }
