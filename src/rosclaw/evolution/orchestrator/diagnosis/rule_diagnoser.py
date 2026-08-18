"""RuleDiagnoser — rule-based failure diagnosis."""

import logging
from typing import Any

from ..core.diagnosis import Diagnosis

logger = logging.getLogger("rosclaw.auto.diagnosis.rule")


class RuleDiagnoser:
    """Diagnose failures using structured rules and metrics.

    Failure taxonomy:
    - missed_grasp
    - object_slip
    - collision
    - timeout
    - perception_lost
    - force_exceeded
    - navigation_stuck
    - button_press_missed
    - valve_rotation_failed
    """

    FAILURE_TAXONOMY = {
        "missed_grasp": {
            "root_causes": ["low_pregrasp_height", "approach_speed_too_fast", "gripper_timing_off"],
            "search_space": {"pregrasp_height": [0.02, 0.08], "approach_speed": [0.05, 0.2]},
            "risk_level": "low",
            "auto_repairable": True,
        },
        "object_slip": {
            "root_causes": ["gripper_force_too_low", "grasp_pose_unstable"],
            "search_space": {"gripper_force": [5, 20], "grasp_offset": [-0.02, 0.02]},
            "risk_level": "low",
            "auto_repairable": True,
        },
        "collision": {
            "root_causes": [
                "approach_speed_too_fast",
                "path_planning_failure",
                "obstacle_not_detected",
            ],
            "search_space": {"approach_speed": [0.02, 0.1], "collision_buffer": [0.01, 0.05]},
            "risk_level": "medium",
            "auto_repairable": True,
        },
        "timeout": {
            "root_causes": ["motion_too_slow", "planning_retry_exceeded", "stuck_in_local_minima"],
            "search_space": {"max_planning_time": [5, 30], "motion_speed": [0.05, 0.3]},
            "risk_level": "medium",
            "auto_repairable": True,
        },
        "perception_lost": {
            "root_causes": ["camera_occlusion", "lighting_change", "object_out_of_fov"],
            "search_space": {"camera_angle": [-30, 30], "detection_confidence": [0.5, 0.9]},
            "risk_level": "high",
            "auto_repairable": False,
        },
        "force_exceeded": {
            "root_causes": ["contact_force_too_high", "impulse_on_collision", "stiff_control_gain"],
            "search_space": {"force_limit": [10, 50], "compliance_gain": [0.1, 1.0]},
            "risk_level": "high",
            "auto_repairable": True,
        },
        "navigation_stuck": {
            "root_causes": ["obstacle_blocking", "planner_failure", "localization_drift"],
            "search_space": {"recovery_radius": [0.5, 2.0], "planner_timeout": [10, 60]},
            "risk_level": "medium",
            "auto_repairable": True,
        },
        "button_press_missed": {
            "root_causes": [
                "approach_angle_wrong",
                "force_threshold_too_high",
                "pose_estimation_error",
            ],
            "search_space": {"approach_angle": [0, 15], "press_force": [1, 10]},
            "risk_level": "low",
            "auto_repairable": True,
        },
        "valve_rotation_failed": {
            "root_causes": ["torque_insufficient", "handle_slip", "alignment_error"],
            "search_space": {"max_torque": [2, 10], "alignment_tolerance": [0.02, 0.1]},
            "risk_level": "medium",
            "auto_repairable": True,
        },
    }

    def diagnose(self, failure_case: Any) -> Diagnosis:
        """Generate a rule-based diagnosis from a failure case."""
        failure_mode = getattr(failure_case, "failure_mode", "unknown")
        rule = self.FAILURE_TAXONOMY.get(
            failure_mode,
            {
                "root_causes": ["unknown"],
                "search_space": {},
                "risk_level": "medium",
                "auto_repairable": False,
            },
        )

        diag = Diagnosis(
            id=f"diag_{failure_mode}_{getattr(failure_case, 'id', 'unknown')[:8]}",
            failure_id=getattr(failure_case, "id", ""),
            task=getattr(failure_case, "task_id", ""),
            skill=getattr(failure_case, "skill_id", ""),
            root_cause_candidates=rule["root_causes"],
            recommended_search_space=rule["search_space"],
            confidence=0.75 if rule["auto_repairable"] else 0.4,
            auto_repairable=rule["auto_repairable"],
            risk_level=rule["risk_level"],
        )
        logger.info(
            "RuleDiagnoser: %s -> %s (repairable=%s)",
            failure_mode,
            diag.root_cause_candidates,
            diag.auto_repairable,
        )
        return diag

    def can_diagnose(self, failure_mode: str) -> bool:
        """Check if a failure mode is in the known taxonomy."""
        return failure_mode in self.FAILURE_TAXONOMY
