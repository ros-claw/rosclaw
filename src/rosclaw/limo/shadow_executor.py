"""LIMO SHADOW executor（审计 FTC-100）：验证与规划，绝不驱动硬件。

SHADOW 模式下 Executor 返回 SHADOW receipt：
- evidence_domain=shadow（与 SIMULATED/REAL 严格分域）；
- actuated=false（硬阻断：不打开串口/不发布 ROS 命令）；
- 携带拟执行的 ROS 命令描述（可审计）；
- usable_for_real_execution=false。
"""

from __future__ import annotations

from typing import Any

from rosclaw.kernel.contracts import (
    ActionEnvelope,
    ActionExecutionResult,
    ActionState,
    EvidenceDomain,
    EvidenceLevel,
)


#: SHADOW 支持的能力与其参数校验器。
def _validate_tone(arguments: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    freq = arguments.get("frequency_hz")
    dur = arguments.get("duration_sec")
    vol = arguments.get("volume_percent")
    if not isinstance(freq, (int, float)) or not (20 <= freq <= 20_000):
        errors.append(f"frequency_hz {freq!r} out of [20, 20000]")
    if not isinstance(dur, (int, float)) or not (0.01 <= dur <= 5.0):
        errors.append(f"duration_sec {dur!r} out of [0.01, 5.0]")
    if not isinstance(vol, (int, float)) or not (0 <= vol <= 100):
        errors.append(f"volume_percent {vol!r} out of [0, 100]")
    return errors


def _validate_pose(arguments: dict[str, Any]) -> list[str]:
    from math import isfinite

    errors: list[str] = []
    for key in ("x", "y", "yaw"):
        value = arguments.get(key)
        if not isinstance(value, (int, float)) or not isfinite(value):
            errors.append(f"{key} {value!r} must be finite")
    return errors


_VALIDATORS = {
    "limo.speaker.play_tone": _validate_tone,
    "limo.localization.set_initial_pose": _validate_pose,
}


def limo_shadow_executor(action: ActionEnvelope) -> ActionExecutionResult:
    """SHADOW 执行：验证 + 规划 ROS 命令 + 硬阻断回报。"""
    validator = _VALIDATORS.get(action.capability_id)
    if validator is None:
        return ActionExecutionResult(
            final_state=ActionState.BLOCKED,
            evidence_level=EvidenceLevel.UNSUPPORTED,
            evidence_domain=EvidenceDomain.SHADOW,
            simulation_result={"actuated": False},
            verification_result={"verified": False, "reason": "unsupported capability"},
            errors=[{"code": "SHADOW_UNSUPPORTED", "message": f"no SHADOW validator for {action.capability_id!r}"}],
        )
    errors = validator(dict(action.arguments))
    if errors:
        return ActionExecutionResult(
            final_state=ActionState.FAILED,
            evidence_level=EvidenceLevel.UNSUPPORTED,
            evidence_domain=EvidenceDomain.SHADOW,
            simulation_result={"actuated": False},
            verification_result={"verified": False, "reason": "argument validation failed"},
            errors=[{"code": "SHADOW_INVALID_ARGUMENTS", "message": e} for e in errors],
        )
    planned = _planned_commands(action)
    return ActionExecutionResult(
        final_state=ActionState.COMPLETED,
        evidence_level=EvidenceLevel.TASK_VERIFIED,
        evidence_domain=EvidenceDomain.SHADOW,
        simulation_result={
            "actuated": False,
            "usable_for_real_execution": False,
            "planned_ros_commands": planned,
        },
        verification_result={
            "verified": True,
            "method": "shadow_plan_validation",
            "actuation_gate": "hard_blocked",
        },
    )


def _planned_commands(action: ActionEnvelope) -> list[str]:
    if action.capability_id == "limo.speaker.play_tone":
        a = action.arguments
        return [
            f"ros2 topic pub --once /buzzer_tone std_msgs/msg/Int32 "
            f"{{data: {int(a.get('frequency_hz', 0))}}}",
            f"sleep {float(a.get('duration_sec', 0)):.2f}",
            "ros2 topic pub --once /buzzer_stop std_msgs/msg/Empty {}",
        ]
    if action.capability_id == "limo.localization.set_initial_pose":
        a = action.arguments
        return [
            "ros2 service call /relocalization std_srvs/srv/Trigger {}",
            f"ros2 param set /amcl initial_pose "
            f"[{a.get('x', 0.0)}, {a.get('y', 0.0)}, {a.get('yaw', 0.0)}]",
        ]
    return [f"# unplanned capability {action.capability_id}"]
