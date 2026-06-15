"""Built-in Critic provider with real evaluation logic.

Evaluates task results based on geometric thresholds, safety constraints,
and success criteria. Provides retry recommendations with parameter patches.
"""

from __future__ import annotations

from rosclaw.provider.core.errors import CapabilityNotSupportedError
from rosclaw.provider.core.provider import Provider
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse


class MockCriticProvider(Provider):
    """Critic provider for success detection and retry advice.

    Capabilities:
        - critic.success_detection: evaluate if task succeeded
        - critic.retry_advice: generate recovery recommendations

    Evaluation criteria:
        - Position error < threshold (default 3cm for reach, 5cm for nav)
        - Orientation error < threshold (default 5 deg)
        - No collision / workspace violation
        - Action norm within limits
        - Timeout not exceeded
    """

    _THRESHOLDS: dict[str, dict[str, float]] = {
        "reach": {"position_error_m": 0.03, "orientation_error_deg": 5.0, "timeout_s": 10.0},
        "grasp": {"position_error_m": 0.02, "orientation_error_deg": 10.0, "timeout_s": 15.0},
        "navigate": {"position_error_m": 0.05, "orientation_error_deg": 15.0, "timeout_s": 60.0},
        "pid_move": {"position_error_m": 0.05, "overshoot_m": 0.15, "timeout_s": 30.0},
        "walk": {"position_error_m": 0.10, "fall": 0.0, "timeout_s": 60.0},
        "default": {"position_error_m": 0.05, "orientation_error_deg": 10.0, "timeout_s": 30.0},
    }

    def __init__(self, manifest):
        super().__init__(manifest)

    async def infer(self, request: ProviderRequest) -> ProviderResponse:
        self._ensure_capability_supported(request.capability)

        if request.capability == "critic.success_detection":
            return await self._success_detection(request)
        if request.capability == "critic.retry_advice":
            return await self._retry_advice(request)

        raise CapabilityNotSupportedError(
            f"MockCriticProvider does not support '{request.capability}'",
            provider=self.name,
        )

    async def health(self):
        return {
            "ok": True,
            "provider": self.name,
            "capabilities": self.capabilities,
            "backend": "rule_based_evaluator",
        }

    async def _success_detection(self, request):
        inputs = request.inputs
        task_type = inputs.get("task_type", "default")
        thresholds = self._THRESHOLDS.get(task_type, self._THRESHOLDS["default"])

        target = inputs.get("target_pose")
        actual = inputs.get("actual_pose")
        collision = inputs.get("collision", False)
        workspace_violation = inputs.get("workspace_violation", False)
        elapsed = inputs.get("elapsed_time", 0.0)
        action_norm = inputs.get("action_norm", 0.0)
        max_action_norm = inputs.get("max_action_norm", 0.0)
        fall_detected = inputs.get("fall_detected", False)
        overshoot = inputs.get("overshoot", 0.0)

        checks = []
        all_passed = True

        if target is not None and actual is not None:
            pos_err = self._position_error(target, actual)
            pos_ok = pos_err <= thresholds["position_error_m"]
            checks.append({"check": "position_error", "value": round(pos_err, 4), "threshold": thresholds["position_error_m"], "passed": pos_ok})
            if not pos_ok:
                all_passed = False

        target_ori = inputs.get("target_orientation")
        actual_ori = inputs.get("actual_orientation")
        if target_ori is not None and actual_ori is not None and "orientation_error_deg" in thresholds:
            ori_err = self._orientation_error(target_ori, actual_ori)
            ori_ok = ori_err <= thresholds["orientation_error_deg"]
            checks.append({"check": "orientation_error", "value": round(ori_err, 2), "threshold": thresholds["orientation_error_deg"], "passed": ori_ok})
            if not ori_ok:
                all_passed = False

        if collision:
            checks.append({"check": "collision", "value": True, "passed": False})
            all_passed = False
        else:
            checks.append({"check": "collision", "value": False, "passed": True})

        if workspace_violation:
            checks.append({"check": "workspace_violation", "value": True, "passed": False})
            all_passed = False
        else:
            checks.append({"check": "workspace_violation", "value": False, "passed": True})

        if elapsed > thresholds.get("timeout_s", 30.0):
            checks.append({"check": "timeout", "value": elapsed, "threshold": thresholds["timeout_s"], "passed": False})
            all_passed = False
        else:
            checks.append({"check": "timeout", "value": elapsed, "threshold": thresholds["timeout_s"], "passed": True})

        if max_action_norm > 0 and action_norm > max_action_norm:
            checks.append({"check": "action_norm", "value": action_norm, "threshold": max_action_norm, "passed": False})
            all_passed = False
        elif max_action_norm > 0:
            checks.append({"check": "action_norm", "value": action_norm, "threshold": max_action_norm, "passed": True})

        if "fall" in thresholds and fall_detected:
            checks.append({"check": "fall_detected", "value": True, "passed": False})
            all_passed = False
        elif "fall" in thresholds:
            checks.append({"check": "fall_detected", "value": False, "passed": True})

        if "overshoot_m" in thresholds and overshoot > thresholds["overshoot_m"]:
            checks.append({"check": "overshoot", "value": overshoot, "threshold": thresholds["overshoot_m"], "passed": False})
            all_passed = False
        elif "overshoot_m" in thresholds:
            checks.append({"check": "overshoot", "value": overshoot, "threshold": thresholds["overshoot_m"], "passed": True})

        if checks:
            passed_ratio = sum(1 for c in checks if c["passed"]) / len(checks)
            confidence = 0.5 + passed_ratio * 0.5
        else:
            confidence = 0.5

        if all_passed:
            reason = f"All {len(checks)} checks passed for '{task_type}'"
        else:
            failed = [c["check"] for c in checks if not c["passed"]]
            reason = f"Failed checks: {', '.join(failed)}"

        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability="critic.success_detection",
            result={
                "success": all_passed,
                "task_type": task_type,
                "checks": checks,
                "confidence": round(confidence, 2),
                "reason": reason,
            },
            confidence=round(confidence, 2),
        )

    async def _retry_advice(self, request):
        inputs = request.inputs
        task_type = inputs.get("task_type", "default")
        failed_checks = inputs.get("failed_checks", [])

        patches = []
        recommendations = []

        for check in failed_checks:
            check_name = check if isinstance(check, str) else check.get("check", "")

            if check_name == "position_error":
                patches.append({"parameter": "approach_z", "delta": -0.02, "unit": "m"})
                patches.append({"parameter": "final_speed", "scale": 0.7})
                recommendations.append("Reduce approach speed and lower z-offset by 2cm")

            elif check_name == "orientation_error":
                patches.append({"parameter": "orientation_tolerance", "delta": 2.0, "unit": "deg"})
                recommendations.append("Increase orientation tolerance or add intermediate waypoint")

            elif check_name == "collision":
                patches.append({"parameter": "safety_margin", "delta": 0.05, "unit": "m"})
                recommendations.append("Increase safety margin by 5cm, use longer approach path")

            elif check_name == "workspace_violation":
                patches.append({"parameter": "target_x", "delta": 0.0, "clamp_to_workspace": True})
                recommendations.append("Clamp target to workspace boundary, use intermediate pose")

            elif check_name == "timeout":
                patches.append({"parameter": "max_velocity", "scale": 1.2})
                recommendations.append("Increase max velocity or simplify trajectory")

            elif check_name == "action_norm":
                patches.append({"parameter": "action_gain", "scale": 0.8})
                recommendations.append("Reduce action gain by 20% to avoid saturation")

            elif check_name == "fall_detected":
                patches.append({"parameter": "walking_speed", "scale": 0.7})
                patches.append({"parameter": "step_length", "scale": 0.8})
                recommendations.append("Reduce walking speed and step length, lower CoM")

            elif check_name == "overshoot":
                patches.append({"parameter": "Kp", "scale": 0.85})
                patches.append({"parameter": "Kd", "delta": 0.01, "unit": "gain"})
                recommendations.append("Reduce Kp by 15%, increase Kd for damping")

        if not recommendations:
            recommendations.append("No specific recovery advice — consider replanning from scratch")

        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability="critic.retry_advice",
            result={
                "task_type": task_type,
                "recommended": len(patches) > 0,
                "patches": patches,
                "recommendations": recommendations,
                "estimated_success_improvement": min(0.3, len(patches) * 0.05),
            },
            confidence=0.75 if patches else 0.5,
        )

    @staticmethod
    def _position_error(target, actual):
        return sum((t - a) ** 2 for t, a in zip(target, actual, strict=False)) ** 0.5

    @staticmethod
    def _orientation_error(target, actual):
        if len(target) == len(actual) == 4:
            dot = abs(sum(t * a for t, a in zip(target, actual, strict=False)))
            dot = min(1.0, dot)
            return 2.0 * (dot ** 0.5) * 57.2958
        return sum(abs(t - a) for t, a in zip(target, actual, strict=False))
