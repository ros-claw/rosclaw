"""Readiness evaluator: decide if a task/capability can run given body state."""

from __future__ import annotations

import time
from typing import Any

from rosclaw.sense.schemas import (
    BodyReadiness,
    BodyRiskSummary,
    BodyState,
    FailedRequirement,
    ReadinessItem,
)

DEFAULT_CAPABILITY_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "observe_scene": {
        "camera_fps_min": 10.0,
        "target_detector_confidence_min": 0.30,
        "battery_percent_min": 5.0,
        "dds_latency_ms_max": 200.0,
        "fall_risk_max": "high",
    },
    "walk_slow": {
        "battery_percent_min": 20.0,
        "max_leg_joint_temp_c": 80.0,
        "dds_latency_ms_max": 50.0,
        "fall_risk_max": "medium",
    },
    "kick_ball": {
        "battery_percent_min": 40.0,
        "max_leg_joint_temp_c": 72.0,
        "fall_risk_max": "low",
        "support_margin_min": 0.15,
        "support_stability_sec_min": 0.8,
        "target_detector_confidence_min": 0.80,
        "dds_latency_ms_max": 25.0,
        "sandbox_required": True,
        "human_supervision_required": True,
    },
    "high_power_motion": {
        "battery_percent_min": 50.0,
        "max_leg_joint_temp_c": 70.0,
        "fall_risk_max": "low",
        "dds_latency_ms_max": 20.0,
    },
}

_RISK_ORDER = {"unknown": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


class ReadinessEvaluator:
    """Evaluate whether a robot is ready to perform a capability or task."""

    def __init__(
        self,
        thresholds: dict[str, Any] | None = None,
        capability_requirements: dict[str, dict[str, Any]] | None = None,
    ):
        self.thresholds = thresholds or {}
        reqs = capability_requirements if capability_requirements is not None else DEFAULT_CAPABILITY_REQUIREMENTS
        self.capability_requirements = reqs if reqs else DEFAULT_CAPABILITY_REQUIREMENTS

    def evaluate(
        self,
        state: BodyState,
        risk_summary: BodyRiskSummary,
        task: str | None = None,
        requirements: dict[str, Any] | None = None,
    ) -> BodyReadiness:
        """Return BodyReadiness for ``task`` using explicit or default requirements."""
        if task and requirements is None:
            requirements = self.capability_requirements.get(task, {})
        requirements = requirements or {}

        item = self._evaluate_one(state, risk_summary, task or "task", requirements)
        overall = self._derive_overall([item])
        return BodyReadiness(
            robot_id=state.robot_id,
            timestamp=time.time(),
            task=task,
            overall_status=overall,
            capabilities={item.capability: item},
        )

    def evaluate_all(
        self,
        state: BodyState,
        risk_summary: BodyRiskSummary,
        capabilities: dict[str, dict[str, Any]] | None = None,
    ) -> BodyReadiness:
        """Evaluate readiness for all known capabilities."""
        capabilities = capabilities or self.capability_requirements
        items: dict[str, ReadinessItem] = {}
        for name, reqs in capabilities.items():
            items[name] = self._evaluate_one(state, risk_summary, name, reqs)
        return BodyReadiness(
            robot_id=state.robot_id,
            timestamp=time.time(),
            overall_status=self._derive_overall(list(items.values())),
            capabilities=items,
        )

    def _evaluate_one(
        self,
        state: BodyState,
        risk_summary: BodyRiskSummary,
        capability: str,
        requirements: dict[str, Any],
    ) -> ReadinessItem:
        failed: list[FailedRequirement] = []
        reasons: list[str] = []

        self._check_battery(state, requirements, failed, reasons)
        self._check_leg_temperature(state, requirements, failed, reasons)
        self._check_fall_risk(risk_summary, requirements, failed, reasons)
        self._check_support_margin(state, requirements, failed, reasons)
        self._check_support_stability(state, requirements, failed, reasons)
        self._check_target_confidence(state, requirements, failed, reasons)
        self._check_camera_fps(state, requirements, failed, reasons)
        self._check_dds_latency(state, requirements, failed, reasons)
        self._check_compute(risk_summary, requirements, failed, reasons)
        self._check_overall_risk(risk_summary, requirements, failed, reasons)

        if failed:
            status = "not_ready"
            alternatives = self._alternatives(capability, requirements)
        elif reasons:
            status = "degraded"
            alternatives = self._alternatives(capability, requirements)
        else:
            status = "ready"
            alternatives = []

        return ReadinessItem(
            capability=capability,
            status=status,
            reasons=reasons,
            failed_requirements=failed,
            allowed_alternatives=alternatives,
        )

    def _derive_overall(self, items: list[ReadinessItem]) -> str:
        if not items:
            return "unknown"
        if any(i.status == "not_ready" for i in items):
            return "not_ready"
        if any(i.status == "degraded" for i in items):
            return "caution"
        if all(i.status == "ready" for i in items):
            return "ready"
        return "unknown"

    def _alternatives(self, capability: str, requirements: dict[str, Any]) -> list[str]:
        base = ["observe_scene", "cooldown", "re_detect_target", "sandbox_only"]
        # Filter out the blocked capability itself and duplicates.
        return [a for a in base if a != capability]

    def _fail(
        self,
        name: str,
        current: Any,
        required: Any,
        severity: str,
        reason: str,
        failed: list[FailedRequirement],
        reasons: list[str],
    ) -> None:
        failed.append(
            FailedRequirement(
                name=name,
                current=current,
                required=required,
                severity=severity,
                reason=reason,
            )
        )
        reasons.append(reason)

    def _check_battery(
        self,
        state: BodyState,
        requirements: dict[str, Any],
        failed: list[FailedRequirement],
        reasons: list[str],
    ) -> None:
        min_battery = requirements.get("battery_percent_min")
        if min_battery is None:
            return
        current = state.energy.battery_percent
        if current is None:
            self._fail(
                "battery_percent",
                "unknown",
                f">={min_battery}",
                "medium",
                "battery_percent unknown",
                failed,
                reasons,
            )
            return
        if current < min_battery:
            self._fail(
                "battery_percent",
                current,
                f">={min_battery}",
                "high" if current < min_battery * 0.5 else "medium",
                f"battery_percent {current}% below required {min_battery}%",
                failed,
                reasons,
            )

    def _check_leg_temperature(
        self,
        state: BodyState,
        requirements: dict[str, Any],
        failed: list[FailedRequirement],
        reasons: list[str],
    ) -> None:
        max_temp = requirements.get("max_leg_joint_temp_c")
        if max_temp is None:
            return
        leg_joints = {
            name: j for name, j in state.joints.items()
            if "hip" in name or "knee" in name or "ankle" in name
        }
        if not leg_joints:
            return
        for name, joint in leg_joints.items():
            temp = joint.temperature_c
            if temp is None:
                continue
            if temp > max_temp:
                self._fail(
                    f"{name}_temperature",
                    temp,
                    f"<={max_temp}",
                    "high" if temp > max_temp + 10 else "medium",
                    f"{name} temperature {temp}C exceeds limit {max_temp}C",
                    failed,
                    reasons,
                )

    def _check_fall_risk(
        self,
        risk_summary: BodyRiskSummary,
        requirements: dict[str, Any],
        failed: list[FailedRequirement],
        reasons: list[str],
    ) -> None:
        max_fall = requirements.get("fall_risk_max")
        if max_fall is None:
            return
        current = risk_summary.balance_risk
        if current == "unknown":
            self._fail(
                "fall_risk",
                "unknown",
                f"<={max_fall}",
                "medium",
                "fall_risk unknown",
                failed,
                reasons,
            )
            return
        if _RISK_ORDER.get(current, 0) > _RISK_ORDER.get(max_fall, 0):
            self._fail(
                "fall_risk",
                current,
                f"<={max_fall}",
                "high" if current in ("high", "critical") else "medium",
                f"fall_risk {current} exceeds allowed {max_fall}",
                failed,
                reasons,
            )

    def _check_support_margin(
        self,
        state: BodyState,
        requirements: dict[str, Any],
        failed: list[FailedRequirement],
        reasons: list[str],
    ) -> None:
        min_margin = requirements.get("support_margin_min")
        if min_margin is None:
            return
        current = state.balance.support_margin
        if current is None:
            self._fail(
                "support_margin",
                "unknown",
                f">={min_margin}",
                "medium",
                "support_margin unknown",
                failed,
                reasons,
            )
            return
        if current < min_margin:
            self._fail(
                "support_margin",
                current,
                f">={min_margin}",
                "high",
                f"support_margin {current} below required {min_margin}",
                failed,
                reasons,
            )

    def _check_support_stability(
        self,
        state: BodyState,
        requirements: dict[str, Any],
        failed: list[FailedRequirement],
        reasons: list[str],
    ) -> None:
        min_stable = requirements.get("support_stability_sec_min")
        if min_stable is None:
            return
        current = state.balance.stable_for_sec
        if current is None:
            self._fail(
                "support_stability_sec",
                "unknown",
                f">={min_stable}",
                "medium",
                "support stability unknown",
                failed,
                reasons,
            )
            return
        if current < min_stable:
            self._fail(
                "support_stability_sec",
                current,
                f">={min_stable}",
                "medium",
                f"support stable for {current}s, required {min_stable}s",
                failed,
                reasons,
            )

    def _check_target_confidence(
        self,
        state: BodyState,
        requirements: dict[str, Any],
        failed: list[FailedRequirement],
        reasons: list[str],
    ) -> None:
        min_conf = requirements.get("target_detector_confidence_min")
        if min_conf is None:
            return
        current = state.perception.target_detector_confidence
        if current is None:
            self._fail(
                "target_detector_confidence",
                "unknown",
                f">={min_conf}",
                "medium",
                "target_detector_confidence unknown",
                failed,
                reasons,
            )
            return
        if current < min_conf:
            self._fail(
                "target_detector_confidence",
                current,
                f">={min_conf}",
                "high" if current < min_conf * 0.8 else "medium",
                f"target_detector_confidence {current} below required {min_conf}",
                failed,
                reasons,
            )

    def _check_camera_fps(
        self,
        state: BodyState,
        requirements: dict[str, Any],
        failed: list[FailedRequirement],
        reasons: list[str],
    ) -> None:
        min_fps = requirements.get("camera_fps_min")
        if min_fps is None:
            return
        current = state.perception.front_camera_fps
        if current is None:
            self._fail(
                "front_camera_fps",
                "unknown",
                f">={min_fps}",
                "low",
                "front_camera_fps unknown",
                failed,
                reasons,
            )
            return
        if current < min_fps:
            self._fail(
                "front_camera_fps",
                current,
                f">={min_fps}",
                "medium",
                f"front_camera_fps {current} below required {min_fps}",
                failed,
                reasons,
            )

    def _check_dds_latency(
        self,
        state: BodyState,
        requirements: dict[str, Any],
        failed: list[FailedRequirement],
        reasons: list[str],
    ) -> None:
        max_latency = requirements.get("dds_latency_ms_max")
        if max_latency is None:
            return
        current = state.communication.dds_latency_ms
        if current is None:
            self._fail(
                "dds_latency_ms",
                "unknown",
                f"<={max_latency}",
                "medium",
                "dds_latency_ms unknown",
                failed,
                reasons,
            )
            return
        if current > max_latency:
            self._fail(
                "dds_latency_ms",
                current,
                f"<={max_latency}",
                "high" if current > max_latency * 2 else "medium",
                f"dds_latency_ms {current}ms exceeds allowed {max_latency}ms",
                failed,
                reasons,
            )

    def _check_compute(
        self,
        risk_summary: BodyRiskSummary,
        requirements: dict[str, Any],
        failed: list[FailedRequirement],
        reasons: list[str],
    ) -> None:
        if requirements.get("compute_ok", False) is False:
            return
        if risk_summary.compute_risk in ("high", "critical"):
            self._fail(
                "compute_risk",
                risk_summary.compute_risk,
                "<=medium",
                "high",
                f"compute_risk {risk_summary.compute_risk} too high",
                failed,
                reasons,
            )

    def _check_overall_risk(
        self,
        risk_summary: BodyRiskSummary,
        requirements: dict[str, Any],
        failed: list[FailedRequirement],
        reasons: list[str],
    ) -> None:
        max_overall = requirements.get("overall_risk_max")
        if max_overall is None:
            return
        current = risk_summary.overall_risk
        if current == "unknown":
            self._fail(
                "overall_risk",
                "unknown",
                f"<={max_overall}",
                "medium",
                "overall_risk unknown",
                failed,
                reasons,
            )
            return
        if _RISK_ORDER.get(current, 0) > _RISK_ORDER.get(max_overall, 0):
            self._fail(
                "overall_risk",
                current,
                f"<={max_overall}",
                "high",
                f"overall_risk {current} exceeds allowed {max_overall}",
                failed,
                reasons,
            )
