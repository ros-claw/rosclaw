"""BodySense summarizer and block explainer."""

from __future__ import annotations

from typing import Any

from rosclaw.sense.schemas import (
    BodyReadiness,
    BodyRiskSummary,
    BodySense,
    BodyState,
)


class SenseExplainer:
    """Convert BodyState + risk + readiness into human/Agent-readable summaries."""

    def summarize(
        self,
        state: BodyState,
        risk_summary: BodyRiskSummary,
        readiness: BodyReadiness,
    ) -> BodySense:
        """Produce a BodySense summary from state, risk, and readiness."""
        blocked: list[str] = []
        degraded: list[str] = []
        main_reasons: list[str] = []
        recommended: list[str] = []

        for name, item in readiness.capabilities.items():
            if item.status == "not_ready":
                blocked.append(name)
            elif item.status == "degraded":
                degraded.append(name)
            for reason in item.reasons:
                if reason not in main_reasons:
                    main_reasons.append(reason)
            for action in item.allowed_alternatives:
                if action not in recommended:
                    recommended.append(action)

        # Add risk-derived recommendations.
        if risk_summary.thermal_risk in ("high", "critical"):
            recommended.append("cooldown")
        if risk_summary.perception_risk in ("medium", "high"):
            recommended.append("re_detect_target")
        if risk_summary.balance_risk in ("high", "critical"):
            recommended.append("stabilize")
        if risk_summary.power_risk in ("medium", "high", "critical"):
            recommended.append("recharge")
        if risk_summary.communication_risk in ("medium", "high"):
            recommended.append("reduce_loop_rate")

        # De-duplicate while preserving order.
        recommended = list(dict.fromkeys(recommended))
        main_reasons = list(dict.fromkeys(main_reasons))

        overall = readiness.overall_status
        # A low estimated risk is not proof that capability requirements are
        # known. Preserve UNKNOWN readiness; observed hazards may still warn.
        if overall == "unknown" and risk_summary.overall_risk in {"medium", "high", "critical"}:
            overall = self._risk_to_status(risk_summary.overall_risk)

        summary_text = self._build_summary_text(
            state, blocked, degraded, main_reasons, recommended, readiness.overall_status
        )

        evidence: dict[str, Any] = {
            "collector_source": state.source,
            "observation_kind": _observation_kind(state.source),
            "battery_percent": state.energy.battery_percent,
            "max_joint_temperature_c": self._max_joint_temp(state),
            "dds_latency_ms": state.communication.dds_latency_ms,
            "support_margin": state.balance.support_margin,
            "target_detector_confidence": state.perception.target_detector_confidence,
        }

        return BodySense(
            robot_id=state.robot_id,
            timestamp=state.timestamp,
            overall_status=overall,
            risk_summary=risk_summary,
            readiness=readiness,
            main_reasons=main_reasons,
            blocked_capabilities=blocked,
            degraded_capabilities=degraded,
            recommended_actions=recommended,
            natural_language_summary=summary_text,
            evidence=evidence,
            source_state_id=f"state_{state.timestamp}",
        )

    def explain_block(self, task: str, readiness: BodyReadiness) -> str:
        """Explain why ``task`` is blocked or degraded."""
        item = readiness.capabilities.get(task)
        if item is None:
            return f"No readiness information for task '{task}'."

        lines: list[str] = []
        if item.status == "ready":
            lines.append(f"{task} is ready.")
            return "\n".join(lines)

        lines.append(f"{task} is {item.status} because:")
        for req in item.failed_requirements:
            lines.append(f"  - {req.name}: current={req.current}, required={req.required}")
        if item.allowed_alternatives:
            lines.append("Allowed alternatives:")
            for alt in item.allowed_alternatives:
                lines.append(f"  - {alt}")
        return "\n".join(lines)

    def _build_summary_text(
        self,
        state: BodyState,
        blocked: list[str],
        degraded: list[str],
        reasons: list[str],
        recommended: list[str],
        readiness_status: str,
    ) -> str:
        robot = state.robot_id
        if not blocked and not degraded:
            if readiness_status != "ready":
                return f"{robot} readiness is {readiness_status}; readiness is not established."
            return f"{robot} is ready for all evaluated capabilities."

        parts: list[str] = [f"{robot} status: {self._describe_status(blocked, degraded)}."]
        if reasons:
            parts.append("Main reasons: " + ", ".join(reasons) + ".")
        if blocked:
            parts.append("Blocked: " + ", ".join(blocked) + ".")
        if degraded:
            parts.append("Degraded: " + ", ".join(degraded) + ".")
        if recommended:
            parts.append("Recommended actions: " + ", ".join(recommended) + ".")
        return " ".join(parts)

    def _describe_status(self, blocked: list[str], degraded: list[str]) -> str:
        if blocked and degraded:
            return "not_ready (some capabilities blocked, others degraded)"
        if blocked:
            return "not_ready (some capabilities blocked)"
        if degraded:
            return "caution (some capabilities degraded)"
        return "ready"

    def _risk_to_status(self, risk: str) -> str:
        return {
            "low": "ready",
            "medium": "caution",
            "high": "not_ready",
            "critical": "emergency",
        }.get(risk, "unknown")

    def _max_joint_temp(self, state: BodyState) -> float | None:
        temps = [j.temperature_c for j in state.joints.values() if j.temperature_c is not None]
        return max(temps) if temps else None


def _observation_kind(source: str) -> str:
    if source in {
        "unknown",
        "unconfigured",
        "file_replay:empty",
        "sense:degraded",
        "ros2:unconfigured",
        "ros2:unavailable",
    } or source.endswith(":stub"):
        return "unavailable"
    if source.startswith("mock:"):
        return "synthetic"
    if source.startswith("file_replay:"):
        return "historical_replay"
    # A source label alone cannot establish current, verified physical input.
    return "unverified"
