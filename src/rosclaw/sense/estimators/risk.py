"""Risk estimator: convert BodyState into BodyRiskSummary and BodyEvents."""

from __future__ import annotations

import time
import uuid
from typing import Any

from rosclaw.sense.schemas import BodyEvent, BodyRiskSummary, BodyState, max_risk


class RiskEstimator:
    """Evaluate subsystem risk levels from a BodyState snapshot."""

    def __init__(self, thresholds: dict[str, Any] | None = None):
        self.thresholds = thresholds or {}

    def evaluate(self, state: BodyState) -> tuple[BodyRiskSummary, list[BodyEvent]]:
        """Return (BodyRiskSummary, list[BodyEvent]) for the given state."""
        events: list[BodyEvent] = []

        power_risk, power_events = self._power_risk(state)
        events.extend(power_events)

        thermal_risk, thermal_events = self._thermal_risk(state)
        events.extend(thermal_events)

        actuator_risk, actuator_events = self._actuator_risk(state)
        events.extend(actuator_events)

        balance_risk, balance_events = self._balance_risk(state)
        events.extend(balance_events)

        contact_risk, contact_events = self._contact_risk(state)
        events.extend(contact_events)

        perception_risk, perception_events = self._perception_risk(state)
        events.extend(perception_events)

        communication_risk, comm_events = self._communication_risk(state)
        events.extend(comm_events)

        compute_risk, compute_events = self._compute_risk(state)
        events.extend(compute_events)

        overall = max_risk(
            power_risk,
            thermal_risk,
            actuator_risk,
            balance_risk,
            contact_risk,
            perception_risk,
            communication_risk,
            compute_risk,
            "unknown",
        )

        summary = BodyRiskSummary(
            power_risk=power_risk,
            thermal_risk=thermal_risk,
            actuator_risk=actuator_risk,
            balance_risk=balance_risk,
            contact_risk=contact_risk,
            perception_risk=perception_risk,
            communication_risk=communication_risk,
            compute_risk=compute_risk,
            fatigue_risk="unknown",
            overall_risk=overall,
        )
        return summary, events

    def _bat(self, key: str, subkey: str, default: Any) -> Any:
        return self.thresholds.get(key, {}).get(subkey, default)

    def _make_event(
        self,
        state: BodyState,
        event_type: str,
        severity: str,
        affected_parts: list[str],
        measurement: dict[str, Any],
        thresholds: dict[str, Any],
        recommended_actions: list[str],
    ) -> BodyEvent:
        return BodyEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            robot_id=state.robot_id,
            timestamp=time.time(),
            type=event_type,
            severity=severity,
            source="risk_estimator",
            affected_parts=affected_parts,
            measurement=measurement,
            thresholds=thresholds,
            recommended_actions=recommended_actions,
        )

    def _power_risk(self, state: BodyState) -> tuple[str, list[BodyEvent]]:
        battery = state.energy.battery_percent
        if battery is None:
            return "unknown", []
        low = self._bat("battery", "low", 25.0)
        critical = self._bat("battery", "critical", 10.0)
        if battery < critical:
            return "critical", [
                self._make_event(
                    state,
                    "critical_battery",
                    "critical",
                    ["battery"],
                    {"battery_percent": battery},
                    {"critical": critical},
                    ["recharge_immediately", "stop_all_motion"],
                )
            ]
        if battery < low:
            return "medium", [
                self._make_event(
                    state,
                    "low_battery",
                    "medium",
                    ["battery"],
                    {"battery_percent": battery},
                    {"low": low},
                    ["recharge_soon", "avoid_high_power_tasks"],
                )
            ]
        return "low", []

    def _thermal_risk(self, state: BodyState) -> tuple[str, list[BodyEvent]]:
        temps = {
            name: j.temperature_c
            for name, j in state.joints.items()
            if j.temperature_c is not None
        }
        if not temps:
            return "unknown", []

        warm = self._bat("joint_temperature_c", "warm", 65.0)
        hot = self._bat("joint_temperature_c", "hot", 75.0)
        overheat = self._bat("joint_temperature_c", "overheat", 85.0)

        max_joint = max(temps, key=lambda k: temps[k])
        max_temp = temps[max_joint]

        if max_temp >= overheat:
            return "critical", [
                self._make_event(
                    state,
                    "joint_overheat",
                    "critical",
                    [max_joint],
                    {"temperature_c": max_temp, "joint": max_joint},
                    {"overheat": overheat},
                    ["stop_motion", "cool_down", "check_cooling_system"],
                )
            ]
        if max_temp >= hot:
            return "high", [
                self._make_event(
                    state,
                    "joint_hot",
                    "high",
                    [max_joint],
                    {"temperature_c": max_temp, "joint": max_joint},
                    {"hot": hot},
                    ["cool_down", "reduce_load"],
                )
            ]
        if max_temp >= warm:
            return "medium", [
                self._make_event(
                    state,
                    "joint_warm",
                    "low",
                    [max_joint],
                    {"temperature_c": max_temp, "joint": max_joint},
                    {"warm": warm},
                    ["monitor_temperature"],
                )
            ]
        return "low", []

    def _actuator_risk(self, state: BodyState) -> tuple[str, list[BodyEvent]]:
        if not state.joints:
            return "unknown", []
        bad = self._bat("tracking_error", "bad", 0.25)
        degraded = self._bat("tracking_error", "degraded", 0.10)
        events: list[BodyEvent] = []
        worst = "low"
        for name, joint in state.joints.items():
            err = joint.tracking_error
            if err is None:
                continue
            if err >= bad:
                worst = max_risk(worst, "critical")
                events.append(
                    self._make_event(
                        state,
                        "tracking_error_high",
                        "high",
                        [name],
                        {"tracking_error": err, "joint": name},
                        {"bad": bad},
                        ["check_actuator", "retune_controller"],
                    )
                )
            elif err >= degraded:
                worst = max_risk(worst, "medium")
        return worst, events

    def _balance_risk(self, state: BodyState) -> tuple[str, list[BodyEvent]]:
        margin = state.balance.support_margin
        if margin is None:
            return "unknown", []
        low = self._bat("support_margin", "low", 0.12)
        ok = self._bat("support_margin", "ok", 0.18)
        if margin < low:
            return "high", [
                self._make_event(
                    state,
                    "support_unstable",
                    "high",
                    ["base", "support_polygon"],
                    {"support_margin": margin},
                    {"low": low, "ok": ok},
                    ["widen_stance", "lower_com", "enter_sandbox_only"],
                )
            ]
        if margin < ok:
            return "medium", [
                self._make_event(
                    state,
                    "fall_risk_high",
                    "medium",
                    ["base", "support_polygon"],
                    {"support_margin": margin},
                    {"low": low, "ok": ok},
                    ["stabilize_before_motion"],
                )
            ]
        return "low", []

    def _contact_risk(self, state: BodyState) -> tuple[str, list[BodyEvent]]:
        if not state.contact:
            return "unknown", []
        for name, contact in state.contact.items():
            if contact.slip_risk in ("high", "critical"):
                return "high", [
                    self._make_event(
                        state,
                        "slip_risk_high",
                        "high",
                        [name],
                        {"slip_risk": contact.slip_risk, "contact": contact.contact},
                        {},
                        ["check_terrain", "reduce_speed"],
                    )
                ]
        return "low", []

    def _perception_risk(self, state: BodyState) -> tuple[str, list[BodyEvent]]:
        events: list[BodyEvent] = []
        worst = "low"
        fps_min = self._bat("perception", "camera_fps_min", 15.0)
        conf_min = self._bat("perception", "target_confidence_min", 0.75)

        fps = state.perception.front_camera_fps
        if fps is not None and fps < fps_min:
            worst = "medium"
            events.append(
                self._make_event(
                    state,
                    "camera_obstructed",
                    "medium",
                    ["front_camera"],
                    {"front_camera_fps": fps},
                    {"camera_fps_min": fps_min},
                    ["clean_lens", "check_camera_feed"],
                )
            )

        conf = state.perception.target_detector_confidence
        if conf is not None and conf < conf_min:
            worst = "medium" if worst != "high" else "high"
            events.append(
                self._make_event(
                    state,
                    "sensor_degraded",
                    "medium",
                    ["target_detector"],
                    {"target_detector_confidence": conf},
                    {"target_confidence_min": conf_min},
                    ["re_detect_target", "check_detector_model"],
                )
            )

        if state.perception.camera_obstructed:
            worst = "high"
            events.append(
                self._make_event(
                    state,
                    "camera_obstructed",
                    "high",
                    ["front_camera"],
                    {"camera_obstructed": True},
                    {},
                    ["clear_camera_view"],
                )
            )

        if fps is None and conf is None and state.perception.status == "unknown":
            return "unknown", []

        return worst, events

    def _communication_risk(self, state: BodyState) -> tuple[str, list[BodyEvent]]:
        latency = state.communication.dds_latency_ms
        if latency is None:
            return "unknown", []
        degraded = self._bat("dds_latency_ms", "degraded", 50.0)
        bad = self._bat("dds_latency_ms", "bad", 100.0)
        if latency >= bad:
            return "high", [
                self._make_event(
                    state,
                    "dds_latency_high",
                    "high",
                    ["dds", "communication"],
                    {"dds_latency_ms": latency},
                    {"bad": bad},
                    ["reduce_loop_rate", "check_network"],
                )
            ]
        if latency >= degraded:
            return "medium", [
                self._make_event(
                    state,
                    "dds_latency_high",
                    "medium",
                    ["dds", "communication"],
                    {"dds_latency_ms": latency},
                    {"degraded": degraded},
                    ["monitor_latency"],
                )
            ]
        return "low", []

    def _compute_risk(self, state: BodyState) -> tuple[str, list[BodyEvent]]:
        cpu = state.compute.cpu_usage_percent
        mem = state.compute.memory_usage_percent
        if cpu is None and mem is None:
            return "unknown", []
        events: list[BodyEvent] = []
        worst = "low"
        cpu_high = self._bat("compute", "cpu_usage_high", 85.0)
        cpu_crit = self._bat("compute", "cpu_usage_critical", 95.0)
        mem_crit = self._bat("compute", "memory_usage_critical", 95.0)

        if cpu is not None and cpu >= cpu_crit:
            worst = "critical"
            events.append(
                self._make_event(
                    state,
                    "compute_overload",
                    "critical",
                    ["compute"],
                    {"cpu_usage_percent": cpu},
                    {"critical": cpu_crit},
                    ["reduce_compute_load", "throttle_perception"],
                )
            )
        elif cpu is not None and cpu >= cpu_high:
            worst = "medium"

        if mem is not None and mem >= mem_crit:
            worst = max_risk(worst, "critical")
            events.append(
                self._make_event(
                    state,
                    "compute_overload",
                    "critical",
                    ["memory"],
                    {"memory_usage_percent": mem},
                    {"critical": mem_crit},
                    ["free_memory", "restart_noncritical_processes"],
                )
            )

        return worst, events
