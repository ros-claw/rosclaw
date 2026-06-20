"""Mock collector providing deterministic BodyState scenarios for tests and demos."""

from __future__ import annotations

import time
from collections.abc import Callable

from rosclaw.sense.collectors.base import BodyStateCollector
from rosclaw.sense.schemas import (
    BalanceState,
    BodyState,
    CommunicationState,
    ComputeState,
    EnergyState,
    FootContactState,
    IMUState,
    JointState,
    PerceptionHealth,
)

SCENARIOS = frozenset({
    "normal",
    "low_battery",
    "critical_battery",
    "hot_knee",
    "overheat_joint",
    "unstable_support",
    "camera_degraded",
    "dds_latency_high",
    "kick_not_ready",
    "compute_overload",
    "unknown_partial",
})


class MockCollector(BodyStateCollector):
    """Generate synthetic BodyState snapshots for testing and demos.

    Supported scenarios:
      - normal: nominal body state
      - low_battery: battery at 20%
      - critical_battery: battery at 8%
      - hot_knee: right knee at 78.2C
      - overheat_joint: right knee at 88C
      - unstable_support: support_margin 0.09
      - camera_degraded: target confidence 0.55, low FPS
      - dds_latency_high: DDS latency 120ms
      - kick_not_ready: combination of hot knee + low confidence + unstable support
      - compute_overload: CPU 96%
      - unknown_partial: most fields missing
    """

    name = "mock"

    def __init__(
        self,
        robot_id: str = "g1_lab_01",
        scenario: str = "normal",
    ):
        self.robot_id = robot_id
        self.scenario = scenario
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown mock scenario: {scenario}")

    def collect(self) -> BodyState:
        """Return a BodyState snapshot for the configured scenario."""
        builder = _SCENARIO_BUILDERS.get(self.scenario, _build_normal)
        state = builder(self.robot_id)
        state.source = f"mock:{self.scenario}"
        return state


def _joints_normal() -> dict[str, JointState]:
    return {
        "left_hip_pitch": JointState(position_rad=0.1, temperature_c=42.0),
        "left_knee": JointState(position_rad=0.5, temperature_c=43.0),
        "left_ankle": JointState(position_rad=-0.2, temperature_c=40.0),
        "right_hip_pitch": JointState(position_rad=-0.1, temperature_c=44.0),
        "right_knee": JointState(position_rad=0.6, temperature_c=45.0),
        "right_ankle": JointState(position_rad=-0.25, temperature_c=41.0),
    }


def _build_normal(robot_id: str) -> BodyState:
    return BodyState(
        robot_id=robot_id,
        timestamp=time.time(),
        energy=EnergyState(battery_percent=80.0, voltage=24.0, power_mode="normal"),
        joints=_joints_normal(),
        imu=IMUState(pitch_deg=2.0, roll_deg=1.0, yaw_deg=0.0),
        contact={
            "left_foot": FootContactState(contact=True, confidence=0.99, slip_risk="low"),
            "right_foot": FootContactState(contact=True, confidence=0.99, slip_risk="low"),
        },
        communication=CommunicationState(
            dds_latency_ms=18.0,
            packet_loss=0.0,
            heartbeat_ok=True,
        ),
        perception=PerceptionHealth(
            front_camera_fps=30.0,
            target_detector_confidence=0.92,
            status="ok",
        ),
        balance=BalanceState(
            support_margin=0.25,
            com_projection="center",
            fall_risk_raw=0.05,
            stable_for_sec=2.0,
        ),
        compute=ComputeState(
            cpu_usage_percent=35.0,
            memory_usage_percent=40.0,
        ),
    )


def _build_low_battery(robot_id: str) -> BodyState:
    state = _build_normal(robot_id)
    state.energy.battery_percent = 20.0
    return state


def _build_critical_battery(robot_id: str) -> BodyState:
    state = _build_normal(robot_id)
    state.energy.battery_percent = 8.0
    return state


def _build_hot_knee(robot_id: str) -> BodyState:
    state = _build_normal(robot_id)
    state.joints["right_knee"].temperature_c = 78.2
    return state


def _build_overheat_joint(robot_id: str) -> BodyState:
    state = _build_normal(robot_id)
    state.joints["right_knee"].temperature_c = 88.0
    return state


def _build_unstable_support(robot_id: str) -> BodyState:
    state = _build_normal(robot_id)
    state.balance.support_margin = 0.09
    state.balance.stable_for_sec = 0.3
    return state


def _build_camera_degraded(robot_id: str) -> BodyState:
    state = _build_normal(robot_id)
    state.perception.front_camera_fps = 8.0
    state.perception.target_detector_confidence = 0.55
    state.perception.status = "degraded"
    return state


def _build_dds_latency_high(robot_id: str) -> BodyState:
    state = _build_normal(robot_id)
    state.communication.dds_latency_ms = 120.0
    return state


def _build_kick_not_ready(robot_id: str) -> BodyState:
    state = _build_normal(robot_id)
    state.joints["right_knee"].temperature_c = 78.2
    state.balance.support_margin = 0.09
    state.balance.stable_for_sec = 0.3
    state.perception.target_detector_confidence = 0.71
    state.energy.battery_percent = 33.0
    return state


def _build_compute_overload(robot_id: str) -> BodyState:
    state = _build_normal(robot_id)
    state.compute.cpu_usage_percent = 96.0
    state.compute.cpu_temp_c = 82.0
    state.compute.memory_usage_percent = 91.0
    return state


def _build_unknown_partial(robot_id: str) -> BodyState:
    return BodyState(
        robot_id=robot_id,
        timestamp=time.time(),
        source="mock:unknown_partial",
        raw={"diagnostic": "partial_observability"},
    )


_SCENARIO_BUILDERS: dict[str, Callable[[str], BodyState]] = {
    "normal": _build_normal,
    "low_battery": _build_low_battery,
    "critical_battery": _build_critical_battery,
    "hot_knee": _build_hot_knee,
    "overheat_joint": _build_overheat_joint,
    "unstable_support": _build_unstable_support,
    "camera_degraded": _build_camera_degraded,
    "dds_latency_high": _build_dds_latency_high,
    "kick_not_ready": _build_kick_not_ready,
    "compute_overload": _build_compute_overload,
    "unknown_partial": _build_unknown_partial,
}
