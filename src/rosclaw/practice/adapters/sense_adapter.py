"""Sense adapter: turn BodyState snapshots into practice events."""

from __future__ import annotations

from collections.abc import Iterable
from time import monotonic_ns
from typing import Any

from rosclaw.practice.adapters.base import SourceAdapter, SourceHealth
from rosclaw.practice.schemas import (
    FootContactPayload,
    IMUPayload,
    JointStatePayload,
    PracticeEventEnvelope,
)


class SenseAdapter(SourceAdapter):
    """Bridge from SenseRuntime/MockCollector to practice events."""

    source_name = "dds"

    def __init__(self, robot_id: str, sense_runtime: Any | None = None, scenario: str = "normal"):
        self._robot_id = robot_id
        self._sense_runtime = sense_runtime
        self._scenario = scenario
        self._practice_id: str | None = None
        self._running = False
        self._collector: Any | None = None

    def start(self, session: Any) -> None:
        self._practice_id = getattr(session, "practice_id", None)
        self._running = True
        if self._sense_runtime is None:
            from rosclaw.sense.collectors.mock_collector import MockCollector
            self._collector = MockCollector(robot_id=self._robot_id, scenario=self._scenario)
        else:
            self._collector = getattr(self._sense_runtime, "collector", None)

    def stop(self) -> None:
        self._running = False

    def health(self) -> SourceHealth:
        return SourceHealth(source=self.source_name, healthy=True)

    def poll(self) -> Iterable[PracticeEventEnvelope]:
        if not self._running or self._practice_id is None or self._collector is None:
            return

        try:
            body_state = self._collector.collect()
        except Exception as e:
            yield PracticeEventEnvelope(
                practice_id=self._practice_id,
                robot_id=self._robot_id,
                source="dds",
                event_type="dds.collect_error",
                payload={"error": str(e)},
            )
            return

        ts_ns = monotonic_ns()
        joints = body_state.joints
        if joints:
            names = list(joints.keys())
            payload = JointStatePayload(
                joint_names=names,
                position=[j.position_rad or 0.0 for j in joints.values()],
                velocity=[j.velocity_rad_s or 0.0 for j in joints.values()],
                effort=[j.torque_nm or 0.0 for j in joints.values()],
                temperature=[j.temperature_c or 0.0 for j in joints.values()],
            )
            yield PracticeEventEnvelope(
                practice_id=self._practice_id,
                robot_id=self._robot_id,
                source="dds",
                event_type="dds.joint_state",
                timestamp_ns=ts_ns,
                payload=payload.model_dump(),
            )

        imu = body_state.imu
        if imu and imu.angular_velocity:
            payload = IMUPayload(
                angular_velocity_xyz=imu.angular_velocity or [0.0, 0.0, 0.0],
                linear_acceleration_xyz=[0.0, 0.0, 9.81],
            )
            yield PracticeEventEnvelope(
                practice_id=self._practice_id,
                robot_id=self._robot_id,
                source="dds",
                event_type="dds.imu",
                timestamp_ns=ts_ns,
                payload=payload.model_dump(),
            )

        contact = body_state.contact
        if contact:
            left = contact.get("left_foot")
            right = contact.get("right_foot")
            payload = FootContactPayload(
                left_contact=bool(left.contact) if left else False,
                right_contact=bool(right.contact) if right else False,
                left_force_n=left.confidence if left else None,
                right_force_n=right.confidence if right else None,
            )
            yield PracticeEventEnvelope(
                practice_id=self._practice_id,
                robot_id=self._robot_id,
                source="dds",
                event_type="dds.foot_contact",
                timestamp_ns=ts_ns,
                payload=payload.model_dump(),
            )

    def on_event(self, callback: Any) -> None:
        pass
