"""One original-event engine for independent live observation and closed replay.

This engine cannot move a robot or a probe. Logical source readiness is an
additional actor constraint; actual Body/world admission remains independent.
"""

import json
import math

from actuator_observation_constraint import observation_envelope
from backend_source_gate import BackendObservationGate
from closed_native_contact_evidence import decode_ros_pose, original_ros_bytes

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest


class BackendObserverReplay:
    def __init__(
        self,
        robot_policy,
        probe_policy,
        *,
        robot_pose_frame,
        probe_pose_frame,
        scene_binding=None,
        probe_declaration=None,
    ):
        if any(
            type(v) is not str or not 0 < len(v) <= 256
            for v in (robot_pose_frame, probe_pose_frame)
        ):
            raise ValueError("explicit actual robot and instrument world frames required")
        self.gate = BackendObservationGate(robot_policy, probe_policy, probe_pose_frame)
        self.robot_frame = robot_pose_frame
        self.binding = robot_policy["contact_policy"]
        self.last_wall = None
        self.sequence = 0
        self.spatial = None
        if (scene_binding is None) != (probe_declaration is None):
            raise ValueError("both frozen scene binding and probe declaration required")
        if scene_binding is not None:
            from probe_scene_geometry import ProbeSceneGeometry
            from probe_scene_join import ProbeSceneJoin

            geometry = ProbeSceneGeometry(scene_binding, probe_declaration, self.gate)
            self.spatial = ProbeSceneJoin(geometry)
            self.gate.policy_hash = digest(
                {
                    "schema_version": "rosclaw.spatial_backend_observer_constraint.v2",
                    "contact_constraint_policy_hash": self.gate.policy_hash,
                    "scene_geometry_policy_hash": geometry.policy_hash,
                }
            )

    def apply(self, kind, payload):
        try:
            if self.gate.fault:
                raise ValueError("joint original observation rejection remains latched")
            wall = payload.get("received_monotonic_sec")
            if (
                type(wall) not in (int, float)
                or not math.isfinite(wall)
                or wall < 0
                or (self.last_wall is not None and wall < self.last_wall)
            ):
                raise ValueError("joint original event receipt clock regressed")
            self.last_wall = wall
            if kind == "backend_scene_components":
                if self.spatial is None:
                    raise ValueError("unregistered original scene source refused")
                raw = original_ros_bytes(payload, "gazebo_ecm_postupdate_observation_json")
                joined = self.spatial.receive(
                    "scene",
                    raw,
                    received_monotonic_sec=wall,
                    received_unix_ns=payload["received_unix_ns"],
                )
                return {
                    "sim_time_sec": json.loads(raw)["sim_time_sec"],
                    "joined_spatial_sources": joined,
                }
            joined = None
            if self.spatial is not None and kind in {
                "backend_robot_components",
                "backend_probe_components",
            }:
                raw = original_ros_bytes(payload, "gazebo_ecm_contact_sensor_data_json")
                joined = self.spatial.receive(
                    "robot" if kind == "backend_robot_components" else "probe",
                    raw,
                    received_monotonic_sec=wall,
                    received_unix_ns=payload["received_unix_ns"],
                )
            if kind == "backend_robot_pose":
                raw = original_ros_bytes(payload, "tf2_msgs/msg/TFMessage")
                sim, pose = decode_ros_pose(raw, self.binding["model_name"], self.robot_frame)
                self.gate.robot.pose(sim, wall, pose)
                return {"sim_time_sec": sim}
            if kind == "backend_robot_components":
                raw = original_ros_bytes(payload, "gazebo_ecm_contact_sensor_data_json")
                if self.gate.robot.world_pose is None:
                    # Retain this original startup packet; never infer empty/zero.
                    return {
                        "startup_pose_pending": True,
                        **({"joined_spatial_sources": joined} if joined is not None else {}),
                    }
                result = self.gate.robot.observe(
                    raw,
                    received_monotonic_sec=wall,
                    received_unix_ns=payload["received_unix_ns"],
                )
                return {
                    **result,
                    **({"joined_spatial_sources": joined} if joined is not None else {}),
                }
            if kind in {
                "backend_probe_pose",
                "backend_probe_components",
                "backend_probe_lift_ack",
                "backend_probe_snapshot",
            }:
                result = self.gate.probe.apply(kind, payload)
                return {
                    **result,
                    **({"joined_spatial_sources": joined} if joined is not None else {}),
                }
            if kind == "backend_observation_sample":
                snapshot = self._snapshot(wall)
                envelope = observation_envelope(snapshot, self.binding, self.sequence)
                self.sequence += 1
                return {"snapshot": snapshot, "actor_envelope": envelope}
            raise ValueError("unknown original joint observation event")
        except (ValueError, TypeError, KeyError, UnicodeError) as exc:
            self.gate.fault = self.gate.fault or str(exc)
            raise ValueError(f"joint original observation source rejected: {exc}") from exc

    def failed_sample(self, wall):
        """Publish only a closed negative projection after source rejection."""
        snapshot = self._snapshot(wall)
        if snapshot["live_source_constraint_satisfied"]:
            raise ValueError("faulted joint observation cannot produce ready projection")
        envelope = observation_envelope(snapshot, self.binding, self.sequence)
        self.sequence += 1
        return {"snapshot": snapshot, "actor_envelope": envelope}

    def _snapshot(self, wall):
        snapshot = self.gate.snapshot(wall)
        if self.spatial is not None:
            spatial = self.spatial.snapshot(wall)
            snapshot["scene_geometry_constraint"] = spatial
            snapshot["live_source_constraint_satisfied"] = bool(
                snapshot["live_source_constraint_satisfied"]
                and spatial["scene_geometry_constraint_satisfied"]
            )
            fault = spatial["source_fault"] or spatial["join_source_fault"]
            if fault:
                self.gate.fault = self.gate.fault or fault
                snapshot["source_fault"] = self.gate.fault
                snapshot["live_source_constraint_satisfied"] = False
        return snapshot
