"""Gazebo fixture observer and simulated cleaning/deadman controller.

Runs only in the isolated simulation container. Ground truth is received from
Gazebo, independently of Nav2 localization and action results.
"""

import json
import math
import time
from collections import deque
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

import rclpy
from geometry_msgs.msg import (
    PointStamped,
    PolygonStamped,
    PoseWithCovarianceStamped,
    Twist,
    TwistStamped,
)
from nav2_msgs.msg import CollisionMonitorState
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path as NavPath
from profiles import PROFILES
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from ros_gz_interfaces.msg import Contacts
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker

from rosclaw.connectors.ros.diagnosis.coverage_audit import CoverageAuditLog, digest
from rosclaw.connectors.ros.verification.brush_timeline import BrushStateEvent, BrushStateTimeline
from rosclaw.connectors.ros.verification.coverage import CoverageVerifier
from rosclaw.connectors.ros.verification.occupancy_geometry import (
    OccupancyProjector,
    parse_physics_packet,
)


class Witness(Node):
    def __init__(self):
        super().__init__("rosclaw_sim_witness")
        saved_profile = json.loads(Path("/evidence/fixture_profile.json").read_text())
        self.profile = PROFILES[saved_profile["name"]]
        if saved_profile != self.profile.to_dict():
            raise ValueError("fixture observer profile differs from supported geometry")
        self.split_actuator = self.declare_parameter("split_actuator", False).value
        self.brush_timeline, self.brush_fault = None, None
        if self.split_actuator:
            from sim_actuator import load_binding

            self.brush_timeline = BrushStateTimeline(**load_binding("/evidence/brush_binding.json"))
        self.dynamic_physics = self.declare_parameter("dynamic_physics", False).value
        self.physics_projector = self.physics_binding = None
        self.physics_queue = deque()
        self.physics_sequence = self.physics_time = self.physics_last_received = None
        self.physics_fault = None
        self.physics_map_verified = False
        if self.dynamic_physics:
            if not self.split_actuator:
                raise ValueError("dynamic evidence requires separate passive observer/actuator")
            binding = json.loads(Path("/evidence/physics_binding.json").read_text())
            if (
                binding.get("frame_transform_source") != "simulator_operator_fixture_policy"
                or binding.get("world_to_map_xyyaw") != [0, 0, 0]
                or binding.get("map_world_identity_approved") is not True
                or type(binding.get("mission_id")) is not str
                or not binding["mission_id"]
                or any(
                    binding.get(k) != v
                    for k, v in zip(
                        ("run_id", "body_snapshot_hash", "attachment_hash"),
                        self.brush_timeline.binding[:3],
                        strict=True,
                    )
                )
            ):
                raise ValueError(
                    "explicit frozen known-fixture world/map identity and source binding required"
                )
            self.physics_binding = binding
            self.physics_grid = CoverageVerifier(**binding["grid"])
        self.cleaning = False
        self.observation_lock = Lock()
        self.pose = None
        self.localization = None
        self.last_pose = 0
        self.lease = 0
        self.lease_updates = 0
        self.collision_count = 0
        self.in_collision = False
        self.published_time = None
        self.contact_topics = json.loads(Path("/evidence/contact_topics.json").read_text())
        self.wheel_topics = [t for t in self.contact_topics if "/wheel_" in t]
        if len(self.wheel_topics) != 2:
            raise ValueError("fixture requires two independently observed wheel contact streams")
        self.contact_seen = {}
        self.contact_active = {}
        self.physics_collision_count = 0
        self.trace = Path("/evidence/witness.jsonl").open("a", buffering=1)  # noqa: SIM115 - node lifecycle closes it
        self.plan_audit = CoverageAuditLog(
            Path("/evidence") / f"plan-events-{time.time_ns()}.jsonl",
            context={
                "source": "passive_ros_debug_topics",
                "evidence_domain": "SIMULATION",
                "run_id": Path("/evidence/run_id.txt").read_text().strip()
                if Path("/evidence/run_id.txt").exists()
                else "UNKNOWN",
            },
        )
        self.publisher = self.create_publisher(String, "/rosclaw_sim/observation", 10)
        self.cleaning_state = (
            None
            if self.split_actuator
            else self.create_publisher(Bool, "/rosclaw_sim/cleaning_state", 10)
        )
        self.controller_watchdog = self.declare_parameter("controller_watchdog", True).value
        self.velocity = (
            None
            if self.split_actuator
            else self.create_publisher(
                TwistStamped if self.controller_watchdog else Twist,
                "/drive_controller/cmd_vel" if self.controller_watchdog else "/cmd_vel",
                10,
            )
        )
        self.control_callbacks = MutuallyExclusiveCallbackGroup()
        self.pose_callbacks = MutuallyExclusiveCallbackGroup()
        self.contact_callbacks = MutuallyExclusiveCallbackGroup()
        self.wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        if self.dynamic_physics:
            self.create_subscription(
                String, "/rosclaw_sim/physics_snapshot", self.physics_event, 16
            )
        else:
            self.create_subscription(
                TFMessage,
                "/rosclaw_sim/ground_truth",
                self.observe,
                QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
                callback_group=self.pose_callbacks,
            )
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self.localized, 10)
        if self.split_actuator:
            self.create_subscription(String, "/rosclaw_sim/brush_events", self.brush_event, 2048)
        else:
            self.create_subscription(Twist, "/nav_cmd_vel", self.command, 10)
        self.create_subscription(
            CollisionMonitorState,
            "/collision_monitor_state",
            self.collision_monitor_observation,
            10,
        )
        for topic in ["/cmd_vel_smoothed", "/nav_cmd_vel"]:
            self.create_subscription(
                Twist,
                topic,
                lambda message, t=topic: self.velocity_observation(t, message),
                10,
            )
        self.create_subscription(Bool, "/is_rotating_to_heading", self.rotation_observation, 10)
        self.create_subscription(
            NavPath,
            "/received_global_plan",
            lambda message: self.record_path("/received_global_plan", message),
            10,
        )
        for topic in ["/lookahead_point", "/curvature_lookahead_point"]:
            self.create_subscription(
                PointStamped,
                topic,
                lambda message, t=topic: self.carrot_observation(t, message),
                10,
            )
        for topic in self.contact_topics:
            self.create_subscription(
                Contacts,
                topic,
                lambda message, t=topic: self.contacts(t, message),
                10,
                callback_group=self.contact_callbacks,
            )
        self.create_subscription(NavPath, "/plan", self.path_observation, 10)
        self.create_subscription(
            NavPath, "/coverage_server/coverage_plan", self.coverage_path_observation, 10
        )
        for topic in ["/coverage_server/field_boundary", "/coverage_server/planning_field"]:
            self.create_subscription(
                PolygonStamped,
                topic,
                lambda message, t=topic: self.polygon_observation(t, message),
                10,
            )
        self.create_subscription(Marker, "/coverage_server/swaths", self.swath_observation, 10)
        self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_observation,
            QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        if not self.split_actuator:
            self.create_service(
                SetBool,
                "/rosclaw_sim/cleaning",
                self.set_cleaning,
                callback_group=self.control_callbacks,
            )
            self.create_service(
                SetBool, "/rosclaw_sim/lease", self.heartbeat, callback_group=self.control_callbacks
            )
        self.create_timer(
            0.05, self.tick, callback_group=self.control_callbacks, clock=self.wall_clock
        )

    def physics_event(self, message):
        if self.physics_fault is not None:
            return
        try:
            binding = self.physics_binding
            raw = message.data.encode("utf-8")
            packet = json.loads(raw)
            if packet.get("complete") is not True and self.physics_projector is None:
                self.plan_audit.emit(
                    "physics_startup_incomplete", packet, sim_time=packet.get("sim_time_sec")
                )
                return  # no readiness/credit while approved fixture bootstrap is incomplete
            decoded = parse_physics_packet(
                raw,
                **{
                    k: binding[k]
                    for k in (
                        "run_id",
                        "body_snapshot_hash",
                        "attachment_hash",
                        "world_name",
                        "body_model_name",
                    )
                },
                obstacle_names=tuple(binding["obstacle_names"]),
                scene_model_names=frozenset(binding["scene_model_names"]),
                maximum_body_planar_radius_m=binding.get("maximum_body_planar_radius_m"),
            )
            if (
                self.physics_sequence is not None
                and packet["sequence"] != self.physics_sequence + 1
            ):
                raise ValueError("physical producer sequence gap/reorder")
            if self.physics_time is not None and packet["sim_time_sec"] <= self.physics_time:
                raise ValueError("physical producer SIM time did not advance")
            if self.physics_projector is None:
                self.physics_projector = OccupancyProjector(self.physics_grid, decoded["geometry"])
                with Path("/evidence/physics_ready.json").open("x") as ready:
                    ready.write(
                        json.dumps(
                            {
                                "binding": binding,
                                "geometry_hash": decoded["geometry"].artifact_hash(),
                                "initial_packet_sha256": decoded["packet_sha256"],
                                "evidence_role": "actual_component_source_admission_not_mission_acceptance",
                            }
                        )
                        + "\n"
                    )
            if decoded["geometry"].artifact_hash() != self.physics_projector.geometry_hash:
                raise ValueError("actual scene collision geometry or model identity changed")
            self.physics_sequence, self.physics_time = packet["sequence"], packet["sim_time_sec"]
            self.physics_last_received = time.monotonic()
            self.plan_audit.emit(
                "physics_snapshot_received",
                {
                    "packet": packet,
                    "raw_packet_utf8": raw.decode("utf-8"),
                    "packet_sha256": decoded["packet_sha256"],
                },
                sim_time=packet["sim_time_sec"],
            )
            if packet["paused"]:
                if self.pose is not None:
                    raise ValueError("physics paused after live observation admission")
                return
            if self.brush_timeline.sequence is None or (
                self.brush_timeline.previous_pose_time is None
                and packet["sim_time_sec"] < self.brush_timeline.events[0].sim_time_sec
            ):
                return  # startup data predating OFF admission has no brush credit
            if len(self.physics_queue) >= 6:
                raise ValueError("pending physics/brush pairing queue overflow")
            self.physics_queue.append((decoded, self.physics_last_received))
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self.physics_fault = str(exc)

    def brush_event(self, message):
        if self.brush_fault is not None:
            return
        try:
            payload = json.loads(message.data)
            event = BrushStateEvent(**payload["event"])
            self.brush_timeline.append(event, artifact_hash=payload["artifact_hash"])
            remaining, updates = payload["lease_remaining_sec"], payload["lease_updates"]
            if (
                type(remaining) not in (int, float)
                or not math.isfinite(remaining)
                or remaining > 1.5
            ):
                raise ValueError("bounded actuator lease observation required")
            if type(updates) is not int or updates < self.lease_updates:
                raise ValueError("actuator lease count reversed")
            self.lease, self.lease_updates = time.monotonic() + remaining, updates
            self.plan_audit.emit("brush_state_received", payload, sim_time=event.sim_time_sec)
        except (ValueError, TypeError, KeyError) as exc:
            self.brush_fault = str(exc)
            self.cleaning = False

    def map_observation(self, message):
        if self.dynamic_physics:
            q = message.info.origin.orientation
            grid = self.physics_grid
            if (
                message.header.frame_id != grid.frame_id
                or message.info.width != grid.width
                or message.info.height != grid.height
                or message.info.resolution != grid.resolution
                or [message.info.origin.position.x, message.info.origin.position.y]
                != list(grid.origin)
                or (q.x, q.y, q.z, abs(q.w)) != (0, 0, 0, 1)
                or len(message.data) != grid.width * grid.height
                or any(message.data[i] != 0 for i in grid.accessible)
            ):
                self.physics_fault = "actual map/frame differs from frozen projection grid"
            else:
                self.physics_map_verified = True
        Path("/evidence/measured_map.json").write_text(
            json.dumps(
                {
                    "width": message.info.width,
                    "height": message.info.height,
                    "resolution": message.info.resolution,
                    "origin": [message.info.origin.position.x, message.info.origin.position.y],
                    "frame_id": message.header.frame_id,
                    "occupancy": list(message.data),
                }
            )
            + "\n"
        )

    def collision_monitor_observation(self, message):
        self.plan_audit.emit(
            "collision_monitor_state",
            {
                "topic": "/collision_monitor_state",
                "action_type": message.action_type,
                "polygon_name": message.polygon_name,
            },
            sim_time=self.get_clock().now().nanoseconds / 1e9,
        )

    def velocity_observation(self, topic, message):
        self.plan_audit.emit(
            "velocity_command",
            {
                "topic": topic,
                "frame_id": "base_footprint",
                "linear_x": message.linear.x,
                "angular_z": message.angular.z,
                "evidence_role": "command_observation_not_measured_motion",
            },
            sim_time=self.get_clock().now().nanoseconds / 1e9,
        )

    def rotation_observation(self, message):
        self.plan_audit.emit(
            "rpp_rotation_state",
            {
                "topic": "/is_rotating_to_heading",
                "rotating": message.data,
                "evidence_role": "controller_rotation_flag_not_goal_or_path_cause",
            },
            sim_time=self.get_clock().now().nanoseconds / 1e9,
        )

    def carrot_observation(self, topic, message):
        self.plan_audit.emit(
            "rpp_lookahead_point",
            {
                "topic": topic,
                "frame_id": message.header.frame_id,
                "point": {"x": message.point.x, "y": message.point.y, "z": message.point.z},
                "header_stamp_sec": message.header.stamp.sec + message.header.stamp.nanosec / 1e9,
                "evidence_role": "controller_debug_target_not_measured_pose",
            },
            sim_time=self.get_clock().now().nanoseconds / 1e9,
        )

    def path_observation(self, message):
        self.record_path("/plan", message)

    def coverage_path_observation(self, message):
        self.record_path("/coverage_server/coverage_plan", message)

    def record_path(self, topic, message):
        poses = []
        for p in message.poses:
            q = p.pose.orientation
            poses.append(
                {
                    "x": p.pose.position.x,
                    "y": p.pose.position.y,
                    "yaw": math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)),
                }
            )
        payload = {
            "topic": topic,
            "frame_id": message.header.frame_id,
            "poses": poses,
            "plan_id": digest(
                {
                    "topic": topic,
                    "header_stamp": [message.header.stamp.sec, message.header.stamp.nanosec],
                    "frame_id": message.header.frame_id,
                    "poses": poses,
                }
            ),
            "nav_goal_id": None,
            "goal_binding": "requires_unique_daemon_interval_correlation",
        }
        self.plan_audit.emit(
            "path", payload, sim_time=message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        )

    def polygon_observation(self, topic, message):
        self.plan_audit.emit(
            "polygon",
            {
                "topic": topic,
                "frame_id": message.header.frame_id,
                "points": [[p.x, p.y] for p in message.polygon.points],
            },
            sim_time=message.header.stamp.sec + message.header.stamp.nanosec / 1e9,
        )

    def swath_observation(self, message):
        self.plan_audit.emit(
            "swaths",
            {
                "topic": "/coverage_server/swaths",
                "frame_id": message.header.frame_id,
                "marker_type": message.type,
                "marker_action": message.action,
                "points": [[p.x, p.y] for p in message.points],
            },
            sim_time=message.header.stamp.sec + message.header.stamp.nanosec / 1e9,
        )

    def set_cleaning(self, request, response):
        if self.split_actuator:
            raise RuntimeError("passive observer has no cleaner service")
        self.cleaning = request.data
        response.success = True
        response.message = "Simulated cleaning state measured by witness"
        return response

    def localized(self, message):
        q = message.pose.pose.orientation
        self.localization = {
            "x": message.pose.pose.position.x,
            "y": message.pose.pose.position.y,
            "yaw": math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)),
            "frame_id": message.header.frame_id,
            "source": "amcl_pose",
            "captured_at": datetime.now(UTC).isoformat(),
            "time_sec": message.header.stamp.sec + message.header.stamp.nanosec / 1e9,
            "covariance": list(message.pose.covariance),
        }

    def contacts(self, topic, message):
        touching = any(
            "floor::" not in c.collision1.name and "floor::" not in c.collision2.name
            for c in message.contacts
        )
        with self.observation_lock:
            self.contact_seen[topic] = time.monotonic()
            if touching and not any(self.contact_active.values()):
                self.physics_collision_count += 1
            self.contact_active[topic] = touching

    def heartbeat(self, request, response):
        if self.split_actuator:
            raise RuntimeError("passive observer has no lease service")
        self.lease = time.monotonic() + 1.5 if request.data else 0
        self.lease_updates += 1
        response.success = True
        return response

    def command(self, message):
        if time.monotonic() < self.lease:
            self.publish_velocity(message)

    def publish_velocity(self, message):
        if self.split_actuator:
            raise RuntimeError("passive observer has no actuator publisher")
        if self.controller_watchdog:
            stamped = TwistStamped()
            stamped.header.stamp = self.get_clock().now().to_msg()
            stamped.twist = message
            self.velocity.publish(stamped)
        else:
            self.velocity.publish(message)

    def observe(self, message):
        for t in message.transforms:
            if t.child_frame_id == self.profile.simulation_model:
                q = t.transform.rotation
                pose = {
                    "x": t.transform.translation.x,
                    "y": t.transform.translation.y,
                    "yaw": math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)),
                    "time_sec": t.header.stamp.sec + t.header.stamp.nanosec / 1e9,
                }
                with self.observation_lock:
                    self.pose = pose
                    self.last_pose = time.monotonic()

    def tick(self):
        # Pose coordinates and their real receive time must be one snapshot.
        # Dedicated callbacks prevent path/map disk writes starving ground truth.
        with self.observation_lock:
            pose = self.pose
            last_pose = self.last_pose
            contact_seen = self.contact_seen.copy()
            physics_collision_count = self.physics_collision_count
            now = time.monotonic()
        if not self.split_actuator:
            if now > self.lease:
                if self.cleaning:
                    self.get_logger().warning(
                        f"Cleaning disabled by expired daemon lease: overdue={now - self.lease:.3f}s"
                    )
                self.publish_velocity(Twist())
                self.cleaning = False
            self.cleaning_state.publish(Bool(data=self.cleaning))
        decoded = None
        if self.dynamic_physics:
            if self.pose is not None and (
                self.physics_last_received is None or now - self.physics_last_received >= 0.3
            ):
                self.physics_fault = self.physics_fault or "physical producer stale"
            if self.physics_fault is None:
                if not self.physics_queue:
                    return
                decoded, last_pose = self.physics_queue[0]
                p = decoded["body_world_pose"]
                pose = {
                    "x": p[0],
                    "y": p[1],
                    "yaw": math.atan2(
                        2 * (p[3] * p[6] + p[4] * p[5]), 1 - 2 * (p[5] ** 2 + p[6] ** 2)
                    ),
                    "time_sec": decoded["packet"]["sim_time_sec"],
                }
                self.pose = pose
        if pose is None:
            return
        if self.published_time == pose["time_sec"] and not (
            self.dynamic_physics and self.physics_fault
        ):
            return
        brush_pair = None
        if (
            self.split_actuator
            and self.brush_fault is None
            and not (self.dynamic_physics and self.physics_fault)
            and (not self.dynamic_physics or self.physics_map_verified)
        ):
            if self.brush_timeline.sequence is None:
                return  # never publish readiness before the first OFF watermark
            if self.brush_timeline.previous_pose_time is None and (
                pose["time_sec"] < self.brush_timeline.events[0].sim_time_sec
            ):
                return  # startup physics predating the source has no brush evidence
            try:
                brush_pair = self.brush_timeline.state_at(pose["time_sec"], now_monotonic=now)
                if brush_pair["status"] == "PENDING":
                    return
                self.cleaning = brush_pair["enabled"]
            except (ValueError, TypeError) as exc:
                self.brush_fault = str(exc)
                self.cleaning = False
        self.published_time = pose["time_sec"]
        # Room geometry comes from the checked-in Gazebo world. A conservative
        # disc encloses this configured robot's physical footprint. This is
        # independent post-physics geometric collision observation, not Nav2's
        # prediction or a claim from the caller. Ground contact is excluded.
        touching = max(abs(pose["x"]), abs(pose["y"])) + self.profile.physical_radius_m >= 1.5
        if touching and not self.in_collision:
            self.collision_count += 1
        self.in_collision = touching
        sample = {
            **pose,
            "captured_at": datetime.now(UTC).isoformat(),
            "cleaning_enabled": self.cleaning,
            "lease_remaining_sec": self.lease - now,
            "lease_updates": self.lease_updates,
            "localization": self.localization,
            "evidence_domain": "GAZEBO_PHYSICS",
            "collision_count": max(self.collision_count, physics_collision_count),
            "geometry_collision_count": self.collision_count,
            "physics_collision_count": physics_collision_count,
            "collision_source": "gazebo_contacts_and_ground_truth_geometry",
            # Non-contacting sensors publish only on contact. Wheel/ground
            # contacts continuously witness the live physics contact pipeline.
            "ground_truth_age_ms": (now - last_pose) * 1000,
            "observation_complete": (
                not self.split_actuator or (self.brush_fault is None and brush_pair is not None)
            )
            and not (self.dynamic_physics and self.physics_fault)
            and 0 <= now - last_pose < 0.3
            and all(0 <= now - contact_seen.get(t, 0) < 1 for t in self.wheel_topics),
            "contact_stream_ages_ms": {t: (now - seen) * 1000 for t, seen in contact_seen.items()},
        }
        if self.split_actuator:
            sample["brush_state_pair"] = brush_pair
            sample["brush_source_fault"] = self.brush_fault
            sample["brush_source_binding"] = dict(
                zip(
                    ("run_id", "body_snapshot_hash", "attachment_hash", "producer_id"),
                    self.brush_timeline.binding,
                    strict=True,
                )
            )
        if self.dynamic_physics:
            sample["physics_source_fault"] = self.physics_fault
            sample["frame_transform_evidence"] = {
                k: self.physics_binding[k]
                for k in (
                    "world_name",
                    "world_to_map_xyyaw",
                    "frame_transform_source",
                    "map_world_identity_approved",
                )
            }
            if decoded is not None and self.physics_fault is None:
                try:
                    age = (time.time_ns() - decoded["packet"]["captured_at_unix_ns"]) / 1e9
                    snapshot = self.physics_projector.project(
                        decoded["model_poses"],
                        run_id=self.physics_binding["run_id"],
                        mission_id=self.physics_binding["mission_id"],
                        sequence=decoded["packet"]["sequence"],
                        frame_id=self.physics_grid.frame_id,
                        sim_time_sec=pose["time_sec"],
                        ground_truth_age_sec=age,
                        complete=True,
                    )
                    sample["occupancy"] = asdict(snapshot)
                    sample["occupancy_hash"] = snapshot.artifact_hash()
                    sample["physics_packet_sha256"] = decoded["packet_sha256"]
                    self.physics_queue.popleft()
                except (ValueError, TypeError) as exc:
                    self.physics_fault = str(exc)
                    sample["physics_source_fault"] = self.physics_fault
                    sample["observation_complete"] = False
            else:
                sample["observation_complete"] = False
        self.trace.write(json.dumps(sample) + "\n")
        self.publisher.publish(String(data=json.dumps(sample)))


def main():
    rclpy.init()
    node = Witness()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        try:
            executor.shutdown()
            if rclpy.ok() and not node.split_actuator:
                node.publish_velocity(Twist())
        finally:
            # SIGINT may have invalidated the ROS context. Diagnostic flushing
            # must not depend on a last ROS publish succeeding at shutdown.
            node.trace.close()
            summary = node.plan_audit.close()
            Path(summary["path"] + ".summary.json").write_text(json.dumps(summary) + "\n")
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
