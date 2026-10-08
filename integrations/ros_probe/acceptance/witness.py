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
from types import SimpleNamespace

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
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from ros_gz_interfaces.msg import Contacts
from runtime_policy import load_frozen_sim_runtime_policy
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


def rejected_physics_source(wire, error):
    """Retain bounded original rejected DDS bytes without inventing an admitted packet."""
    import hashlib

    record = {
        "error": str(error)[:512],
        "evidence_role": "actual_rejected_source_not_observation_or_credit",
        "source_admitted": False,
    }
    try:
        raw = wire.encode("utf-8")
    except (AttributeError, UnicodeError):
        return {**record, "source_bytes_complete": False, "source_encoding_fault": True}
    record.update(packet_sha256=hashlib.sha256(raw).hexdigest(), original_size_bytes=len(raw))
    if len(raw) > 262144:
        return {**record, "source_bytes_complete": False, "source_omission": "bounded_event_budget"}
    return {**record, "source_bytes_complete": True, "raw_packet_utf8": wire}


class Witness(Node):
    def __init__(self):
        super().__init__("rosclaw_sim_witness")
        self.runtime_policy = load_frozen_sim_runtime_policy(Path("/evidence"))
        self.generic_runtime = self.runtime_policy is not None
        if self.generic_runtime:
            policy = self.runtime_policy["policy"]
            self.profile = SimpleNamespace(
                simulation_model=policy["body_model_name"],
                physical_radius_m=self.runtime_policy["body"]["physical_radius_m"],
            )
            self.topics = policy["topics"]
            self.base_frame = self.runtime_policy["body"]["base_frame"]
            self.ground_models = tuple(policy["ground_model_names"])
        else:
            from profiles import PROFILES

            saved_profile = json.loads(Path("/evidence/fixture_profile.json").read_text())
            self.profile = PROFILES[saved_profile["name"]]
            if saved_profile != self.profile.to_dict():
                raise ValueError("fixture observer profile differs from supported geometry")
            self.base_frame = "base_footprint"
            self.ground_models = ("floor",)
            self.topics = {
                "observation": "/rosclaw_sim/observation",
                "physics": "/rosclaw_sim/physics_snapshot",
                "brush_events": "/rosclaw_sim/brush_events",
                "cleaning_state": "/rosclaw_sim/cleaning_state",
                "localization": "/amcl_pose",
                "map": "/map",
                "nav_velocity": "/nav_cmd_vel",
                "drive_velocity": "/drive_controller/cmd_vel",
            }
        self.split_actuator = self.declare_parameter("split_actuator", False).value
        self.brush_timeline, self.brush_fault = None, None
        if self.split_actuator:
            from sim_actuator import load_binding

            self.brush_timeline = BrushStateTimeline(**load_binding("/evidence/brush_binding.json"))
        self.dynamic_physics = self.declare_parameter("dynamic_physics", False).value
        if self.generic_runtime and not (self.split_actuator and self.dynamic_physics):
            raise ValueError(
                "generic observer requires split actuator and actual component physics"
            )
        self.physics_projector = self.physics_binding = None
        self.physics_queue = deque()
        self.physics_sequence = self.physics_time = self.physics_last_received = None
        self.physics_fault = None
        self.body_contact_mapping_hash = None
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
        if self.generic_runtime:
            self.contact_topics = self.runtime_policy["contact_topics"]
            self.support_topics = self.runtime_policy["policy"]["support_contact_topics"]
        else:
            self.contact_topics = json.loads(Path("/evidence/contact_topics.json").read_text())
            self.support_topics = [t for t in self.contact_topics if "/wheel_" in t]
            if len(self.support_topics) != 2:
                raise ValueError(
                    "fixture requires two independently observed wheel contact streams"
                )
        self.contact_seen = {}
        self.contact_active = {}
        self.contact_fault = None
        self.contact_sim_stamps = {}
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
        self.publisher = self.create_publisher(String, self.topics["observation"], 10)
        self.cleaning_state = (
            None
            if self.split_actuator
            else self.create_publisher(Bool, self.topics["cleaning_state"], 10)
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
            self.create_subscription(String, self.topics["physics"], self.physics_event, 16)
        else:
            self.create_subscription(
                TFMessage,
                "/rosclaw_sim/ground_truth",
                self.observe,
                QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
                callback_group=self.pose_callbacks,
            )
        self.create_subscription(
            PoseWithCovarianceStamped, self.topics["localization"], self.localized, 10
        )
        if self.split_actuator:
            self.create_subscription(String, self.topics["brush_events"], self.brush_event, 2048)
        else:
            self.create_subscription(Twist, self.topics["nav_velocity"], self.command, 10)
        if not self.generic_runtime:
            self.create_subscription(
                CollisionMonitorState,
                "/collision_monitor_state",
                self.collision_monitor_observation,
                10,
            )
            for topic in ["/cmd_vel_smoothed", self.topics["nav_velocity"]]:
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
        if not self.generic_runtime:
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
            self.topics["map"],
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
                required_body_reference_link=binding.get("body_reference_link"),
                required_body_contact_mapping=tuple(binding["body_contact_mapping"])
                if "body_contact_mapping" in binding
                else None,
            )
            if self.generic_runtime:
                actual_mapping_hash = decoded["body_contact_mapping_hash"]
                if self.body_contact_mapping_hash is None:
                    self.body_contact_mapping_hash = actual_mapping_hash
                elif actual_mapping_hash != self.body_contact_mapping_hash:
                    raise ValueError("actual Body contact component identities changed")
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
                                "body_contact_mapping_hash": self.body_contact_mapping_hash,
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
            self.plan_audit.emit(
                "physics_snapshot_rejected", rejected_physics_source(message.data, exc)
            )

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
                    "origin_orientation": [
                        message.info.origin.orientation.x,
                        message.info.origin.orientation.y,
                        message.info.origin.orientation.z,
                        message.info.origin.orientation.w,
                    ],
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
                "frame_id": self.base_frame,
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

    def validate_generic_contact(self, topic, message):
        streams = self.runtime_policy["policy"]["collision_streams"]
        rows = [row for row in streams if row["topic"] == topic]
        if len(rows) != 1 or not 0 <= len(message.contacts) <= 4096:
            raise ValueError("bounded unique mapped actual contact stream required")
        row = rows[0]
        model = self.runtime_policy["policy"]["body_model_name"]
        identity = "::".join((model, row["link"], row["collision_name"]))
        stamp = message.header.stamp
        if (
            type(stamp.sec) is not int
            or stamp.sec < 0
            or type(stamp.nanosec) is not int
            or not 0 <= stamp.nanosec < 1_000_000_000
        ):
            raise ValueError("actual contact SIM timestamp required")
        sim_time = stamp.sec + stamp.nanosec / 1e9
        age = self.get_clock().now().nanoseconds / 1e9 - sim_time
        if not -0.1 <= age < 0.3 or sim_time < self.contact_sim_stamps.get(topic, 0):
            raise ValueError("actual contact timestamp stale/future/reversed")
        if topic in self.support_topics and not message.contacts:
            raise ValueError("declared continuous support requires actual ground contact")
        for contact in message.contacts:
            names = (contact.collision1.name, contact.collision2.name)
            if (
                any(type(name) is not str or not 1 <= len(name) <= 1024 for name in names)
                or identity not in names
            ):
                raise ValueError("actual contact belongs to another Body collision/source")
        self.contact_sim_stamps[topic] = sim_time

    def contacts(self, topic, message):
        if self.generic_runtime:
            try:
                self.validate_generic_contact(topic, message)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                with self.observation_lock:
                    self.contact_fault = self.contact_fault or str(exc)
                return
        touching = any(
            not any(
                name.startswith(model + "::")
                for name in (c.collision1.name, c.collision2.name)
                for model in self.ground_models
            )
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
        touching = (
            not self.generic_runtime
            and max(abs(pose["x"]), abs(pose["y"])) + self.profile.physical_radius_m >= 1.5
        )
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
            "geometry_collision_count": None if self.generic_runtime else self.collision_count,
            "physics_collision_count": physics_collision_count,
            "collision_source": (
                "gazebo_contacts_with_explicit_URDF_stream_policy"
                if self.generic_runtime
                else "gazebo_contacts_and_ground_truth_geometry"
            ),
            # Non-contacting sensors publish only on contact. Wheel/ground
            # contacts continuously witness the live physics contact pipeline.
            "ground_truth_age_ms": (now - last_pose) * 1000,
            "observation_complete": (
                not self.split_actuator or (self.brush_fault is None and brush_pair is not None)
            )
            and not (self.dynamic_physics and self.physics_fault)
            and self.contact_fault is None
            and 0 <= now - last_pose < 0.3
            and all(0 <= now - contact_seen.get(t, 0) < 1 for t in self.support_topics),
            "contact_stream_ages_ms": {t: (now - seen) * 1000 for t, seen in contact_seen.items()},
            "contact_source_fault": self.contact_fault,
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
