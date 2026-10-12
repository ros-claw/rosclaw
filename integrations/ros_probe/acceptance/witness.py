"""Gazebo fixture observer and simulated cleaning/deadman controller.

Runs only in the isolated simulation container. Ground truth is received from
Gazebo, independently of Nav2 localization and action results.
"""

import hashlib
import json
import math
import os
import time
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
from rosidl_runtime_py.convert import message_to_ordereddict
from simulation_witness_sources import SimulationWitnessSources
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker

from rosclaw.body.resolver import BodyResolver
from rosclaw.connectors.ros.diagnosis.coverage_audit import CoverageAuditLog, digest
from rosclaw.connectors.ros.mission.navigation_binding import SimulationNavigationBinding


class Witness(Node):
    def __init__(self):
        super().__init__("rosclaw_sim_witness")
        self.source_only = self.declare_parameter("read_only_source_observer", False).value
        body_workspace = Path(self.declare_parameter("body_workspace", "/evidence/home").value)
        execution_config = Path(
            self.declare_parameter("execution_config_path", "/evidence/execution_config.json").value
        )
        source_path = self.declare_parameter("observer_sources_path", "").value
        source_pin = self.declare_parameter("observer_sources_sha256", "").value
        if bool(source_path) != bool(source_pin):
            raise ValueError("observer inputs require a separate operator SHA pin")
        self.sources = None
        self.profile = None
        if source_path:
            if not self.source_only:
                raise ValueError(
                    "generic observer is source-only; L2 continuity acceptance is not implemented"
                )
            self.sources = SimulationWitnessSources.from_bytes(
                Path(source_path).read_bytes(),
                expected_sha256=source_pin,
                urdf_bytes=Path("/evidence/robot.urdf").read_bytes(),
                world_sdf_bytes=Path("/evidence/world.sdf").read_bytes(),
            )
            if self.get_fully_qualified_name() != self.sources.observer_node:
                raise ValueError("actual observer identity differs from pinned source inputs")
        else:
            saved_profile = json.loads(Path("/evidence/fixture_profile.json").read_text())
            self.profile = PROFILES[saved_profile["name"]]
            if saved_profile != self.profile.to_dict():
                raise ValueError("fixture observer profile differs from supported geometry")
        self.navigation_binding = None
        config_path = execution_config
        if config_path.exists():
            config = json.loads(config_path.read_text())
            if config.get("executor_kind") == "navigation_only":
                effective = BodyResolver(
                    workspace=body_workspace, body_id=config["body_id"]
                ).get_effective_body(recompile_if_stale=False)
                self.navigation_binding = SimulationNavigationBinding.from_body(effective)
                if (
                    self.navigation_binding.body_snapshot_hash != config["body_snapshot_hash"]
                    or self.navigation_binding.run_id
                    != Path("/evidence/run_id.txt").read_text().strip()
                    or effective.provider_interfaces["ros_capability_bindings"][
                        "safety.simulation_observation"
                    ]["urdf_sha256"]
                    != hashlib.sha256(Path("/evidence/robot.urdf").read_bytes()).hexdigest()
                    or self.navigation_binding.frame_id
                    != (self.sources.map_frame if self.sources else "map")
                ):
                    raise ValueError("navigation observer Body/model/run/frame mismatch")
        if self.sources:
            if self.sources.run_id != Path("/evidence/run_id.txt").read_text().strip():
                raise ValueError("observer input run differs from current World")
            if self.navigation_binding:
                if (
                    self.navigation_binding.run_id != self.sources.run_id
                    or self.navigation_binding.observation_topic
                    != self.sources.topic("observation")
                    or self.navigation_binding.lease_service != self.sources.topic("lease")
                ):
                    raise ValueError(
                        "immutable navigation Body differs from pinned observer routes"
                    )
            elif not self.source_only:
                raise ValueError(
                    "generic bootstrap is source-only until an immutable Body is bound"
                )
        self.source_received_control_messages = 0
        self.previous_measured_pose = None
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
        self.contact_topics = (
            list(self.sources.contact_topics)
            if self.sources
            else json.loads(Path("/evidence/contact_topics.json").read_text())
        )
        self.wheel_topics = (
            list(self.sources.ground_witness_topics)
            if self.sources
            else [t for t in self.contact_topics if "/wheel_" in t]
        )
        if not self.sources and len(self.wheel_topics) != 2:
            raise ValueError("fixture requires two independently observed wheel contact streams")
        self.contact_trace = (
            Path(f"/evidence/contact-messages-{os.getpid()}.jsonl").open("x", buffering=1)  # noqa: SIM115 - node lifecycle closes evidence stream
            if self.sources
            else None
        )
        self.contact_trace_bytes = 0
        self.contact_trace_rows = 0
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
        self.publisher = self.create_publisher(
            String,
            self.navigation_binding.observation_topic
            if self.navigation_binding
            else self.source_topic("observation", "/rosclaw_sim/observation"),
            10,
        )
        self.cleaning_state = (
            None
            if self.navigation_binding or self.sources
            else self.create_publisher(Bool, "/rosclaw_sim/cleaning_state", 10)
        )
        self.controller_watchdog = (
            self.sources.drive_twist_stamped
            if self.sources
            else self.declare_parameter("controller_watchdog", True).value
        )
        self.velocity = (
            None
            if self.source_only
            else self.create_publisher(
                TwistStamped if self.controller_watchdog else Twist,
                self.source_topic(
                    "drive_command",
                    "/drive_controller/cmd_vel" if self.controller_watchdog else "/cmd_vel",
                ),
                10,
            )
        )
        self.control_callbacks = MutuallyExclusiveCallbackGroup()
        self.pose_callbacks = MutuallyExclusiveCallbackGroup()
        self.contact_callbacks = MutuallyExclusiveCallbackGroup()
        self.wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_subscription(
            TFMessage,
            self.source_topic("ground_truth", "/rosclaw_sim/ground_truth"),
            self.observe,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
            callback_group=self.pose_callbacks,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            self.source_topic("localization", "/amcl_pose"),
            self.localized,
            10,
        )
        self.create_subscription(
            Twist, self.source_topic("velocity_input", "/nav_cmd_vel"), self.command, 10
        )
        self.create_subscription(
            CollisionMonitorState,
            self.source_topic("collision_monitor_state", "/collision_monitor_state"),
            self.collision_monitor_observation,
            10,
        )
        if not self.source_only and not self.sources:
            for topic in ["/cmd_vel_smoothed", "/nav_cmd_vel"]:
                self.create_subscription(
                    Twist,
                    topic,
                    lambda message, t=topic: self.velocity_observation(t, message),
                    10,
                )
        self.create_subscription(Bool, "/is_rotating_to_heading", self.rotation_observation, 10)
        if not self.sources:
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
        if not self.sources:
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
            if not self.navigation_binding:
                self.create_service(
                    SetBool,
                    "/rosclaw_sim/cleaning",
                    self.set_cleaning,
                    callback_group=self.control_callbacks,
                )
        self.create_service(
            SetBool,
            self.navigation_binding.lease_service
            if self.navigation_binding
            else self.source_topic("lease", "/rosclaw_sim/lease"),
            self.heartbeat,
            callback_group=self.control_callbacks,
        )
        self.create_timer(
            0.05, self.tick, callback_group=self.control_callbacks, clock=self.wall_clock
        )

    def source_topic(self, role, legacy):
        return self.sources.topic(role) if self.sources else legacy

    def map_observation(self, message):
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
            "source": self.source_topic("localization", "amcl_pose"),
            "captured_at": datetime.now(UTC).isoformat(),
            "time_sec": message.header.stamp.sec + message.header.stamp.nanosec / 1e9,
            "covariance": list(message.pose.covariance),
        }

    def contacts(self, topic, message):
        if self.contact_trace is not None:
            raw = (
                json.dumps(
                    {
                        "schema_version": "rosclaw.original_sim_contact_message.v1",
                        "topic": topic,
                        "run_id": self.sources.run_id,
                        "captured_at": datetime.now(UTC).isoformat(),
                        "received_monotonic_ns": time.monotonic_ns(),
                        "message": message_to_ordereddict(message),
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            self.contact_trace_bytes += len(raw.encode("utf-8"))
            self.contact_trace_rows += 1
            if (
                self.contact_trace_bytes > 30_000_000
                or self.contact_trace_rows > 20_000
                or len(raw) > 500_000
            ):
                raise RuntimeError("bounded original contact capture exhausted")
            self.contact_trace.write(raw)
        touching = any(
            (
                c.collision1.name not in self.sources.ground_collision_names
                and c.collision2.name not in self.sources.ground_collision_names
            )
            if self.sources
            else ("floor::" not in c.collision1.name and "floor::" not in c.collision2.name)
            for c in message.contacts
        )
        with self.observation_lock:
            self.contact_seen[topic] = time.monotonic()
            if touching and not any(self.contact_active.values()):
                self.physics_collision_count += 1
            self.contact_active[topic] = touching

    def heartbeat(self, request, response):
        if self.source_only:
            response.success = False
            response.message = "source-only observation never enables a lease"
            return response
        self.lease = time.monotonic() + 1.5 if request.data else 0
        self.lease_updates += 1
        response.success = True
        return response

    def command(self, message):
        if self.source_only:
            self.source_received_control_messages += 1
            return
        if time.monotonic() < self.lease:
            self.publish_velocity(message)

    def publish_velocity(self, message):
        if self.source_only:
            return
        if self.controller_watchdog:
            stamped = TwistStamped()
            stamped.header.stamp = self.get_clock().now().to_msg()
            stamped.twist = message
            self.velocity.publish(stamped)
        else:
            self.velocity.publish(message)

    def observe(self, message):
        for t in message.transforms:
            if t.child_frame_id == (
                self.sources.robot_entity if self.sources else self.profile.simulation_model
            ):
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
        if now > self.lease:
            if self.cleaning:
                self.get_logger().warning(
                    f"Cleaning disabled by expired daemon lease: overdue={now - self.lease:.3f}s"
                )
            self.publish_velocity(Twist())
            self.cleaning = False
        if self.cleaning_state is not None:
            self.cleaning_state.publish(Bool(data=self.cleaning))
        if pose is None:
            return
        if self.published_time == pose["time_sec"]:
            return
        self.published_time = pose["time_sec"]
        # Room geometry comes from the checked-in Gazebo world. A conservative
        # disc encloses this configured robot's physical footprint. This is
        # independent post-physics geometric collision observation, not Nav2's
        # prediction or a claim from the caller. Ground contact is excluded.
        touching = (
            False
            if self.sources
            else max(abs(pose["x"]), abs(pose["y"])) + self.profile.physical_radius_m >= 1.5
        )
        if touching and not self.in_collision:
            self.collision_count += 1
        self.in_collision = touching
        sample = {
            **pose,
            "source_only_observer": self.source_only,
            "source_received_control_messages": self.source_received_control_messages,
            "source_published_velocity_messages": 0 if self.source_only else None,
            "captured_at": datetime.now(UTC).isoformat(),
            "cleaning_enabled": self.cleaning,
            "lease_remaining_sec": self.lease - now,
            "lease_updates": self.lease_updates,
            "localization": self.localization,
            "evidence_domain": "GAZEBO_PHYSICS",
            "collision_count": max(self.collision_count, physics_collision_count),
            "geometry_collision_count": None if self.sources else self.collision_count,
            "physics_collision_count": physics_collision_count,
            "collision_source": "gazebo_contacts"
            if self.sources
            else "gazebo_contacts_and_ground_truth_geometry",
            # Non-contacting sensors publish only on contact. Wheel/ground
            # contacts continuously witness the live physics contact pipeline.
            "ground_truth_age_ms": (now - last_pose) * 1000,
            "observation_complete": 0 <= now - last_pose < 0.3
            and all(0 <= now - contact_seen.get(t, 0) < 1 for t in self.wheel_topics),
            "contact_stream_ages_ms": {t: (now - seen) * 1000 for t, seen in contact_seen.items()},
        }
        if self.navigation_binding:
            previous = self.previous_measured_pose
            self.previous_measured_pose = pose
            # Velocity comes from successive Gazebo poses, never Nav2 commands.
            if previous is None or pose["time_sec"] <= previous["time_sec"]:
                return
            dt = pose["time_sec"] - previous["time_sec"]
            dyaw = math.atan2(
                math.sin(pose["yaw"] - previous["yaw"]),
                math.cos(pose["yaw"] - previous["yaw"]),
            )
            sample.pop("cleaning_enabled")
            sample.update(
                body_id=self.navigation_binding.body_id,
                body_snapshot_hash=self.navigation_binding.body_snapshot_hash,
                run_id=self.navigation_binding.run_id,
                frame_id=self.navigation_binding.frame_id,
                lease_enabled=now < self.lease,
                linear_speed_mps=math.hypot(pose["x"] - previous["x"], pose["y"] - previous["y"])
                / dt,
                angular_speed_rps=dyaw / dt,
            )
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
            if rclpy.ok():
                node.publish_velocity(Twist())
        finally:
            # SIGINT may have invalidated the ROS context. Diagnostic flushing
            # must not depend on a last ROS publish succeeding at shutdown.
            node.trace.close()
            if node.contact_trace is not None:
                node.contact_trace.close()
            summary = node.plan_audit.close()
            Path(summary["path"] + ".summary.json").write_text(json.dumps(summary) + "\n")
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
