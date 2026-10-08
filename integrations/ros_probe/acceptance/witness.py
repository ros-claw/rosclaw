"""Gazebo fixture observer and simulated cleaning/deadman controller.

Runs only in the isolated simulation container. Ground truth is received from
Gazebo, independently of Nav2 localization and action results.
"""

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

import rclpy
from geometry_msgs.msg import PolygonStamped, PoseWithCovarianceStamped, Twist, TwistStamped
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


class Witness(Node):
    def __init__(self):
        super().__init__("rosclaw_sim_witness")
        saved_profile = json.loads(Path("/evidence/fixture_profile.json").read_text())
        self.profile = PROFILES[saved_profile["name"]]
        if saved_profile != self.profile.to_dict():
            raise ValueError("fixture observer profile differs from supported geometry")
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
        self.cleaning_state = self.create_publisher(Bool, "/rosclaw_sim/cleaning_state", 10)
        self.controller_watchdog = self.declare_parameter("controller_watchdog", True).value
        self.velocity = self.create_publisher(
            TwistStamped if self.controller_watchdog else Twist,
            "/drive_controller/cmd_vel" if self.controller_watchdog else "/cmd_vel",
            10,
        )
        self.control_callbacks = MutuallyExclusiveCallbackGroup()
        self.pose_callbacks = MutuallyExclusiveCallbackGroup()
        self.contact_callbacks = MutuallyExclusiveCallbackGroup()
        self.wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_subscription(
            TFMessage,
            "/rosclaw_sim/ground_truth",
            self.observe,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
            callback_group=self.pose_callbacks,
        )
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self.localized, 10)
        self.create_subscription(Twist, "/nav_cmd_vel", self.command, 10)
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
        self.lease = time.monotonic() + 1.5 if request.data else 0
        self.lease_updates += 1
        response.success = True
        return response

    def command(self, message):
        if time.monotonic() < self.lease:
            self.publish_velocity(message)

    def publish_velocity(self, message):
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
        if now > self.lease:
            if self.cleaning:
                self.get_logger().warning(
                    f"Cleaning disabled by expired daemon lease: overdue={now - self.lease:.3f}s"
                )
            self.publish_velocity(Twist())
            self.cleaning = False
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
            "observation_complete": 0 <= now - last_pose < 0.3
            and all(0 <= now - contact_seen.get(t, 0) < 1 for t in self.wheel_topics),
            "contact_stream_ages_ms": {t: (now - seen) * 1000 for t, seen in contact_seen.items()},
        }
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
            summary = node.plan_audit.close()
            Path(summary["path"] + ".summary.json").write_text(json.dumps(summary) + "\n")
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
