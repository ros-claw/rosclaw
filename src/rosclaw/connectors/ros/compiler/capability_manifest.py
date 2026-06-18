"""ROS Connector - Capability Manifest Compiler.

Compiles a discovered ROS graph into a ROSClaw Capability Manifest that
exposes safe, high-level capabilities instead of raw ROS primitives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from rosclaw.connectors.ros.discovery.graph import (
    RosActionInfo,
    RosGraphSnapshot,
    RosServiceInfo,
    RosTopicInfo,
)


@dataclass
class RosInterface:
    """Concrete ROS interface backing a capability."""

    ros_kind: str  # topic, service, action
    name: str
    msg_type: str = ""


@dataclass
class RosCapabilityRisk:
    """Risk metadata for a ROS capability."""

    level: str = "low"
    read_only: bool = True
    destructive: bool = False
    requires_sandbox: bool = False
    requires_runtime_guard: bool = False
    requires_stop_guard: bool = False
    max_duration_sec: float | None = None
    max_rate_hz: float | None = None


@dataclass
class RosCapability:
    """A single compiled capability."""

    id: str
    kind: str  # observation, actuation, state, parameter
    interface: RosInterface
    schema: dict[str, Any] = field(default_factory=dict)
    risk: RosCapabilityRisk = field(default_factory=RosCapabilityRisk)
    safety: dict[str, Any] = field(default_factory=dict)
    practice: dict[str, Any] = field(default_factory=dict)
    preferred: bool = True
    enabled: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "interface": {
                "ros_kind": self.interface.ros_kind,
                "name": self.interface.name,
                "type": self.interface.msg_type,
            },
            "schema": self.schema,
            "risk": {
                "level": self.risk.level,
                "read_only": self.risk.read_only,
                "destructive": self.risk.destructive,
                "requires_sandbox": self.risk.requires_sandbox,
                "requires_runtime_guard": self.risk.requires_runtime_guard,
                "requires_stop_guard": self.risk.requires_stop_guard,
                "max_duration_sec": self.risk.max_duration_sec,
                "max_rate_hz": self.risk.max_rate_hz,
            },
            "safety": self.safety,
            "practice": self.practice,
            "preferred": self.preferred,
            "enabled": self.enabled,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RosCapability":
        iface_data = data.get("interface", {})
        risk_data = data.get("risk", {})
        return cls(
            id=data.get("id", ""),
            kind=data.get("kind", ""),
            interface=RosInterface(
                ros_kind=iface_data.get("ros_kind", ""),
                name=iface_data.get("name", ""),
                msg_type=iface_data.get("type", ""),
            ),
            schema=data.get("schema", {}),
            risk=RosCapabilityRisk(
                level=risk_data.get("level", "low"),
                read_only=risk_data.get("read_only", True),
                destructive=risk_data.get("destructive", False),
                requires_sandbox=risk_data.get("requires_sandbox", False),
                requires_runtime_guard=risk_data.get("requires_runtime_guard", False),
                requires_stop_guard=risk_data.get("requires_stop_guard", False),
                max_duration_sec=risk_data.get("max_duration_sec"),
                max_rate_hz=risk_data.get("max_rate_hz"),
            ),
            safety=data.get("safety", {}),
            practice=data.get("practice", {}),
            preferred=data.get("preferred", True),
            enabled=data.get("enabled", True),
            reason=data.get("reason", ""),
        )


@dataclass
class CapabilityManifest:
    """Compiled capability manifest for a ROS system."""

    schema_version: str = "rosclaw.capability_manifest.v1"
    source: str = "ros_graph_discovery"
    robot_id: str = "unknown"
    endpoint: dict[str, Any] = field(default_factory=dict)
    ros: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    capabilities: list[RosCapability] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "robot_id": self.robot_id,
            "endpoint": self.endpoint,
            "ros": self.ros,
            "generated_at": self.generated_at,
            "capabilities": [c.to_dict() for c in self.capabilities],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapabilityManifest":
        manifest = cls(
            schema_version=data.get("schema_version", "rosclaw.capability_manifest.v1"),
            source=data.get("source", "ros_graph_discovery"),
            robot_id=data.get("robot_id", "unknown"),
            endpoint=data.get("endpoint", {}),
            ros=data.get("ros", {}),
            generated_at=data.get("generated_at", datetime.now(timezone.utc).isoformat()),
        )
        for cap_data in data.get("capabilities", []):
            manifest.capabilities.append(RosCapability.from_dict(cap_data))
        return manifest

    def get_capability(self, capability_id: str) -> RosCapability | None:
        for cap in self.capabilities:
            if cap.id == capability_id:
                return cap
        return None


class CapabilityManifestCompiler:
    """Compile a ROS graph snapshot into a CapabilityManifest."""

    def __init__(self, robot_id: str = "unknown", robot_spec: dict[str, Any] | None = None):
        self.robot_id = robot_id
        self.robot_spec = robot_spec or {}

    def compile(self, snapshot: RosGraphSnapshot) -> CapabilityManifest:
        """Compile capabilities from a graph snapshot."""
        manifest = CapabilityManifest(
            robot_id=self.robot_id,
            endpoint={"transport": "rosbridge", "host": "127.0.0.1", "port": 9090},
            ros={
                "version": snapshot.ros_version,
                "distro": snapshot.distro,
                "endpoint": snapshot.endpoint,
            },
        )

        preferred_interfaces = self._preferred_interfaces_from_spec()

        # 1. Services first (high-level, preferred when declared).
        service_caps = [self._compile_service(s) for s in snapshot.services]
        service_by_name = {c.interface.name: c for c in service_caps}
        for cap in service_caps:
            if cap.interface.name in preferred_interfaces:
                cap.preferred = True
                cap.reason = "Preferred high-level interface from robot spec."
            manifest.capabilities.append(cap)

        # 2. Actions next.
        for action in snapshot.actions:
            manifest.capabilities.append(self._compile_action(action))

        # 3. Topics last: observation topics first, guarded command topics last.
        for topic in snapshot.topics:
            if topic.is_command:
                # Skip if a preferred service covers the same semantic space.
                if self._has_preferred_service_for_command(topic, service_by_name):
                    continue
                manifest.capabilities.append(self._compile_command_topic(topic))
            elif topic.is_sensor:
                manifest.capabilities.append(self._compile_sensor_topic(topic))
            else:
                manifest.capabilities.append(self._compile_state_topic(topic))

        # Apply robot spec overrides (disable discouraged, adjust safety defaults).
        self._apply_spec_overrides(manifest)
        self._deduplicate_capability_ids(manifest)
        return manifest

    # ------------------------------------------------------------------
    # Compilers per ROS kind
    # ------------------------------------------------------------------
    def _compile_service(self, service: RosServiceInfo) -> RosCapability:
        capability_id = self._make_capability_id("service", service.name)
        risk = RosCapabilityRisk(
            level=service.risk_hint or "low",
            read_only=False,
            destructive=service.risk_hint == "high",
            requires_sandbox=service.risk_hint in {"medium", "high"},
            requires_runtime_guard=service.risk_hint in {"medium", "high"},
        )
        return RosCapability(
            id=capability_id,
            kind="actuation" if service.risk_hint in {"medium", "high"} else "state",
            interface=RosInterface(
                ros_kind="service",
                name=service.name,
                msg_type=service.srv_type,
            ),
            schema={"request": service.request_schema, "response": service.response_schema},
            risk=risk,
            practice={
                "capture_artifact": True,
                "artifact_type": "service_call",
                "capture_policy": "always",
            },
        )

    def _compile_action(self, action: RosActionInfo) -> RosCapability:
        return RosCapability(
            id=self._make_capability_id("action", action.name),
            kind="actuation",
            interface=RosInterface(
                ros_kind="action",
                name=action.name,
                msg_type=action.action_type,
            ),
            schema={
                "goal": action.goal_schema,
                "feedback": action.feedback_schema,
                "result": action.result_schema,
            },
            risk=RosCapabilityRisk(
                level=action.risk_hint or "medium",
                read_only=False,
                destructive=False,
                requires_sandbox=True,
                requires_runtime_guard=True,
            ),
            practice={
                "capture_artifact": True,
                "artifact_type": "action_trace",
                "capture_policy": "always",
            },
        )

    def _compile_sensor_topic(self, topic: RosTopicInfo) -> RosCapability:
        artifact_type = "image" if "image" in topic.msg_type.lower() else "message"
        return RosCapability(
            id=self._make_capability_id("observe", topic.name),
            kind="observation",
            interface=RosInterface(
                ros_kind="topic",
                name=topic.name,
                msg_type=topic.msg_type,
            ),
            schema={"msg_type": topic.msg_type},
            risk=RosCapabilityRisk(
                level="low",
                read_only=True,
                destructive=False,
                requires_sandbox=False,
            ),
            practice={
                "capture_artifact": True,
                "artifact_type": artifact_type,
                "capture_policy": "on_demand",
            },
        )

    def _compile_state_topic(self, topic: RosTopicInfo) -> RosCapability:
        return RosCapability(
            id=self._make_capability_id("state", topic.name),
            kind="state",
            interface=RosInterface(
                ros_kind="topic",
                name=topic.name,
                msg_type=topic.msg_type,
            ),
            schema={"msg_type": topic.msg_type},
            risk=RosCapabilityRisk(
                level="low",
                read_only=True,
                destructive=False,
                requires_sandbox=False,
            ),
        )

    def _compile_command_topic(self, topic: RosTopicInfo) -> RosCapability:
        return RosCapability(
            id=self._make_capability_id("command", topic.name),
            kind="actuation",
            interface=RosInterface(
                ros_kind="topic",
                name=topic.name,
                msg_type=topic.msg_type,
            ),
            schema={"msg_type": topic.msg_type},
            risk=RosCapabilityRisk(
                level="high",
                read_only=False,
                destructive=True,
                requires_sandbox=True,
                requires_runtime_guard=True,
                requires_stop_guard=True,
                max_duration_sec=1.0,
                max_rate_hz=10.0,
            ),
            safety={
                "constraints": {
                    "linear.x": [-0.2, 0.2],
                    "linear.y": [-0.1, 0.1],
                    "angular.z": [-0.5, 0.5],
                },
            },
            practice={
                "capture_artifact": True,
                "artifact_type": "command_trace",
                "capture_policy": "always",
            },
            preferred=False,
            reason="Low-level command topic; prefer high-level service/action when available.",
        )

    # ------------------------------------------------------------------
    # Robot spec integration
    # ------------------------------------------------------------------
    def _preferred_interfaces_from_spec(self) -> set[str]:
        preferred = self.robot_spec.get("preferred_interfaces", [])
        return {item.get("ros_name") for item in preferred if item.get("ros_name")}

    def _has_preferred_service_for_command(
        self,
        topic: RosTopicInfo,
        service_by_name: dict[str, RosCapability],
    ) -> bool:
        # Skip a low-level command topic only when the robot spec explicitly
        # prefers a high-level service that covers the same semantic space.
        # Discouraging a topic without a preferred alternative keeps the
        # capability available (it is already marked preferred=False).
        return bool(self._preferred_interfaces_from_spec() & set(service_by_name.keys()))

    def _apply_spec_overrides(self, manifest: CapabilityManifest) -> None:
        safety_defaults = self.robot_spec.get("safety_defaults", {})
        max_linear = safety_defaults.get("max_linear_velocity", 0.2)
        max_angular = safety_defaults.get("max_angular_velocity", 0.5)
        max_duration = safety_defaults.get("max_motion_duration_sec", 1.0)

        for cap in manifest.capabilities:
            if cap.interface.ros_kind != "topic" or cap.risk.level != "high":
                continue
            cap.safety.setdefault("constraints", {})
            cap.safety["constraints"]["linear.x"] = [-max_linear, max_linear]
            cap.safety["constraints"]["linear.y"] = [-max_linear * 0.5, max_linear * 0.5]
            cap.safety["constraints"]["angular.z"] = [-max_angular, max_angular]
            if cap.risk.max_duration_sec is None or cap.risk.max_duration_sec > max_duration:
                cap.risk.max_duration_sec = max_duration

    def _deduplicate_capability_ids(self, manifest: CapabilityManifest) -> None:
        """Append a namespace suffix to duplicate capability IDs.

        When a ROS graph contains multiple interfaces that collapse to the same
        generic capability ID (for example ``/turtle1/cmd_vel`` and
        ``/turtle2/cmd_vel`` both become ``robot.base.velocity_command``),
        disambiguate them using the interface namespace so every capability
        remains addressable.
        """
        from collections import Counter

        counts = Counter(cap.id for cap in manifest.capabilities)
        duplicates = {cap_id for cap_id, count in counts.items() if count > 1}
        if not duplicates:
            return

        seen: dict[str, int] = {}
        for cap in manifest.capabilities:
            if cap.id not in duplicates:
                continue
            iface_parts = [p for p in cap.interface.name.strip("/").split("/") if p]
            namespace = "_".join(iface_parts[:-1]) if len(iface_parts) > 1 else "alt"
            suffix = self.sanitize_id(namespace) or "alt"
            new_id = f"{cap.id}.{suffix}"
            if new_id in seen:
                seen[new_id] += 1
                new_id = f"{new_id}_{seen[new_id]}"
            else:
                seen[new_id] = 1
            cap.id = new_id

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------
    def _make_capability_id(self, kind: str, ros_name: str) -> str:
        """Generate a stable, human-readable capability id.

        Examples:
            /go2/move        -> robot.go2.move
            /turtle1/cmd_vel -> robot.base.velocity_command
            /camera/image_raw-> robot.observe.camera.rgb
        """
        parts = [p for p in ros_name.strip("/").split("/") if p]
        if not parts:
            parts = ["unknown"]

        # Special-case common topic patterns (only for topics).
        if kind in ("observe", "state", "command"):
            if "cmd_vel" in ros_name:
                return f"{self.robot_id}.base.velocity_command"
            if any(p in ros_name.lower() for p in ["image", "camera", "rgb"]):
                return f"{self.robot_id}.observe.camera.rgb"
            if "joint_states" in ros_name:
                return f"{self.robot_id}.observe.joint_states"
            if "pose" in ros_name.lower():
                return f"{self.robot_id}.observe.pose"
            if "odom" in ros_name.lower():
                return f"{self.robot_id}.observe.odom"

        if kind == "service":
            return f"{self.robot_id}.{".".join(parts)}"
        if kind == "action":
            return f"{self.robot_id}.action.{".".join(parts)}"
        if kind == "observe":
            return f"{self.robot_id}.observe.{'_'.join(parts[-2:])}"
        if kind == "state":
            return f"{self.robot_id}.state.{'_'.join(parts[-2:])}"
        return f"{self.robot_id}.{kind}.{'_'.join(parts[-2:])}"

    @staticmethod
    def sanitize_id(raw: str) -> str:
        """Sanitize a string for use in a capability id."""
        return re.sub(r"[^a-zA-Z0-9_.-]", "_", raw).strip("._")
