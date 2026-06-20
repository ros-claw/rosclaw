"""ROS Connector - ROS graph discovery.

Discovers topics, services, actions, nodes, and parameters via rosapi
without importing ROS Python client libraries.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from rosclaw.connectors.ros.discovery.normalizer import (
    normalize_msg_type,
    normalize_srv_type,
)
from rosclaw.connectors.ros.discovery.rosapi_resolver import RosApiProfile, RosVersion
from rosclaw.connectors.ros.transport.base import RosTransportResult

logger = logging.getLogger("rosclaw.connectors.ros.discovery.graph")

COMMAND_TOPIC_PATTERNS = [
    "/cmd_vel",
    "/cmd_vel_*",
    "*/cmd_vel",
    "*/joint_commands",
    "*/joint_trajectory",
    "*/effort_controller/*",
    "*/velocity_controller/*",
]

SENSOR_TOPIC_PATTERNS = [
    "*/image_raw",
    "*/camera/*",
    "*/pointcloud",
    "*/cloud",
    "*/scan",
    "*/joint_states",
    "*/odom",
    "*/tf",
]


@dataclass
class RosTopicInfo:
    name: str
    msg_type: str
    publishers: list[str] = field(default_factory=list)
    subscribers: list[str] = field(default_factory=list)
    hz_estimate: float | None = None
    is_sensor: bool = False
    is_command: bool = False
    risk_hint: str | None = None


@dataclass
class RosServiceInfo:
    name: str
    srv_type: str
    providers: list[str] = field(default_factory=list)
    request_schema: dict[str, Any] = field(default_factory=dict)
    response_schema: dict[str, Any] = field(default_factory=dict)
    risk_hint: str | None = None


@dataclass
class RosActionInfo:
    name: str
    action_type: str
    goal_schema: dict[str, Any] = field(default_factory=dict)
    feedback_schema: dict[str, Any] = field(default_factory=dict)
    result_schema: dict[str, Any] = field(default_factory=dict)
    risk_hint: str | None = None


@dataclass
class RosGraphSnapshot:
    ros_version: str
    distro: str
    endpoint: str
    topics: list[RosTopicInfo]
    services: list[RosServiceInfo]
    actions: list[RosActionInfo]
    nodes: list[dict[str, Any]]
    params: list[str]
    captured_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ros_version": self.ros_version,
            "distro": self.distro,
            "endpoint": self.endpoint,
            "topics": [
                {
                    "name": t.name,
                    "msg_type": t.msg_type,
                    "publishers": t.publishers,
                    "subscribers": t.subscribers,
                    "hz_estimate": t.hz_estimate,
                    "is_sensor": t.is_sensor,
                    "is_command": t.is_command,
                    "risk_hint": t.risk_hint,
                }
                for t in self.topics
            ],
            "services": [
                {
                    "name": s.name,
                    "srv_type": s.srv_type,
                    "providers": s.providers,
                    "request_schema": s.request_schema,
                    "response_schema": s.response_schema,
                    "risk_hint": s.risk_hint,
                }
                for s in self.services
            ],
            "actions": [
                {
                    "name": a.name,
                    "action_type": a.action_type,
                    "goal_schema": a.goal_schema,
                    "feedback_schema": a.feedback_schema,
                    "result_schema": a.result_schema,
                    "risk_hint": a.risk_hint,
                }
                for a in self.actions
            ],
            "nodes": self.nodes,
            "params": self.params,
            "captured_at": self.captured_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RosGraphSnapshot:
        topics = []
        for t in data.get("topics", []):
            topics.append(
                RosTopicInfo(
                    name=t.get("name", ""),
                    msg_type=t.get("msg_type", ""),
                    publishers=t.get("publishers", []),
                    subscribers=t.get("subscribers", []),
                    hz_estimate=t.get("hz_estimate"),
                    is_sensor=t.get("is_sensor", False),
                    is_command=t.get("is_command", False),
                    risk_hint=t.get("risk_hint"),
                )
            )
        services = []
        for s in data.get("services", []):
            services.append(
                RosServiceInfo(
                    name=s.get("name", ""),
                    srv_type=s.get("srv_type", ""),
                    providers=s.get("providers", []),
                    request_schema=s.get("request_schema", {}),
                    response_schema=s.get("response_schema", {}),
                    risk_hint=s.get("risk_hint"),
                )
            )
        actions = []
        for a in data.get("actions", []):
            actions.append(
                RosActionInfo(
                    name=a.get("name", ""),
                    action_type=a.get("action_type", ""),
                    goal_schema=a.get("goal_schema", {}),
                    feedback_schema=a.get("feedback_schema", {}),
                    result_schema=a.get("result_schema", {}),
                    risk_hint=a.get("risk_hint"),
                )
            )
        return cls(
            ros_version=data.get("ros_version", ""),
            distro=data.get("distro", ""),
            endpoint=data.get("endpoint", ""),
            topics=topics,
            services=services,
            actions=actions,
            nodes=data.get("nodes", []),
            params=data.get("params", []),
            captured_at=data.get("captured_at", ""),
        )


class RosGraphDiscovery:
    """Discover ROS graph entities via rosapi services over rosbridge."""

    def __init__(self, transport, profile: RosApiProfile | None = None):
        self._transport = transport
        self._profile = profile
        self._topic_types: dict[str, str] = {}
        self._service_types: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def discover(self) -> RosGraphSnapshot:
        """Capture a full snapshot of the ROS graph."""
        if self._profile is None:
            from rosclaw.connectors.ros.discovery import RosApiResolver
            self._profile = RosApiResolver(self._transport).resolve()

        endpoint = getattr(self._transport, "endpoint", None)
        endpoint_str = str(endpoint.url) if endpoint else "unknown"

        topics = self.list_topics()
        services = self.list_services()
        actions = self.list_actions()
        nodes = self.list_nodes()
        params = self.list_params()

        return RosGraphSnapshot(
            ros_version=self._profile.version.value,
            distro=self._profile.distro,
            endpoint=endpoint_str,
            topics=topics,
            services=services,
            actions=actions,
            nodes=nodes,
            params=params,
            captured_at=datetime.now(UTC).isoformat(),
        )

    def list_topics(self) -> list[RosTopicInfo]:
        result = self._call_rosapi("topics")
        if not result.ok:
            logger.warning("list_topics failed: %s", result.error)
            return []

        values = _extract_values(result.data)
        topic_names = values.get("topics") or []
        topic_types = values.get("types") or []

        # rosapi/topics may return topics and types as parallel arrays.
        if len(topic_names) == len(topic_types):
            self._topic_types = dict(zip(topic_names, topic_types, strict=True))
        else:
            self._topic_types = {}

        infos: list[RosTopicInfo] = []
        for name in topic_names:
            raw_type = self._topic_types.get(name, "")
            msg_type = normalize_msg_type(raw_type, self._profile.version)
            info = RosTopicInfo(
                name=name,
                msg_type=msg_type,
                publishers=[],
                subscribers=[],
            )
            self._classify_topic(info)
            infos.append(info)
        return infos

    def get_topic_type(self, topic: str) -> str:
        result = self._call_rosapi("topic_type", {"topic": topic})
        if result.ok:
            values = _extract_values(result.data)
            raw = values.get("type", "")
            return normalize_msg_type(raw, self._profile.version)
        return ""

    def get_topic_details(self, topic: str) -> RosTopicInfo | None:
        msg_type = self.get_topic_type(topic)
        if not msg_type:
            return None

        publishers = self._get_publishers(topic)
        subscribers = self._get_subscribers(topic)
        info = RosTopicInfo(
            name=topic,
            msg_type=msg_type,
            publishers=publishers,
            subscribers=subscribers,
        )
        self._classify_topic(info)
        return info

    def get_message_details(self, message_type: str) -> dict[str, Any]:
        """Return a structural sketch of a message type.

        rosapi/message_details returns a JSON string of the message definition
        graph. We parse it defensively and return a field-oriented dict.
        """
        native_type = self._denormalize_msg(message_type)
        result = self._call_rosapi("message_details", {"type": native_type})
        if not result.ok:
            return {"type": message_type, "error": result.error, "fields": {}}

        values = _extract_values(result.data)
        details = values.get("message") or values.get("details") or "{}"
        try:
            import json
            parsed = json.loads(details) if isinstance(details, str) else details
        except Exception as exc:
            return {"type": message_type, "error": str(exc), "fields": {}}

        if not isinstance(parsed, dict):
            return {"type": message_type, "fields": {}}

        return {"type": message_type, "fields": parsed.get("fields", parsed)}

    def list_services(self) -> list[RosServiceInfo]:
        result = self._call_rosapi("services")
        if not result.ok:
            logger.warning("list_services failed: %s", result.error)
            return []

        values = _extract_values(result.data)
        service_names = values.get("services") or []

        infos: list[RosServiceInfo] = []
        for name in service_names:
            srv_type = self.get_service_type(name)
            providers = self._get_service_providers(name)
            info = RosServiceInfo(
                name=name,
                srv_type=srv_type,
                providers=providers,
            )
            self._classify_service(info)
            infos.append(info)
        return infos

    def get_service_type(self, service: str) -> str:
        if service in self._service_types:
            return self._service_types[service]
        result = self._call_rosapi("service_type", {"service": service})
        if result.ok:
            values = _extract_values(result.data)
            raw = values.get("type", "")
            srv_type = normalize_srv_type(raw, self._profile.version)
            self._service_types[service] = srv_type
            return srv_type
        return ""

    def get_service_details(self, service: str) -> RosServiceInfo | None:
        srv_type = self.get_service_type(service)
        if not srv_type:
            return None
        providers = self._get_service_providers(service)
        return RosServiceInfo(
            name=service,
            srv_type=srv_type,
            providers=providers,
        )

    def list_actions(self) -> list[RosActionInfo]:
        # rosapi/action_servers is available in rosbridge for ROS2.
        result = self._call_rosapi("action_servers")
        if not result.ok:
            logger.debug("list_actions not supported or failed: %s", result.error)
            return []

        values = _extract_values(result.data)
        action_names = values.get("actions") or values.get("action_servers") or []

        infos: list[RosActionInfo] = []
        for name in action_names:
            action_type = self._get_action_type(name)
            info = RosActionInfo(name=name, action_type=action_type)
            self._classify_action(info)
            infos.append(info)
        return infos

    def get_action_type(self, action: str) -> str:
        return self._get_action_type(action)

    def get_action_details(self, action: str) -> RosActionInfo | None:
        action_type = self._get_action_type(action)
        if not action_type:
            return None
        return RosActionInfo(name=action, action_type=action_type)

    def list_nodes(self) -> list[dict[str, Any]]:
        result = self._call_rosapi("nodes")
        if not result.ok:
            logger.warning("list_nodes failed: %s", result.error)
            return []
        values = _extract_values(result.data)
        node_names = values.get("nodes") or []
        return [{"name": n} for n in node_names]

    def get_node_details(self, node: str) -> dict[str, Any]:
        result = self._call_rosapi("node_details", {"node": node})
        if not result.ok:
            return {"name": node, "error": result.error}
        values = _extract_values(result.data)
        return {
            "name": node,
            "subscriptions": values.get("subscriptions", []),
            "publications": values.get("publications", []),
            "services": values.get("services", []),
        }

    def list_params(self) -> list[str]:
        result = self._call_rosapi("params")
        if not result.ok:
            logger.debug("list_params not supported or failed: %s", result.error)
            return []
        values = _extract_values(result.data)
        return values.get("params") or []

    def get_param(self, name: str) -> Any:
        result = self._call_rosapi("get_param", {"name": name})
        if not result.ok:
            return None
        values = _extract_values(result.data)
        return values.get("value")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _call_rosapi(self, short_name: str, args: dict[str, Any] | None = None):
        service = self._profile.service(short_name)
        result = self._transport.call_service(service, args or {})
        if isinstance(result, RosTransportResult):
            return result
        # Allow transports that return dicts directly (legacy/mock compatibility).
        return RosTransportResult(ok=True, data=result)

    def _get_publishers(self, topic: str) -> list[str]:
        result = self._call_rosapi("publishers", {"topic": topic})
        if not result.ok:
            return []
        values = _extract_values(result.data)
        return values.get("publishers") or []

    def _get_subscribers(self, topic: str) -> list[str]:
        result = self._call_rosapi("subscribers", {"topic": topic})
        if not result.ok:
            return []
        values = _extract_values(result.data)
        return values.get("subscribers") or []

    def _get_service_providers(self, service: str) -> list[str]:
        result = self._call_rosapi("service_node", {"service": service})
        if not result.ok:
            return []
        values = _extract_values(result.data)
        node = values.get("node")
        return [node] if node else []

    def _get_action_type(self, action: str) -> str:
        # There is no standard rosapi action_type service; return empty.
        # Providers may override this by subscribing to the action topics.
        return ""

    def _denormalize_msg(self, canonical: str) -> str:
        if self._profile.version == RosVersion.ROS1:
            return canonical.replace("/msg/", "/")
        return canonical

    # ------------------------------------------------------------------
    # Risk classification
    # ------------------------------------------------------------------
    def _classify_topic(self, info: RosTopicInfo) -> None:
        name_lower = info.name.lower()
        for pattern in COMMAND_TOPIC_PATTERNS:
            if _match_pattern(name_lower, pattern.lower()):
                info.is_command = True
                info.risk_hint = "high"
                return
        for pattern in SENSOR_TOPIC_PATTERNS:
            if _match_pattern(name_lower, pattern.lower()):
                info.is_sensor = True
                info.risk_hint = "low"
                return

    def _classify_service(self, info: RosServiceInfo) -> None:
        name_lower = info.name.lower()
        motion_tokens = [
            "move", "execute", "command", "control", "grip", "stand", "sit", "recover",
            "balance", "walk", "navigate", "stop", "halt", "reset",
        ]
        if any(tok in name_lower for tok in motion_tokens):
            info.risk_hint = "medium"
        else:
            info.risk_hint = "low"

    def _classify_action(self, info: RosActionInfo) -> None:
        info.risk_hint = "medium"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _extract_values(response_data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(response_data, dict):
        return {}
    values = response_data.get("values")
    if isinstance(values, dict):
        return values
    return {}


def _match_pattern(name: str, pattern: str) -> bool:
    """Simple glob matcher supporting ``*`` wildcards."""
    regex = "^" + re.escape(pattern).replace(r"\*", ".*") + "$"
    return bool(re.match(regex, name))
