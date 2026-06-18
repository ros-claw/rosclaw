"""ROS Connector - ROS1/ROS2 message type normalizer.

Converts raw type strings from rosapi into a canonical internal form.
"""

from __future__ import annotations

from rosclaw.connectors.ros.discovery.rosapi_resolver import RosVersion


def normalize_msg_type(raw: str, ros_version: RosVersion) -> str:
    """Normalize a ROS message type to ``pkg/msg/Type`` form."""
    if not raw:
        return raw
    raw = raw.strip()

    # Already canonical ROS2 form.
    if "/msg/" in raw:
        return raw

    # ROS1 short form: geometry_msgs/Twist
    if "/" in raw and ros_version == RosVersion.ROS1:
        pkg, name = raw.split("/", 1)
        return f"{pkg}/msg/{name}"

    # Some rosapi responses include ROS2 form without explicit /msg/.
    if "/" in raw:
        parts = raw.split("/")
        if len(parts) == 2:
            return f"{parts[0]}/msg/{parts[1]}"
        return raw

    return raw


def normalize_srv_type(raw: str, ros_version: RosVersion) -> str:
    """Normalize a ROS service type to ``pkg/srv/Type`` form."""
    if not raw:
        return raw
    raw = raw.strip()

    if "/srv/" in raw:
        return raw

    if "/" in raw and ros_version == RosVersion.ROS1:
        pkg, name = raw.split("/", 1)
        return f"{pkg}/srv/{name}"

    if "/" in raw:
        parts = raw.split("/")
        if len(parts) == 2:
            return f"{parts[0]}/srv/{parts[1]}"
        return raw

    return raw


def normalize_action_type(raw: str, ros_version: RosVersion) -> str:
    """Normalize a ROS action type to ``pkg/action/Type`` form."""
    if not raw:
        return raw
    raw = raw.strip()

    if "/action/" in raw:
        return raw

    if "/" in raw and ros_version == RosVersion.ROS1:
        pkg, name = raw.split("/", 1)
        return f"{pkg}/action/{name}"

    if "/" in raw:
        parts = raw.split("/")
        if len(parts) == 2:
            return f"{parts[0]}/action/{parts[1]}"
        return raw

    return raw


def denormalize_msg_type(canonical: str, ros_version: RosVersion) -> str:
    """Convert canonical type back to the ROS version's native string."""
    if ros_version == RosVersion.ROS1:
        return canonical.replace("/msg/", "/")
    return canonical


def denormalize_srv_type(canonical: str, ros_version: RosVersion) -> str:
    if ros_version == RosVersion.ROS1:
        return canonical.replace("/srv/", "/")
    return canonical
