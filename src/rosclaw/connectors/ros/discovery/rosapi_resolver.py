"""ROS Connector - rosapi resolver.

Detects ROS version, distro, and rosapi service prefix without importing
ROS Python client libraries.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger("rosclaw.connectors.ros.discovery.rosapi_resolver")


class RosVersion(StrEnum):
    ROS1 = "ros1"
    ROS2 = "ros2"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RosApiProfile:
    """Resolved rosapi interface profile."""

    version: RosVersion
    distro: str
    service_prefix: str
    type_prefix: str

    def service(self, short_name: str) -> str:
        return f"{self.service_prefix}/{short_name}"

    def service_type(self, short_name: str) -> str:
        return f"{self.type_prefix}/{short_name}"


class RosApiDetectionError(Exception):
    """Raised when the rosapi profile cannot be determined."""


class RosApiResolver:
    """Detect ROS version and rosapi service layout via rosbridge."""

    CANDIDATE_PREFIXES = ["/rosapi", "/rosapi_node"]

    def __init__(self, transport):
        self._transport = transport
        self._profile: RosApiProfile | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def resolve(self, force: bool = False) -> RosApiProfile:
        """Resolve (or return cached) rosapi profile."""
        if self._profile is not None and not force:
            return self._profile

        profile = self._detect()
        self._profile = profile
        return profile

    def reset_for_test(self) -> None:
        """Clear cached profile (used in tests)."""
        self._profile = None

    # ------------------------------------------------------------------
    # Detection logic
    # ------------------------------------------------------------------
    def _detect(self) -> RosApiProfile:
        not_found_count = 0
        last_error: str | None = None

        # 1. Try get_ros_version on candidate prefixes.
        for prefix in self.CANDIDATE_PREFIXES:
            profile, status = self._call_get_ros_version(prefix)
            if profile is not None:
                return profile
            if status == "not_found":
                not_found_count += 1
            elif status:
                last_error = status

        # 2. get_ros_version service does not exist on any prefix -> likely ROS1.
        if not_found_count == len(self.CANDIDATE_PREFIXES):
            ros1_profile = self._detect_ros1()
            if ros1_profile is not None:
                return ros1_profile

        raise RosApiDetectionError(
            f"Could not detect ROS version; rosbridge may be unreachable or rosapi not running: {last_error or 'unknown'}"
        )

    def _call_get_ros_version(self, prefix: str) -> tuple[RosApiProfile | None, str | None]:
        """Call get_ros_version and classify the outcome.

        Returns:
            (profile, None) on success,
            (None, "not_found") if the service does not exist,
            (None, error_message) for network/transport errors.
        """
        result = self._transport.call_service(
            f"{prefix}/get_ros_version",
            {},
        )
        if not result.ok:
            logger.debug("get_ros_version failed for %s: %s", prefix, result.error)
            error = result.error or "unknown"
            if "not found" in error.lower() or "service" in error.lower():
                return None, "not_found"
            return None, error

        values = _extract_values(result.data)
        version_number = values.get("version")
        distro = values.get("distro", "")

        if not isinstance(version_number, int):
            return None, "not_found"

        if version_number >= 2:
            return RosApiProfile(
                version=RosVersion.ROS2,
                distro=distro or "unknown",
                service_prefix=prefix,
                type_prefix="rosapi_msgs/srv",
            ), None
        # ROS1 explicitly reports version 1.
        return RosApiProfile(
            version=RosVersion.ROS1,
            distro=distro or self._get_ros1_distro(prefix),
            service_prefix=prefix,
            type_prefix="rosapi",
        ), None

    def _detect_ros1(self) -> RosApiProfile | None:
        prefix = "/rosapi"
        distro = self._get_ros1_distro(prefix)
        return RosApiProfile(
            version=RosVersion.ROS1,
            distro=distro or "unknown",
            service_prefix=prefix,
            type_prefix="rosapi",
        )

    def _get_ros1_distro(self, prefix: str) -> str:
        result = self._transport.call_service(
            f"{prefix}/get_param",
            {"name": "/rosdistro"},
        )
        if not result.ok:
            return ""
        values = _extract_values(result.data)
        value = values.get("value")
        if isinstance(value, str):
            # ROS1 rosapi may return the parameter as a JSON-encoded string.
            try:
                decoded = json.loads(value)
                if isinstance(decoded, str):
                    value = decoded
            except json.JSONDecodeError:
                pass
            return value.strip().strip("/").split("/")[-1]
        return ""


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _extract_values(response_data: dict[str, Any] | None) -> dict[str, Any]:
    """Extract the ``values`` dict from a rosbridge service response."""
    if not isinstance(response_data, dict):
        return {}
    values = response_data.get("values")
    if isinstance(values, dict):
        return values
    return {}


def make_ros2_profile(prefix: str, distro: str) -> RosApiProfile:
    """Factory for tests."""
    return RosApiProfile(
        version=RosVersion.ROS2,
        distro=distro,
        service_prefix=prefix,
        type_prefix="rosapi_msgs/srv",
    )


def make_ros1_profile(prefix: str, distro: str) -> RosApiProfile:
    """Factory for tests."""
    return RosApiProfile(
        version=RosVersion.ROS1,
        distro=distro,
        service_prefix=prefix,
        type_prefix="rosapi",
    )
