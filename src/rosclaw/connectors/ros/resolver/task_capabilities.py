"""Task dependency proposals and observed control prerequisites, never admission."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, model_validator

from ..intelligence.graph_contracts import GraphRecord
from .capability_resolver import resolve_capabilities


class ExistingRos2ControlV1(GraphRecord):
    command_topic: str = Field(min_length=2, max_length=512)
    consumer_node: str = Field(min_length=2, max_length=512)
    message_type: Literal["geometry_msgs/msg/Twist", "geometry_msgs/msg/TwistStamped"]
    driver_package: str = Field(pattern=r"^[a-z][a-z0-9_]{0,127}$")
    interface_kind: Literal["EXISTING_ROS2_TOPIC"] = "EXISTING_ROS2_TOPIC"
    declaration: Literal["OPERATOR_SELECTED_PREREQUISITE_NOT_AUTHORITY"] = (
        "OPERATOR_SELECTED_PREREQUISITE_NOT_AUTHORITY"
    )

    @model_validator(mode="after")
    def concrete_names(self):
        import re

        for name in (self.command_topic, self.consumer_node):
            if not re.fullmatch(r"/(?:[A-Za-z_][A-Za-z0-9_]*)(?:/[A-Za-z_][A-Za-z0-9_]*)*", name):
                raise ValueError("explicit absolute ROS names required; no conventional defaults")
        return self


NAVIGATION = (
    "base.ros2_control_interface",
    "state.odometry",
    "sensing.lidar",
    "mapping.occupancy_map",
    "localization.pose_estimate",
    "safety.collision_monitor",
    "navigation.navigate_to_pose",
)
CLEANING = (
    "coverage.compute_path",
    "coverage.execute",
    "cleaning.enable",
    "cleaning.disable",
    "coverage.verify",
)


def resolve_task_capability_graph(model, task_class, control=None, *, now=None):
    """Observed interface presence is a prerequisite, never verified driver physics.

    Missing endpoints return NEEDS_DRIVER as an unmet application prerequisite,
    not an assertion that no driver exists anywhere. Incomplete observation is
    UNKNOWN and requests read-only evidence. No topic/frame/model defaults exist.
    """
    if task_class not in {"navigate_to_pose", "complete_area_cleaning"}:
        raise ValueError("explicit supported application task class required")
    if model.compute_snapshot_hash() != model.snapshot_hash:
        raise ValueError("unchanged sealed system model required")
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("aware application review time required")
    declaration = ExistingRos2ControlV1.model_validate(control) if control is not None else None
    fresh = 0 <= model.age_ms(now) <= 500
    complete = model.completeness.get("graph") is True and model.completeness.get("qos") is True
    driver_status = "UNKNOWN"
    reason = "current complete ROS2 graph and endpoint observation required"
    pointers = []
    if fresh and complete and model.environment.get("ros_generation") == "ros2":
        if declaration is None:
            driver_status, reason = (
                "NEEDS_DRIVER",
                "explicit existing ROS2 control interface required",
            )
        else:
            topics = [
                (i, row)
                for i, row in enumerate(model.graph.get("topics", []))
                if row.get("name") == declaration.command_topic
            ]
            subscribers = [
                (i, row)
                for i, row in enumerate(model.qos.get("endpoints", []))
                if row.get("topic") == declaration.command_topic
                and row.get("node") == declaration.consumer_node
                and row.get("kind") == "subscriber"
            ]
            packages = model.observations.get("package_inventory")
            if len(topics) > 1 or len(subscribers) > 1:
                reason = "ambiguous selected control endpoint; read-only identity probe required"
            elif not topics or not subscribers:
                driver_status = "NEEDS_DRIVER"
                reason = "selected existing ROS2 control prerequisite not observed; request driver adaptation"
            elif topics[0][1].get("msg_type") != declaration.message_type:
                driver_status, reason = "INCOMPATIBLE", "observed selected command type differs"
            elif type(packages) is not list:
                reason = "actual installed driver package inventory missing"
            elif declaration.driver_package not in packages:
                driver_status, reason = (
                    "NEEDS_DRIVER",
                    "selected driver package not in original inventory",
                )
            else:
                driver_status = "OBSERVED_ROS2_CONTROL_INTERFACE"
                reason = "selected typed subscriber and installed package observed; driver physics unverified"
                pointers = [
                    "/graph/topics/" + str(topics[0][0]),
                    "/qos/endpoints/" + str(subscribers[0][0]),
                    "/observations/package_inventory/"
                    + str(packages.index(declaration.driver_package)),
                ]
    observed = {row["semantic_id"]: row for row in resolve_capabilities(model, now=now)}
    required = NAVIGATION + (CLEANING if task_class == "complete_area_cleaning" else ())
    nodes = []
    for semantic in required:
        if semantic == "base.ros2_control_interface":
            nodes.append(
                {"semantic_id": semantic, "status": driver_status, "original_pointers": pointers}
            )
        else:
            value = observed.get(semantic)
            nodes.append(
                {
                    "semantic_id": semantic,
                    "status": value["status"] if fresh and value else "UNKNOWN",
                    "original_observation": value if fresh else None,
                }
            )
    return {
        "schema_version": "rosclaw.task_capability_graph.v1",
        "snapshot_hash": model.snapshot_hash,
        "reviewed_at": now.isoformat(),
        "task_class": task_class,
        "control_prerequisite": {
            "status": driver_status,
            "reason": reason,
            "selection": declaration.model_dump(mode="json") if declaration else None,
            "driver_physics": "NOT_ESTABLISHED_BY_ENDPOINT_PRESENCE",
        },
        "nodes": nodes,
        "dependencies": [
            {"consumer": "navigation.navigate_to_pose", "requires": role}
            for role in NAVIGATION
            if role != "navigation.navigate_to_pose"
        ]
        + (
            [
                {"consumer": "coverage.execute", "requires": role}
                for role in (
                    "navigation.navigate_to_pose",
                    "coverage.compute_path",
                    "cleaning.enable",
                    "cleaning.disable",
                )
            ]
            + [{"consumer": "coverage.verify", "requires": "coverage.execute"}]
            if task_class == "complete_area_cleaning"
            else []
        ),
        "execution_authority": "NONE",
        "current_l1": "NOT_EVALUATED",
        "physical_acceptance": "NOT_RUN",
    }
