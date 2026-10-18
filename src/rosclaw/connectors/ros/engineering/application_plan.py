"""Source-bound ROS2 application reuse plans; no installation or execution."""

import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import Field

from rosclaw.connectors.ros.context.navigation_review import review_navigation_candidate
from rosclaw.contracts.common import content_hash

from ..intelligence.graph_contracts import GraphRecord, RosOperationalGraphV1
from ..resolver.component_compatibility import COMPONENTS, review_component_compatibility
from ..resolver.task_capabilities import ExistingRos2ControlV1, resolve_task_capability_graph

VALIDATION = (
    "independent_original_driver_package_endpoint_and_manifest_review",
    "review_pinned_reused_launch_params_BT_and_adapter_no_guessed_control_interface",
    "new_owned_isolated_SIM_World_with_original_URDF_physics_and_Source_binding",
    "fresh_full_L1_READY_and_new_immutable_Body_before_any_action",
    "two_canonical_MCP_rosclawd_navigation_receipts_with_independent_stop",
    "zero_contacts_source_continuity_unchanged_500ms_source_250ms_proof_and_Body_limits",
    "coverage_and_cleaner_are_separate_bindings_and_independent_measured_verification",
)


class ApplicationRequestV1(GraphRecord):
    task_class: Literal["navigate_to_pose", "complete_area_cleaning"]
    control: ExistingRos2ControlV1 | None = None
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    evidence_domain: Literal["SIMULATION"] = "SIMULATION"
    component_policies: list[dict] = Field(min_length=1, max_length=32)


class RosApplicationPlanV1(GraphRecord):
    schema_version: Literal["rosclaw.ros_application_plan.v1"] = "rosclaw.ros_application_plan.v1"
    plan_hash: str
    snapshot_hash: str
    graph_hash: str
    source_ref: str
    evidence_domain: Literal["SEALED_REPLAY", "OBSERVATION"]
    reviewed_at: datetime
    source_urdf_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requirements_hash: str
    installed_manifests_hash: str
    request: ApplicationRequestV1
    status: Literal[
        "ABSTAIN",
        "NEEDS_DRIVER",
        "NEEDS_READ_ONLY_PROBE",
        "NEEDS_DEPENDENCIES",
        "INCOMPATIBLE",
        "NEEDS_APPLICATION_BRINGUP",
        "NEEDS_SEPARATE_COVERAGE_BINDING",
        "CONDITIONAL_REUSE_STAGING",
    ]
    reasons: list[str] = Field(max_length=32)
    target_capability_graph: dict
    component_compatibility: dict
    required_packages: list[str] = Field(max_length=32)
    missing_package_policies: list[str] = Field(max_length=32)
    navigation_observation_review: dict | None = None
    validation_plan: tuple[str, ...] = VALIDATION
    scope: Literal["EXISTING_ROS2_DRIVER_COMPONENT_REUSE_NOT_DRIVER_GENERATION"] = (
        "EXISTING_ROS2_DRIVER_COMPONENT_REUSE_NOT_DRIVER_GENERATION"
    )
    execution_authority: Literal["NONE"] = "NONE"
    source_authentication: Literal["NOT_ESTABLISHED_BY_HASH"] = "NOT_ESTABLISHED_BY_HASH"
    current_l1: Literal["NOT_EVALUATED"] = "NOT_EVALUATED"
    physical_acceptance: Literal["NOT_RUN"] = "NOT_RUN"
    operator_review: Literal["REQUIRED_BEFORE_NEW_WORLD"] = "REQUIRED_BEFORE_NEW_WORLD"

    @classmethod
    def from_dict(cls, data):
        obj = cls.model_validate(data)
        payload = obj.model_dump(mode="json")
        digest = payload.pop("plan_hash")
        if (
            digest != content_hash("rosapplication", payload)
            or obj.validation_plan != VALIDATION
            or obj.reviewed_at.tzinfo is None
        ):
            raise ValueError(
                "application digest, aware original time and unchanged validation plan required"
            )
        return obj


def compile_application_plan(graph, model, urdf_bytes, requirements, original_manifests, request):
    """Plan reuse from current originals; missing drivers never become cmd_vel.

    NEEDS_APPLICATION_BRINGUP describes absent observed stack prerequisites. It
    never pretends to have synthesized or validated a new stack. Conditional
    reuse staging requires the existing generic navigation observation review.
    A source-only preview never grants Body, L1 or execution authority.
    """
    graph = RosOperationalGraphV1.from_dict(graph.model_dump(mode="json"))
    request = ApplicationRequestV1.model_validate(request)
    if (
        model.snapshot_hash != graph.snapshot_hash
        or model.compute_snapshot_hash() != graph.snapshot_hash
    ):
        raise ValueError("unchanged original model and graph required")
    if type(urdf_bytes) is not bytes or not 0 < len(urdf_bytes) <= 5_000_000:
        raise ValueError("bounded original expanded URDF bytes required")
    if type(requirements) is not dict or type(original_manifests) is not dict:
        raise ValueError("original requirements and installed manifests required")
    if len(json.dumps(original_manifests, allow_nan=False).encode()) > 3_000_000:
        raise ValueError("bounded original installed manifests required")
    control = request.control.model_dump(mode="json") if request.control else None
    target = resolve_task_capability_graph(
        model, request.task_class, control, now=graph.reviewed_at
    )
    compatibility = review_component_compatibility(
        model, request.component_policies, original_manifests
    )
    required = set(
        COMPONENTS["navigation"] + COMPONENTS["localization"] + COMPONENTS["obstacle_guard"]
    )
    if request.task_class == "complete_area_cleaning":
        required.update(COMPONENTS["coverage"])
    if request.control:
        required.add(request.control.driver_package)
    declared = {p["package"] for p in request.component_policies}
    missing = sorted(required - declared)
    values = {
        "plan_hash": "",
        "snapshot_hash": model.snapshot_hash,
        "graph_hash": graph.graph_hash,
        "source_ref": graph.source_ref,
        "evidence_domain": graph.evidence_domain,
        "reviewed_at": graph.reviewed_at,
        "source_urdf_sha256": hashlib.sha256(urdf_bytes).hexdigest(),
        "requirements_hash": content_hash("navrequirements", requirements),
        "installed_manifests_hash": content_hash("installedmanifests", original_manifests),
        "request": request,
        "target_capability_graph": target,
        "component_compatibility": compatibility,
        "required_packages": sorted(required),
        "missing_package_policies": missing,
    }

    def finish(status, reason):
        obj = RosApplicationPlanV1(**values, status=status, reasons=[reason])
        payload = obj.model_dump(mode="json")
        payload.pop("plan_hash")
        return RosApplicationPlanV1.from_dict(
            {"plan_hash": content_hash("rosapplication", payload), **payload}
        )

    if graph.source_status == "MISMATCH" or graph.snapshot_freshness != "FRESH":
        return finish("ABSTAIN", "original source mismatch or stale graph; no staging")
    driver = target["control_prerequisite"]["status"]
    if driver == "NEEDS_DRIVER":
        return finish("NEEDS_DRIVER", target["control_prerequisite"]["reason"])
    if driver == "INCOMPATIBLE":
        return finish("INCOMPATIBLE", target["control_prerequisite"]["reason"])
    if driver != "OBSERVED_ROS2_CONTROL_INTERFACE":
        return finish("NEEDS_READ_ONLY_PROBE", target["control_prerequisite"]["reason"])
    statuses = {r["status"] for r in compatibility["components"]}
    if statuses & {"VERSION_MISMATCH", "UNSUPPORTED_DECLARED_ENVIRONMENT"}:
        return finish("INCOMPATIBLE", "pinned component version or ROS/distro comparison failed")
    if "MISSING_DEPENDENCY" in statuses:
        return finish(
            "NEEDS_DEPENDENCIES", "required reuse package missing; no installation authority"
        )
    if missing or not compatibility["all_declared_requirements_match"]:
        return finish(
            "NEEDS_READ_ONLY_PROBE",
            "all required exact-version policies and actual installed manifests required",
        )
    review = review_navigation_candidate(model, urdf_bytes, requirements, now=graph.reviewed_at)
    values["navigation_observation_review"] = review
    if review["status"] != "PROPOSED":
        return finish(
            "NEEDS_APPLICATION_BRINGUP",
            "selected stack evidence incomplete; component dependencies and validation plan only, not synthesized stack",
        )
    if request.task_class == "complete_area_cleaning":
        return finish(
            "NEEDS_SEPARATE_COVERAGE_BINDING",
            "navigation reuse does not establish coverage planner, cleaner attachment, stop or measured coverage",
        )
    return finish(
        "CONDITIONAL_REUSE_STAGING",
        "original component reuse candidate only; independent new World source, fresh L1 and guarded physics remain required",
    )
