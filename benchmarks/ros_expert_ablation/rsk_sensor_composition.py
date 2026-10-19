"""Opt-in reuse of existing source-envelope filtering in proposed SIM apps.

No ROS, subprocess, Runtime, driver, installation or execution occurs here.
An explicit operator-owned component catalog is required. The installed-driver
observations and frozen Core output are retained, never rewritten as current
filter observations. Actual package/bridge/source checks stay in the SDK owner.
"""

import copy
import hashlib
import json
from typing import Literal

import yaml
from pydantic import Field, StrictStr, model_validator

from benchmarks.ros_expert_ablation.physical_aci import Args
from integrations.ros_probe.acceptance.simulation_source_body_scan_filter import (
    PACKAGE_BYTES,
    PACKAGE_SHA256,
    check_original_scan_bridge,
    source_body_filter_parameters,
)
from rosclaw.connectors.ros.context.collision_assets import collision_assets_from_observations
from rosclaw.contracts.common import content_hash

INPUTS_V2 = "rosclaw.rsk_application_inputs.v2"
SDK_IMAGE = "sha256:8cc3fbf1eaa57058f0c1386c061253a55651106bd46b98fb046439f966c9019b"


class FixedFilterArtifactV1(Args):
    path: StrictStr = Field(min_length=1, max_length=4096)
    sha256: Literal[PACKAGE_SHA256]


class SourceSensorComponentV1(Args):
    schema_version: Literal["rosclaw.source_envelope_sensor_component.v1"]
    scope: Literal["EXPLICIT_OPERATOR_OWNED_DISPOSABLE_SIM_ONLY"]
    component: Literal["laser_filters/StaticLaserScanPolygonFilter"]
    availability: Literal["PINNED_ARTIFACT_NOT_CURRENTLY_INSTALLED_OR_ADMITTED"]
    sdk_image: Literal[SDK_IMAGE]
    artifact: FixedFilterArtifactV1
    artifact_bytes: Literal[PACKAGE_BYTES]
    source_urdf_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    scan_bridge_yaml: StrictStr = Field(min_length=1, max_length=65536)
    scan_bridge_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def original_bridge_bytes(self):
        if hashlib.sha256(self.scan_bridge_yaml.encode()).hexdigest() != self.scan_bridge_sha256:
            raise ValueError("original scan bridge bytes changed")
        return self


def compose_source_sensor_component(core_plan, model, urdf_bytes, catalog):
    """Compose a conditional four-file plan from actual roles, with no robot branch."""
    component = SourceSensorComponentV1.model_validate(catalog)
    saved = copy.deepcopy(core_plan)
    core_hash = saved.pop("plan_hash")
    if core_hash != content_hash("driverbootstrap", saved):
        raise ValueError("frozen Core application plan changed")
    if core_plan["status"] != "CONDITIONAL_DRIVER_BOOTSTRAP_STAGING":
        return copy.deepcopy(core_plan)
    if model.environment.get("distro") != "jazzy":
        raise ValueError("fixed optional filter artifact supports the declared Jazzy SDK only")
    if hashlib.sha256(urdf_bytes).hexdigest() != component.source_urdf_sha256:
        raise ValueError("optional component differs from the current original URDF")
    if core_plan["source_urdf_sha256"] != component.source_urdf_sha256:
        raise ValueError("optional component differs from Core source binding")
    for name, raw in core_plan["files"].items():
        if hashlib.sha256(raw.encode()).hexdigest() != core_plan["file_sha256"][name]:
            raise ValueError("original Core application files changed")
    frames = core_plan["current_body_candidate"]["frames"]
    bringup = json.loads(core_plan["files"]["bringup.json"])
    if "source_body_scan_filter" in bringup:
        raise ValueError(
            "pre-existing filter declaration requires an explicit separate composition"
        )
    declaration = {
        "schema_version": "rosclaw.sdk_source_body_scan_filter.v1",
        "source_urdf_sha256": component.source_urdf_sha256,
        "base_frame": frames["base"],
        "scan_frame": frames["lidar"],
    }
    bringup["source_body_scan_filter"] = declaration
    # Use the same pure source/geometry/transform and typed bridge checks as the
    # existing SDK implementation. No polygon or geometry comes from the model.
    mesh = collision_assets_from_observations(model.observations)
    _, original_geometry = source_body_filter_parameters(
        bringup, urdf_bytes, mesh_originals=mesh[0] if mesh is not None else None
    )
    check_original_scan_bridge(bringup, yaml.safe_load(component.scan_bridge_yaml))
    plan = copy.deepcopy(core_plan)
    plan.pop("plan_hash")
    plan["schema_version"] = "rosclaw.source_sensor_composed_driver_application.v1"
    plan["core_compiler_plan_hash"] = core_hash
    plan["sensor_component_inputs_hash"] = content_hash(
        "rsksensorcomponent", component.model_dump(mode="json")
    )
    plan["sensor_component"] = {
        "component": component.component,
        "artifact": component.artifact.model_dump(mode="json"),
        "artifact_bytes": component.artifact_bytes,
        "required_SDK_image": component.sdk_image,
        "artifact_load_verified": False,
        "declaration": declaration,
        "original_scan_bridge_sha256": component.scan_bridge_sha256,
        "original_geometry": original_geometry,
        "raw_source_preserved": True,
        "consumer_scope": "COLLISION_MONITOR_SENSOR_INPUT_ONLY",
        "runtime_installation": "NOT_RUN_REQUIRES_OWNED_SDK_PACKAGE_CHECK",
        "component_rollback": {
            "file": "bringup.json",
            "pointer": "/source_body_scan_filter",
            "operation": "REMOVE",
            "before": declaration,
            "restored_file_sha256": core_plan["file_sha256"]["bringup.json"],
        },
        "current_authority": "NONE",
    }
    plan["files"]["bringup.json"] = json.dumps(bringup, indent=2) + "\n"
    plan["file_sha256"] = {
        name: hashlib.sha256(raw.encode()).hexdigest() for name, raw in plan["files"].items()
    }
    plan["validation_plan"] += [
        "owned SDK verifies and loads the exact fixed optional filter artifact",
        "preserve raw scan and all original Body limits; verify only CM sensor remapping",
        "new-World geometry, filter/TF failure closure, full L1 and canonical stop/contact proof",
    ]
    return {"plan_hash": content_hash("rskcomposeddriverbootstrap", plan), **plan}
