"""Bounded reuse bundle compilation; no writes, installation or World launch."""

import copy
import hashlib
import json
from pathlib import PurePosixPath
from typing import Literal
from xml.etree import ElementTree as ET

from pydantic import Field, model_validator

from rosclaw.contracts.common import content_hash

from ..intelligence.graph_contracts import GraphRecord
from .application_plan import RosApplicationPlanV1, compile_application_plan
from .intervention_plan import _node_parameters
from .staging_plan import parse_staging_documents

ROLES = {
    "NAV2_PARAMETERS": "nav2.yaml",
    "BRINGUP_DESCRIPTOR": "bringup.json",
    "BEHAVIOR_TREE": "behavior-tree.xml",
    "LAUNCH_REFERENCE": "launch-reference.json",
    "EXISTING_DRIVER_REFERENCE": "existing-driver-reference.json",
}


class ReusableApplicationArtifactV1(GraphRecord):
    role: Literal[
        "NAV2_PARAMETERS",
        "BRINGUP_DESCRIPTOR",
        "BEHAVIOR_TREE",
        "LAUNCH_REFERENCE",
        "EXISTING_DRIVER_REFERENCE",
    ]
    package: str = Field(pattern=r"^[a-z][a-z0-9_]{0,127}$")
    package_version: str = Field(min_length=1, max_length=64)
    original_package_path: str = Field(min_length=1, max_length=1024)
    content: str = Field(min_length=1, max_length=1_000_000)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_snapshot_hash: str
    source_urdf_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_authentication: Literal["OPERATOR_DECLARED_REUSE_REQUIRES_ORIGINAL_REVIEW"] = (
        "OPERATOR_DECLARED_REUSE_REQUIRES_ORIGINAL_REVIEW"
    )

    @model_validator(mode="after")
    def original_integrity(self):
        path = PurePosixPath(self.original_package_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or str(path) != self.original_package_path
            or "\\" in str(path)
        ):
            raise ValueError("literal relative installed package artifact path required")
        if (
            not path.parts
            or len(self.content.encode()) > 1_000_000
            or hashlib.sha256(self.content.encode()).hexdigest() != self.sha256
        ):
            raise ValueError("bounded byte-exact original reusable artifact required")
        return self


def compile_application_staging(
    plan, graph, model, urdf_bytes, requirements, original_manifests, artifacts
):
    """Assemble explicitly selected existing assets, never synthesize a driver.

    Source-bound reuse is deliberately smaller than arbitrary stack synthesis.
    Missing stack observations remain NEEDS_APPLICATION_BRINGUP in the planner.
    This compiler cannot turn a dependency plan into an executable application.
    Package metadata and copied content still need independent original review;
    nothing in this bundle admits a Body or authenticates driver physics.
    """
    plan = RosApplicationPlanV1.from_dict(plan.model_dump(mode="json"))
    recomputed = compile_application_plan(
        graph, model, urdf_bytes, requirements, original_manifests, plan.request
    )
    if recomputed.plan_hash != plan.plan_hash:
        raise ValueError(
            "application plan differs from original inputs; hashes alone do not authorize reuse"
        )
    if plan.status != "CONDITIONAL_REUSE_STAGING":
        raise ValueError(
            "observed, reviewed navigation reuse prerequisite required; no partial staging"
        )
    if type(artifacts) is not list or len(artifacts) != len(ROLES):
        raise ValueError("exactly the five explicit original reuse roles required")
    records = [ReusableApplicationArtifactV1.model_validate(v) for v in artifacts]
    if {a.role for a in records} != set(ROLES) or len(
        {(a.package, a.original_package_path) for a in records}
    ) != len(records):
        raise ValueError("unique original artifacts for every reuse role required")
    components = {c["package"]: c for c in plan.component_compatibility["components"]}
    for artifact in records:
        component = components.get(artifact.package)
        if (
            component is None
            or component["status"] != "DECLARED_VERSION_AND_DISTRO_MATCH"
            or component["installed_version"] != artifact.package_version
        ):
            raise ValueError("original artifact differs from declared installed component/version")
        if (
            artifact.target_snapshot_hash != plan.snapshot_hash
            or artifact.source_urdf_sha256 != plan.source_urdf_sha256
        ):
            raise ValueError(
                "reused artifacts must explicitly target this original Body/source, never historical geometry"
            )
    by_role = {a.role: a for a in records}
    staging = parse_staging_documents(
        by_role["NAV2_PARAMETERS"].content.encode(), by_role["BRINGUP_DESCRIPTOR"].content.encode()
    )
    json.dumps(staging, allow_nan=False)
    nodes = list(_node_parameters(staging["nav2.yaml"]))
    for selected in requirements["lifecycle_nodes"].values():
        if len([row for row in nodes if row[0] == selected]) != 1:
            raise ValueError(
                "reused parameters must explicitly cover each selected observed lifecycle node"
            )
    namespace = next(iter(requirements["lifecycle_nodes"].values())).rsplit("/", 1)[0]
    if staging["bringup.json"].get("namespace") != namespace:
        raise ValueError("reused bringup must explicitly target the original selected namespace")
    try:
        tree_text = by_role["BEHAVIOR_TREE"].content
        if "<!" in tree_text:
            raise ValueError("concrete BT XML without declarations required")
        tree = ET.fromstring(tree_text)
        launch = json.loads(by_role["LAUNCH_REFERENCE"].content)
        driver = json.loads(by_role["EXISTING_DRIVER_REFERENCE"].content)
    except (ET.ParseError, UnicodeError, RecursionError) as exc:
        raise ValueError("concrete bounded reuse references required") from exc
    json.dumps(launch, allow_nan=False)
    json.dumps(driver, allow_nan=False)
    if tree.tag != "root" or not tree.findall("BehaviorTree"):
        raise ValueError(
            "original explicit Nav2 behavior tree required; plugins remain independently unverified"
        )
    if (
        type(launch) is not dict
        or set(launch) != {"package", "launch_file", "arguments"}
        or launch["package"] != by_role["LAUNCH_REFERENCE"].package
        or type(launch["arguments"]) is not dict
    ):
        raise ValueError("explicit installed launch reference and declared arguments required")
    launch_path = (
        PurePosixPath(launch["launch_file"]) if type(launch["launch_file"]) is str else None
    )
    if (
        launch_path is None
        or launch_path.is_absolute()
        or ".." in launch_path.parts
        or not launch_path.parts
        or str(launch_path) != launch["launch_file"]
        or "\\" in str(launch_path)
    ):
        raise ValueError("relative installed launch file reference required")
    if (
        driver != plan.request.control.model_dump(mode="json")
        or by_role["EXISTING_DRIVER_REFERENCE"].package != plan.request.control.driver_package
    ):
        raise ValueError(
            "existing driver reference differs from observed explicit control prerequisite"
        )
    files = {ROLES[a.role]: a.content for a in records}
    result = {
        "schema_version": "rosclaw.application_staging_bundle.v1",
        "plan_hash": plan.plan_hash,
        "snapshot_hash": plan.snapshot_hash,
        "evidence_domain": plan.evidence_domain,
        "requires_current_remeasurement": True,
        "source_urdf_sha256": plan.source_urdf_sha256,
        "scope": "OPERATOR_REVIEWED_EXISTING_COMPONENT_REUSE_ONLY",
        "status": "PROPOSED_NOT_INSTALLED_OR_EXECUTED",
        "files": files,
        "file_sha256": {
            name: hashlib.sha256(raw.encode()).hexdigest() for name, raw in files.items()
        },
        "staging_hash": content_hash("acistaging", staging),
        "original_artifacts": [a.model_dump(mode="json", exclude={"content"}) for a in records],
        "validation_plan": list(plan.validation_plan),
        "current_l1": "NOT_EVALUATED",
        "source_authentication": "NOT_ESTABLISHED_BY_HASH",
        "execution_authority": "NONE",
        "driver_generated": False,
        "physical_acceptance": "NOT_RUN",
    }
    return copy.deepcopy({"bundle_hash": content_hash("applicationstaging", result), **result})
