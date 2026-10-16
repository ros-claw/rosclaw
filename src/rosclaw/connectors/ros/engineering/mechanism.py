"""Evidence-bound historical repair mechanisms; never executable authority.

The operator verifies original bytes before distillation and again at retrieval.
A mechanism can suggest an intent; the current typed compiler and independent
new-World verification remain mandatory. Historical geometry never transfers.
"""

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rosclaw.contracts.common import content_hash

from .contracts import RosEngineeringCaseV1
from .repair_compiler import compile_repair

MechanismKind = Literal[
    "enable_localization_tf",
    "resume_scan_bridge",
    "activate_managed_nodes",
    "cover_urdf_envelope",
    "disable_owned_fixture_endpoint",
]
ISSUES = {
    "enable_localization_tf": "F-TF-001",
    "resume_scan_bridge": "F-SCAN-001",
    "activate_managed_nodes": "F-LIFE-001",
    "cover_urdf_envelope": "F-GEOM-001",
    "disable_owned_fixture_endpoint": "F-ROUTE-001",
}
LESSONS = {
    "enable_localization_tf": (
        "Disabled localization TF broadcast can leave map-to-odom absent",
        "localization.tf_broadcast=true",
        "competing authoritative map-to-odom publisher; inactive localization; unknown TF ownership",
    ),
    "resume_scan_bridge": (
        "An explicitly paused owned scan bridge can leave scan observations stale",
        "resume original staged scan bridge",
        "scan stale without an original paused bridge declaration; wrong source",
    ),
    "activate_managed_nodes": (
        "Omitting a required node from a managed list can leave it inactive",
        "append missing current required observed nodes; preserve current manager prefix",
        "ambiguous manager; stale readback; unknown nodes or changed existing prefix",
    ),
    "cover_urdf_envelope": (
        "An undersized circular costmap envelope can exclude the physical collision envelope",
        "derive radius from current original URDF; never copy a historical numeric radius",
        "different URDF; unsupported geometry; contradictory polygon/circle; ambiguous publisher",
    ),
    "disable_owned_fixture_endpoint": (
        "A separately declared owned fixture can compete on the protected command route",
        "disable only the current host-proven owned additional staging endpoint",
        "unowned or expected controller/monitor endpoint; missing current ownership proof; live process action",
    ),
}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class ArtifactRef(Strict):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ThreeWorldProof(Strict):
    source: str = Field(pattern=r"^[0-9a-f]{40}$")
    normalized_world_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    measured_capture_to_receipt_sec: float = Field(ge=0)
    sealed_archive_manifest: ArtifactRef
    before_run: str = Field(pattern=r"^[0-9a-f]{32}$")
    after_run: str = Field(pattern=r"^[0-9a-f]{32}$")
    rollback_run: str = Field(pattern=r"^[0-9a-f]{32}$")
    episode: ArtifactRef
    before: ArtifactRef
    after_l1: ArtifactRef
    rollback: ArtifactRef
    canonical_receipts: list[ArtifactRef] = Field(min_length=2, max_length=2)
    stop_source: ArtifactRef
    asset_refs: list[ArtifactRef] = Field(min_length=6)

    @model_validator(mode="after")
    def distinct_worlds(self):
        if len({self.before_run, self.after_run, self.rollback_run}) != 3:
            raise ValueError("three distinct original World identities required")
        if (
            len({ref.sha256 for ref in self.canonical_receipts}) != 2
            or len({ref.path for ref in self.canonical_receipts}) != 2
        ):
            raise ValueError("two distinct canonical receipt references required")
        return self


class MechanismExperienceV1(Strict):
    schema_version: Literal["rosclaw.ros_repair_mechanism.v1"] = "rosclaw.ros_repair_mechanism.v1"
    experience_hash: str
    kind: MechanismKind
    source_urdf_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_only: Literal[True] = True
    mechanism: str
    semantic_remedy: str
    contraindications: list[str]
    observation_signature: str
    original_changes: list[dict[str, Any]] = Field(min_length=1, max_length=8)
    proof: ThreeWorldProof
    evidence_scope: Literal["MEASURED_THREE_WORLD_SIM_REPAIR_ONLY"] = (
        "MEASURED_THREE_WORLD_SIM_REPAIR_ONLY"
    )
    body_geometry_transferable: Literal[False] = False
    current_authority: Literal["NONE"] = "NONE"
    current_l1: Literal["NOT_EVALUATED"] = "NOT_EVALUATED"

    @classmethod
    def from_dict(cls, value):
        record = cls.model_validate(value)
        payload = record.model_dump(mode="json")
        digest = payload.pop("experience_hash")
        if digest != content_hash("rosmechanism", payload):
            raise ValueError("historical mechanism hash mismatch")
        return record


def _diff(before, after, path=()):
    if type(before) is dict and type(after) is dict:
        if set(before) != set(after):
            raise ValueError("historical repair must preserve staging shape")
        rows = []
        for key in sorted(before):
            rows.extend(_diff(before[key], after[key], (*path, key)))
        return rows
    if type(before) is type(after) and before == after:
        return []
    return [{"path": list(path), "before": before, "after": after}]


def distill_mechanism(before_case, before_staging, after_staging, rollback_staging, proof):
    """Pure distillation after host original-file/proof checks, without gold labels."""
    case = RosEngineeringCaseV1.from_dict(before_case.model_dump(mode="json"))
    proof = ThreeWorldProof.model_validate(proof)
    if before_staging != rollback_staging:
        raise ValueError("rollback does not restore original staging")
    if set(before_staging) != {"nav2.yaml", "bringup.json"}:
        raise ValueError("closed original staging required")
    changes = _diff(before_staging, after_staging)
    if not changes or len(changes) > 8:
        raise ValueError("bounded actual historical repair required")
    for row in changes:
        path = row["path"]
        if path[0] != "nav2.yaml" and path not in (
            ["bringup.json", "scan_bridge_state"],
            ["bringup.json", "additional_command_endpoint", "enabled"],
        ):
            raise ValueError("historical repair outside closed staging fields")
        if path[0] == "nav2.yaml" and not (
            path[-3:] == ["amcl", "ros__parameters", "tf_broadcast"]
            or path[-3:] == ["lifecycle_manager_navigation", "ros__parameters", "node_names"]
            or path[-4:]
            in (
                ["local_costmap", "local_costmap", "ros__parameters", "robot_radius"],
                ["global_costmap", "global_costmap", "ros__parameters", "robot_radius"],
            )
        ):
            raise ValueError("historical repair outside closed navigation parameter fields")
    leaves = {row["path"][-1] for row in changes}
    if leaves == {"tf_broadcast"} and all(
        row["before"] is False and row["after"] is True for row in changes
    ):
        kind = "enable_localization_tf"
    elif leaves == {"scan_bridge_state"} and all(
        row["path"] == ["bringup.json", "scan_bridge_state"]
        and row["before"] == "paused_after_initialization"
        and row["after"] == "running"
        for row in changes
    ):
        kind = "resume_scan_bridge"
    elif leaves == {"node_names"} and all(
        type(row["before"]) is list
        and type(row["after"]) is list
        and set(row["before"]) < set(row["after"])
        for row in changes
    ):
        kind = "activate_managed_nodes"
    elif leaves == {"robot_radius"} and all(
        type(row["before"]) in (int, float)
        and type(row["after"]) in (int, float)
        and math.isfinite(row["after"])
        and row["after"] > row["before"]
        for row in changes
    ):
        kind = "cover_urdf_envelope"
    elif leaves == {"enabled"} and all(
        row["path"] == ["bringup.json", "additional_command_endpoint", "enabled"]
        and row["before"] is True
        and row["after"] is False
        for row in changes
    ):
        kind = "disable_owned_fixture_endpoint"
    else:
        raise ValueError("historical change is not one closed semantic mechanism")
    if not any(c.id == ISSUES[kind] and c.status == "VIOLATED" for c in case.constraints):
        raise ValueError("historical mechanism lacks original violated observation")
    mechanism, remedy, contraindication = LESSONS[kind]
    if kind == "cover_urdf_envelope":
        radius = case.derived_documents["navigation_observation_review"]["candidate"]["geometry"][
            "physical_radius_m"
        ]
        if any(row["after"] < radius for row in changes):
            raise ValueError("historical repaired radius still excludes original URDF")
    payload = {
        "schema_version": "rosclaw.ros_repair_mechanism.v1",
        "kind": kind,
        "source_urdf_sha256": case.source_urdf_sha256,
        "training_only": True,
        "mechanism": mechanism,
        "semantic_remedy": remedy,
        "contraindications": [contraindication],
        "observation_signature": ISSUES[kind] + ":VIOLATED",
        "original_changes": changes,
        "proof": proof.model_dump(mode="json"),
        "evidence_scope": "MEASURED_THREE_WORLD_SIM_REPAIR_ONLY",
        "body_geometry_transferable": False,
        "current_authority": "NONE",
        "current_l1": "NOT_EVALUATED",
    }
    return MechanismExperienceV1.from_dict(
        {"experience_hash": content_hash("rosmechanism", payload), **payload}
    )


def instantiate_mechanism(
    record, case, model, requirements, staging, *, minimum_radius_m, ownership_reviews=None
):
    """Re-derive a current intent, then use the existing typed repair compiler.

    Historical parameter values, namespaces, PID identities and radii are never
    copied into current staging. An independent new SDK World is still required.
    """
    record = MechanismExperienceV1.from_dict(record.model_dump(mode="json"))
    if record.source_urdf_sha256 != case.source_urdf_sha256:
        raise ValueError("historical geometry/Body transfer is not admitted")
    intent: dict[str, Any] = {"kind": record.kind}
    if record.kind == "cover_urdf_envelope":
        intent["radius_m"] = math.ceil(minimum_radius_m * 10) / 10
    elif record.kind == "activate_managed_nodes":
        managers = [
            (node, params)
            for node, params in model.observations.get("node_parameters", {}).items()
            if "node_names" in params
        ]
        if len(managers) != 1:
            raise ValueError("unique current observed manager required")
        node, params = managers[0]
        namespace = node.rsplit("/", 1)[0]
        names = list(params["node_names"])
        resolved = {name if name.startswith("/") else namespace + "/" + name for name in names}
        names += [
            name.rsplit("/", 1)[1] if name.rsplit("/", 1)[0] == namespace else name
            for name in requirements["lifecycle_nodes"].values()
            if name not in resolved
        ]
        intent["node_names"] = names
    elif record.kind == "disable_owned_fixture_endpoint":
        if len(ownership_reviews or {}) != 1:
            raise ValueError("unique current operator ownership review required")
        intent["ownership_ref"] = next(iter(ownership_reviews))
    plan = compile_repair(
        case,
        model,
        requirements,
        staging,
        intent,
        minimum_radius_m=minimum_radius_m,
        ownership_reviews=ownership_reviews,
    )
    return {
        "historical_experience_ref": record.experience_hash,
        "intent": intent,
        "proposal": plan,
        "historical_values_reused": False,
        "current_authority": "NONE",
    }
