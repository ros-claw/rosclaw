"""Typed semantic repairs compile to proposals only; no files or ROS are written."""

from __future__ import annotations

import copy
import math
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, TypeAdapter

from rosclaw.contracts.common import content_hash

from .contracts import RosEngineeringCaseV1


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class EnableLocalizationTF(Args):
    kind: Literal["enable_localization_tf"]


class ResumeScanBridge(Args):
    kind: Literal["resume_scan_bridge"]


class ActivateManagedNodes(Args):
    kind: Literal["activate_managed_nodes"]
    node_names: list[StrictStr] = Field(min_length=1, max_length=32)


class CoverURDF(Args):
    kind: Literal["cover_urdf_envelope"]
    radius_m: float = Field(gt=0)


class DisableOwnedFixtureEndpoint(Args):
    kind: Literal["disable_owned_fixture_endpoint"]
    ownership_ref: StrictStr = Field(min_length=1, max_length=256)


SemanticRepair = Annotated[
    EnableLocalizationTF
    | ResumeScanBridge
    | ActivateManagedNodes
    | CoverURDF
    | DisableOwnedFixtureEndpoint,
    Field(discriminator="kind"),
]
REPAIR_ADAPTER = TypeAdapter(SemanticRepair)


def _parts(pointer):
    if not pointer.startswith("/"):
        raise ValueError("operator JSON pointer required")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _get(document, pointer):
    for part in _parts(pointer):
        document = document[part]
    return document


def _set(document, pointer, value):
    parts = _parts(pointer)
    parent = document
    for part in parts[:-1]:
        parent = parent[part]
    parent[parts[-1]] = copy.deepcopy(value)


def _paths(value, prefix=()):
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        if key == "ros__parameters":
            yield prefix, item
        elif isinstance(item, dict):
            yield from _paths(item, (*prefix, key))


def _pointer(parts):
    return "/" + "/".join(str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def compile_repair(
    case: RosEngineeringCaseV1,
    model,
    requirements,
    staging,
    intent,
    *,
    minimum_radius_m,
    ownership_reviews=None,
):
    """Bind semantic intent to original observations and minimal staging diffs.

    Preconditions needing fresh probes are retained as proof obligations. No
    historical case, proposal or ownership review grants execution authority.
    """
    case = RosEngineeringCaseV1.from_dict(case.model_dump(mode="json"))
    if (
        model.snapshot_hash != case.snapshot_hash
        or model.compute_snapshot_hash() != case.snapshot_hash
    ):
        raise ValueError("proposal/case/original snapshot mismatch")
    if case.source_status == "MISMATCH":
        raise ValueError("mismatched source cannot propose repair")
    repair = REPAIR_ADAPTER.validate_python(intent)
    states = {constraint.id: constraint.status for constraint in case.constraints}
    required = {
        "enable_localization_tf": "F-TF-001",
        "resume_scan_bridge": "F-SCAN-001",
        "activate_managed_nodes": "F-LIFE-001",
        "cover_urdf_envelope": "F-GEOM-001",
        "disable_owned_fixture_endpoint": "F-ROUTE-001",
    }[repair.kind]
    if states.get(required) != "VIOLATED":
        raise ValueError(
            "repair requires an observed violated constraint; unknown/healthy is not actionable"
        )
    changes = []
    preconditions = []

    def change(file, pointer, value):
        prior = _get(staging[file], pointer)
        if prior == value:
            raise ValueError("no-op proposal is not a repair")
        changes.append(
            {
                "file": file,
                "pointer": pointer,
                "before": copy.deepcopy(prior),
                "value": copy.deepcopy(value),
            }
        )

    node_paths = {}
    for path, params in _paths(staging["nav2.yaml"]):
        node = "/" + "/".join(str(part).strip("/") for part in path)
        if node in node_paths:
            raise ValueError("ambiguous staged node binding")
        node_paths[node] = (path, params)
    observed = model.observations.get("node_parameters", {})
    if isinstance(repair, EnableLocalizationTF):
        node = requirements["lifecycle_nodes"]["localization"]
        path, params = node_paths[node]
        if (
            params.get("tf_broadcast") is not False
            or observed.get(node, {}).get("tf_broadcast") is not False
        ):
            raise ValueError("original parameter readback and staging must agree")
        candidate = [
            r for r in case.repair_candidates if r.semantic_field == "localization.tf_broadcast"
        ]
        if len(candidate) != 1:
            raise ValueError("no evidence-supported conditional TF candidate")
        change("nav2.yaml", _pointer((*path, "ros__parameters", "tf_broadcast")), True)
        preconditions.extend(candidate[0].preconditions)
    elif isinstance(repair, ResumeScanBridge):
        if staging["bringup.json"].get("scan_bridge_state") != "paused_after_initialization":
            raise ValueError("stale scan alone does not prove a paused bridge")
        change("bringup.json", "/scan_bridge_state", "running")
        preconditions.append(
            "confirm staged bridge policy is the relevant source failure in new World"
        )
    elif isinstance(repair, ActivateManagedNodes):
        managers = [
            (node, path, params)
            for node, (path, params) in node_paths.items()
            if "node_names" in params
            and observed.get(node, {}).get("node_names") == params["node_names"]
        ]
        if len(managers) != 1:
            raise ValueError("unique original managed-node readback required")
        manager, path, params = managers[0]
        stamp = model.observations.get("parameter_captured_at", {}).get(manager)
        try:
            age = (
                datetime.fromisoformat(case.reviewed_at) - datetime.fromisoformat(stamp)
            ).total_seconds() * 1000
        except (ValueError, TypeError) as error:
            raise ValueError(
                "current original managed-node parameter timestamp required"
            ) from error
        if not 0 <= age <= 500:
            raise ValueError("stale managed-node readback cannot compile repair")
        prior = params["node_names"]
        requested = repair.node_names
        namespace = manager.rsplit("/", 1)[0]
        if (
            not isinstance(prior, list)
            or len(requested) != len(set(requested))
            or requested[: len(prior)] != prior
        ):
            raise ValueError("managed-node repair may only append missing selected roles")
        selected = set(requirements["lifecycle_nodes"].values())
        existing_nodes = {row.name for row in model.lifecycle}
        added = requested[len(prior) :]
        resolved = [name if name.startswith("/") else namespace + "/" + name for name in requested]
        if len(resolved) != len(set(resolved)):
            raise ValueError("duplicate resolved managed-node identity")
        if not added or any(
            (name if name.startswith("/") else namespace + "/" + name)
            not in selected & existing_nodes
            for name in added
        ):
            raise ValueError("only observed selected missing lifecycle roles may be added")
        change("nav2.yaml", _pointer((*path, "ros__parameters", "node_names")), requested)
    elif isinstance(repair, CoverURDF):
        radius = case.derived_documents["navigation_observation_review"]["candidate"]["geometry"][
            "physical_radius_m"
        ]
        if (
            type(minimum_radius_m) not in (int, float)
            or not math.isfinite(minimum_radius_m)
            or radius != minimum_radius_m
            or repair.radius_m < radius
        ):
            raise ValueError("immutable original URDF envelope cannot shrink")
        gap = next(c for c in case.constraints if c.id == required)
        for topic in gap.observed["undersized_costmaps"]:
            matches = [row for row in model.graph["topics"] if row["name"] == topic]
            if len(matches) != 1 or len(matches[0].get("publishers", [])) != 1:
                raise ValueError("unique original costmap publisher required")
            node = matches[0]["publishers"][0]
            path, params = node_paths[node]
            if params.get("robot_radius") != observed.get(node, {}).get("robot_radius"):
                raise ValueError("costmap original readback/staging mismatch")
            change(
                "nav2.yaml", _pointer((*path, "ros__parameters", "robot_radius")), repair.radius_m
            )
        if not changes:
            raise ValueError("no observed undersized costmap")
        preconditions.append("review padding and full effective footprint using existing L1")
    else:
        review = (ownership_reviews or {}).get(repair.ownership_ref)
        declaration = staging["bringup.json"].get("additional_command_endpoint")
        if (
            not isinstance(review, dict)
            or review.get("case_hash") != case.case_hash
            or review.get("staging_hash") != content_hash("acistaging", staging)
            or review.get("scope") != "SDK_OWNED_ADDITIONAL_STAGING_ENDPOINT"
            or review.get("declaration") != declaration
            or not review.get("original_evidence_refs")
        ):
            raise ValueError(
                "operator-pinned ownership review required; never disable a node by its name"
            )
        if not isinstance(declaration, dict) or declaration.get("enabled") is not True:
            raise ValueError("enabled owned staging endpoint required")
        owned_node = review.get("node_name")
        topic = declaration.get("topic")
        matches = [row for row in model.graph.get("topics", []) if row.get("name") == topic]
        expected = {
            requirements["velocity_input_topic"]: requirements["lifecycle_nodes"]["controller"],
            requirements["velocity_output_topic"]: requirements["lifecycle_nodes"][
                "collision_monitor"
            ],
        }
        if (
            len(matches) != 1
            or topic not in expected
            or owned_node not in matches[0].get("publishers", [])
            or owned_node == expected[topic]
            or matches[0].get("msg_type") != declaration.get("msg_type")
        ):
            raise ValueError(
                "owned declaration must identify an observed competing protected publisher"
            )
        change("bringup.json", "/additional_command_endpoint/enabled", False)
        preconditions.append("new World only; never mutate or terminate original live endpoint")
    refs = next(c.evidence_refs for c in case.constraints if c.id == required)
    payload = {
        "schema_version": "rosclaw.typed_staging_proposal.v1",
        "case_hash": case.case_hash,
        "snapshot_hash": case.snapshot_hash,
        "source_urdf_sha256": case.source_urdf_sha256,
        "base_staging_hash": content_hash("acistaging", staging),
        "requirements_hash": content_hash("acirequirements", requirements),
        "intent": repair.model_dump(mode="json"),
        "changes": changes,
        "rollback": [
            {"file": c["file"], "pointer": c["pointer"], "value": c["before"]}
            for c in reversed(changes)
        ],
        "evidence_refs": refs,
        "preconditions": preconditions,
        "proof_obligations": [
            "new_World_fresh_source_and_Body_binding",
            "independent_full_L1_READY",
            "two_canonical_navigation_receipts",
            "independent_stop_zero_collision_and_source_continuity",
        ],
        "stage_only": True,
        "current_authority": "NONE",
        "admission": "NOT_ADMITTED",
    }
    return {"proposal_id": content_hash("aciproposal", payload), **payload}


def compile_repair_sequence(
    case, model, requirements, staging, intents, *, minimum_radius_m, ownership_reviews=None
):
    """Compile disjoint repairs against one immutable original staging baseline.

    Never feed one proposed repair into another compiler: original parameter
    readbacks and whole-staging ownership reviews stay bound to the same capture.
    A combined proposal still needs independent new-World physical validation.
    """
    if type(intents) is not list or not 2 <= len(intents) <= 5:
        raise ValueError("two to five distinct semantic repairs required")
    typed = [REPAIR_ADAPTER.validate_python(intent) for intent in intents]
    if len({repair.kind for repair in typed}) != len(typed):
        raise ValueError("duplicate semantic repair kind")
    typed.sort(key=lambda repair: repair.kind)
    components = [
        compile_repair(
            case,
            model,
            requirements,
            staging,
            repair.model_dump(mode="json"),
            minimum_radius_m=minimum_radius_m,
            ownership_reviews=ownership_reviews,
        )
        for repair in typed
    ]
    changes = []
    paths = []
    for component in components:
        for change in component["changes"]:
            path = (change["file"], *_parts(change["pointer"]))
            if any(path[: len(prior)] == prior or prior[: len(path)] == path for prior in paths):
                raise ValueError("overlapping compound staging fields")
            paths.append(path)
            changes.append(copy.deepcopy(change))
    if not 1 <= len(changes) <= 8:
        raise ValueError("existing operator eight-field patch budget exceeded")
    payload = {
        key: copy.deepcopy(value)
        for key, value in components[0].items()
        if key
        not in {"proposal_id", "intent", "changes", "rollback", "evidence_refs", "preconditions"}
    }
    payload.update(
        intent={
            "kind": "compound_repair",
            "repairs": [repair.model_dump(mode="json") for repair in typed],
        },
        component_proposal_ids=[component["proposal_id"] for component in components],
        changes=changes,
        rollback=[
            {"file": row["file"], "pointer": row["pointer"], "value": copy.deepcopy(row["before"])}
            for row in reversed(changes)
        ],
        evidence_refs=list(
            dict.fromkeys(ref for component in components for ref in component["evidence_refs"])
        ),
        preconditions=list(
            dict.fromkeys(
                condition for component in components for condition in component["preconditions"]
            )
        ),
    )
    return {"proposal_id": content_hash("aciproposal", payload), **payload}


def proposal_staging(proposal, staging):
    """Pure staging clone; no writes and no admission. Host must validate policy."""
    payload = {k: v for k, v in proposal.items() if k != "proposal_id"}
    if proposal.get("proposal_id") != content_hash("aciproposal", payload) or proposal.get(
        "base_staging_hash"
    ) != content_hash("acistaging", staging):
        raise ValueError("proposal/base changed; recompile required")
    result = copy.deepcopy(staging)
    for change in proposal["changes"]:
        if _get(staging[change["file"]], change["pointer"]) != change["before"]:
            raise ValueError("original staging precondition differs")
        _set(result[change["file"]], change["pointer"], change["value"])
    return result
