"""Pure conditional staging options; physical preconditions remain unproved.

A static option is an unexecuted hypothesis, never current L1 or permission.
Every option uses the same original case, model, requirements and staging. No
fault-factory label, historical repair literal, I/O or execution gateway is used.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import StrictStr, model_validator

from rosclaw.contracts.common import content_hash

from .contracts import RosEngineeringCaseV1
from .repair_compiler import Args, SemanticRepair, compile_repair


class StagingChangeV1(Args):
    file: Literal["nav2.yaml", "bringup.json"]
    pointer: StrictStr
    before: Any
    value: Any


class StagingOptionV1(Args):
    schema_version: Literal["rosclaw.conditional_staging_option.v1"] = (
        "rosclaw.conditional_staging_option.v1"
    )
    option_ref: StrictStr
    issue_id: StrictStr
    case_candidate_id: StrictStr | None
    case_hash: StrictStr
    snapshot_hash: StrictStr
    source_urdf_sha256: StrictStr
    base_staging_hash: StrictStr
    intent: SemanticRepair
    proposal_id: StrictStr
    changes: list[StagingChangeV1]
    evidence_refs: list[StrictStr]
    original_candidate_preconditions: list[StrictStr]
    required_new_world_proofs: list[StrictStr]
    classification: Literal["INFERRED_CONDITIONAL_STAGING_HYPOTHESIS"] = (
        "INFERRED_CONDITIONAL_STAGING_HYPOTHESIS"
    )
    static_diff_checks: Literal["PASSED_NOT_PHYSICAL_ADMISSION"] = "PASSED_NOT_PHYSICAL_ADMISSION"
    current_authority: Literal["NONE"] = "NONE"
    current_case_l1: Literal["NOT_EVALUATED"] = "NOT_EVALUATED"

    @model_validator(mode="after")
    def valid_hash(self):
        payload = self.model_dump(mode="json")
        digest = payload.pop("option_ref")
        if digest != content_hash("acistagingoption", payload):
            raise ValueError("conditional staging option content hash mismatch")
        return self


def compile_staging_options(
    case: RosEngineeringCaseV1,
    model,
    requirements,
    staging,
    *,
    minimum_radius_m,
    ownership_reviews=None,
):
    """Compile available semantic diffs while retaining all physical obligations.

    Unknown/healthy constraints yield no repair. A symptom whose mechanism is not
    supported by original readbacks yields a rejection, rather than a guessed
    repair. The host must still independently apply its staging whitelist.
    """
    RosEngineeringCaseV1.from_dict(case.model_dump(mode="json"))
    if model.compute_snapshot_hash() != case.snapshot_hash:
        raise ValueError("original option snapshot changed")
    if case.source_status == "MISMATCH":
        return {"options": [], "blocked": [{"reason": "source mismatch"}]}
    options = []
    blocked = []
    for issue in case.constraints:
        if issue.status != "VIOLATED" or issue.id == "F-SOURCE-001":
            continue
        intent: dict[str, Any]
        try:
            if issue.id == "F-TF-001":
                intent = {"kind": "enable_localization_tf"}
            elif issue.id == "F-SCAN-001":
                intent = {"kind": "resume_scan_bridge"}
            elif issue.id == "F-LIFE-001":
                managers = [
                    (node, values)
                    for node, values in model.observations.get("node_parameters", {}).items()
                    if "node_names" in values
                ]
                if len(managers) != 1:
                    raise ValueError("unique original lifecycle manager required")
                node, values = managers[0]
                namespace = node.rsplit("/", 1)[0]
                names = list(values["node_names"])
                resolved = {
                    name if name.startswith("/") else namespace + "/" + name for name in names
                }
                names += [
                    role.rsplit("/", 1)[1] if role.rsplit("/", 1)[0] == namespace else role
                    for role in requirements["lifecycle_nodes"].values()
                    if role not in resolved
                ]
                intent = {"kind": "activate_managed_nodes", "node_names": names}
            elif issue.id == "F-GEOM-001":
                # The existing deterministic baseline uses the same conservative
                # rounding; it is a staging hypothesis, not a new safety limit.
                intent = {
                    "kind": "cover_urdf_envelope",
                    "radius_m": math.ceil(minimum_radius_m * 10) / 10,
                }
            elif issue.id == "F-ROUTE-001":
                if len(ownership_reviews or {}) != 1:
                    raise ValueError("operator-bound ownership review required")
                intent = {
                    "kind": "disable_owned_fixture_endpoint",
                    "ownership_ref": next(iter(ownership_reviews)),
                }
            else:
                raise ValueError("no supported staging mechanism for observed violation")
            proposal = compile_repair(
                case,
                model,
                requirements,
                staging,
                intent,
                minimum_radius_m=minimum_radius_m,
                ownership_reviews=ownership_reviews,
            )
            semantic = {
                "enable_localization_tf": "localization.tf_broadcast",
                "resume_scan_bridge": "sensor.scan_bridge_state",
                "activate_managed_nodes": "navigation.managed_nodes",
                "cover_urdf_envelope": "costmap.radius",
                "disable_owned_fixture_endpoint": "control.route_review",
            }[intent["kind"]]
            candidates = [r.id for r in case.repair_candidates if r.semantic_field == semantic]
            if len(candidates) > 1:
                raise ValueError("ambiguous original conditional candidate")
            payload = {
                "schema_version": "rosclaw.conditional_staging_option.v1",
                "issue_id": issue.id,
                "case_candidate_id": candidates[0] if candidates else None,
                **{
                    key: proposal[key]
                    for key in [
                        "case_hash",
                        "snapshot_hash",
                        "source_urdf_sha256",
                        "base_staging_hash",
                        "intent",
                        "proposal_id",
                        "changes",
                        "evidence_refs",
                    ]
                },
                "original_candidate_preconditions": proposal["preconditions"],
                "required_new_world_proofs": proposal["proof_obligations"],
                "classification": "INFERRED_CONDITIONAL_STAGING_HYPOTHESIS",
                "static_diff_checks": "PASSED_NOT_PHYSICAL_ADMISSION",
                "current_authority": "NONE",
                "current_case_l1": "NOT_EVALUATED",
            }
            options.append(
                StagingOptionV1.model_validate(
                    {"option_ref": content_hash("acistagingoption", payload), **payload}
                )
            )
        except (ValueError, KeyError, TypeError) as error:
            blocked.append({"issue_id": issue.id, "reason": str(error)[:400]})
    return {"options": options, "blocked": blocked}
