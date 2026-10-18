"""Transfer a historical TF concept; recompute current bindings and staging.

This separate path leaves original same-Body Memory admission unchanged. A plan
never transfers geometry, Body, numeric parameters, execution or prior L1.
"""

from datetime import datetime

from rosclaw.contracts.common import content_hash

from .contracts import RosEngineeringCaseV1
from .mechanism import LESSONS, MechanismExperienceV1
from .repair_compiler import compile_repair


def propose_tf_mechanism_transfer(record, case, model, requirements, staging):
    """Pure conditional candidate; original files and a new World remain required."""
    record = MechanismExperienceV1.from_dict(record.model_dump(mode="json"))
    case = RosEngineeringCaseV1.from_dict(case.model_dump(mode="json"))
    if (
        model.compute_snapshot_hash() != case.snapshot_hash
        or model.snapshot_hash != case.snapshot_hash
    ):
        raise ValueError("current mechanism target snapshot differs from original case")
    base = {
        "schema_version": "rosclaw.tf_mechanism_transfer.v1",
        "historical_experience_hash": record.experience_hash,
        "historical_source_urdf_sha256": record.source_urdf_sha256,
        "target_source_urdf_sha256": case.source_urdf_sha256,
        "target_case_hash": case.case_hash,
        "target_snapshot_hash": model.snapshot_hash,
        "cross_body_source": record.source_urdf_sha256 != case.source_urdf_sha256,
        "historical_parameters_reused": False,
        "body_geometry_transferred": False,
        "current_authority": "NONE",
        "current_l1": "NOT_EVALUATED",
        "new_World_verified": False,
        "causal_Memory_benefit": "NOT_ESTABLISHED",
        "historical_evidence_quality": "DECLARED_THREE_WORLD_REFERENCES_REQUIRE_HOST_ORIGINAL_SHA_CHECKS",
    }

    def finish(status, reason, **fields):
        payload = {**base, "status": status, "reason": reason, **fields}
        return {"transfer_plan_hash": content_hash("tfmechanismtransfer", payload), **payload}

    if record.kind != "enable_localization_tf":
        return finish("NOT_APPLICABLE", "only the TF broadcast concept supports this transfer path")
    if (
        record.mechanism != LESSONS[record.kind][0]
        or record.semantic_remedy != LESSONS[record.kind][1]
        or record.observation_signature != "F-TF-001:VIOLATED"
        or any(
            change["before"] is not False
            or change["after"] is not True
            or change["path"][-3:] != ["amcl", "ros__parameters", "tf_broadcast"]
            for change in record.original_changes
        )
    ):
        raise ValueError("canonical historical TF concept and original minimal changes required")
    violated = [c for c in case.constraints if c.id == "F-TF-001"]
    if len(violated) != 1 or violated[0].status != "VIOLATED":
        return finish(
            "NOT_APPLICABLE",
            "current TF finding is healthy or unresolved; do not reuse a historical diagnosis",
        )
    if case.source_status == "MISMATCH":
        return finish("NOT_APPLICABLE", "current source mismatch contraindicates transfer")
    candidate = case.derived_documents["navigation_observation_review"]["candidate"]
    geometry = candidate.get("geometry") or {}
    if (
        geometry.get("complete") is not True
        or geometry.get("source_urdf_sha256") != case.source_urdf_sha256
    ):
        return finish("NEEDS_READ_ONLY_PROBE", "complete current original URDF geometry required")
    node = requirements["lifecycle_nodes"]["localization"]
    now = datetime.fromisoformat(case.reviewed_at)

    def fresh(stamp):
        try:
            at = datetime.fromisoformat(stamp)
            return at.tzinfo is not None and 0 <= (now - at).total_seconds() * 1000 <= 500
        except (TypeError, ValueError):
            return False

    params = model.observations.get("node_parameters", {}).get(node, {})
    stamp = model.observations.get("parameter_captured_at", {}).get(node)
    if not fresh(stamp):
        return finish(
            "NEEDS_READ_ONLY_PROBE", "fresh current localization parameter readback required"
        )
    if params.get("tf_broadcast") is not False:
        return finish(
            "NOT_APPLICABLE",
            "current broadcast readback contradicts the historical disabled-broadcast concept",
        )
    lifecycle = [row for row in model.lifecycle if row.name == node]
    if len(lifecycle) != 1 or not fresh(lifecycle[0].captured_at.isoformat()):
        return finish(
            "NEEDS_READ_ONLY_PROBE", "unique fresh current localization lifecycle required"
        )
    if lifecycle[0].state != "ACTIVE":
        return finish(
            "NOT_APPLICABLE",
            "inactive localization is a competing cause; enabling broadcast alone is insufficient",
        )
    frames = candidate.get("frames", {})
    if not frames.get("map") or not frames.get("odom"):
        return finish("NEEDS_READ_ONLY_PROBE", "current map and odometry frame bindings required")
    if any(e.parent == frames["map"] and e.child == frames["odom"] for e in model.transforms):
        return finish(
            "NOT_APPLICABLE",
            "observed map-to-odom edge requires current authority/ownership review",
        )
    signature = {
        "mechanism": "disabled_localization_broadcast",
        "required_constraint": "F-TF-001:VIOLATED",
        "localization_state": "ACTIVE",
        "observed_broadcast": "FALSE",
        "target_edge": "map_role_to_odom_role",
    }
    proposal = compile_repair(
        case,
        model,
        requirements,
        staging,
        {"kind": "enable_localization_tf"},
        minimum_radius_m=geometry["physical_radius_m"],
    )
    return finish(
        "CONDITIONAL_CURRENT_TF_TRANSFER_PROPOSAL",
        "abstract historical concept matched; current compiler rederived the current minimal change",
        structural_signature=signature,
        structural_signature_hash=content_hash("tfmechanismsignature", signature),
        current_bindings={
            "localization_node": node,
            "map_frame": frames["map"],
            "odom_frame": frames["odom"],
        },
        proposal=proposal,
        proof_obligations=[
            "verify_historical_original_SHA_and_exclude_current_episode_runs",
            "independent_new_Body_full_fresh_L1_and_canonical_L2",
            "independent_stop_zero_contacts_source_continuity_and_unchanged_limits",
            "wrong_current_cause_counterexample_must_reject",
        ],
    )
