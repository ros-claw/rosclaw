"""Deterministic conditional staging plans, never execution or physical admission.

Reuse existing options and repair compilers. Unsupported observed violations cause
abstention rather than a partial guessed fix. No model, files, ROS or Runtime I/O.
"""

from __future__ import annotations

import copy
import hashlib
import math
from datetime import datetime

from rosclaw.connectors.ros.context.navigation_review import review_navigation_candidate
from rosclaw.contracts.common import content_hash

from .contracts import RosEngineeringCaseV1
from .repair_compiler import compile_repair, compile_repair_sequence
from .staging_options import compile_staging_options


def plan_staging_repairs(case, model, urdf_bytes, requirements, staging, *, ownership_reviews=None):
    """Plan supported original-input hypotheses; retain all future proof obligations."""
    case = RosEngineeringCaseV1.from_dict(case.model_dump(mode="json"))
    if (
        model.snapshot_hash != case.snapshot_hash
        or model.compute_snapshot_hash() != case.snapshot_hash
    ):
        raise ValueError("plan requires the unchanged original case/model snapshot")
    if (
        type(urdf_bytes) is not bytes
        or hashlib.sha256(urdf_bytes).hexdigest() != case.source_urdf_sha256
    ):
        raise ValueError("plan requires the original case-bound URDF bytes")
    original_review = case.derived_documents["navigation_observation_review"]
    if content_hash("navrequirements", requirements) != original_review["requirements_hash"]:
        raise ValueError("plan requires the original case-bound requirements")
    if (
        type(staging) is not dict
        or set(staging) != {"nav2.yaml", "bringup.json"}
        or any(type(value) is not dict for value in staging.values())
    ):
        raise ValueError("two original staging object inputs required")
    baseline = copy.deepcopy(staging)
    result = {
        "schema_version": "rosclaw.deterministic_staging_plan.v1",
        "case_hash": case.case_hash,
        "snapshot_hash": case.snapshot_hash,
        "source_urdf_sha256": case.source_urdf_sha256,
        "base_staging_hash": content_hash("acistaging", baseline),
        "scope": "ORIGINAL_INPUT_CONDITIONAL_PLAN_NOT_CURRENT_STATE_OR_EXECUTION",
        "decision_owner": "DETERMINISTIC_EXISTING_COMPILERS",
        "current_authority": "NONE",
        "current_l1": "NOT_EVALUATED",
        "admission": "NOT_ADMITTED",
        "unknown_constraints": [c.id for c in case.constraints if c.status == "UNKNOWN"],
        "proposal": None,
        "reasons": [],
    }

    def finish(status, reasons):
        result.update(status=status, reasons=reasons)
        return copy.deepcopy({"plan_hash": content_hash("aciplan", result), **result})

    if case.source_status == "MISMATCH":
        return finish("ABSTAIN_SOURCE_MISMATCH", ["original source/request mismatch"])
    # Recompute geometry from original bytes/model at the explicit case clock;
    # do not let a supplied minimum radius or edited derived geometry set the floor.
    review = review_navigation_candidate(
        model, urdf_bytes, requirements, now=datetime.fromisoformat(case.reviewed_at)
    )
    candidate = review["candidate"]
    geometry = candidate.get("geometry") or {}
    radius = geometry.get("physical_radius_m")
    if (
        len(candidate.get("urdf_provenance", [])) != 1
        or geometry.get("complete") is not True
        or type(radius) not in (int, float)
        or not math.isfinite(radius)
        or radius <= 0
    ):
        return finish(
            "NEEDS_ORIGINAL_BODY_EVIDENCE",
            [
                "unique original URDF provenance and complete conservative geometry required; no current Body/L1 claim"
            ],
        )
    violations = {
        c.id for c in case.constraints if c.status == "VIOLATED" and c.id != "F-SOURCE-001"
    }
    if not violations:
        return finish(
            "NO_OBSERVED_REPAIR_WITH_UNKNOWN_CONSTRAINTS"
            if result["unknown_constraints"]
            else "NO_OBSERVED_REPAIR",
            ["no observed violated repair constraint; unknowns are not guessed"],
        )
    options = compile_staging_options(
        case,
        model,
        requirements,
        baseline,
        minimum_radius_m=radius,
        ownership_reviews=ownership_reviews,
    )
    if options["blocked"] or {option.issue_id for option in options["options"]} != violations:
        return finish(
            "NEEDS_OPERATOR_OR_UNSUPPORTED_MECHANISM",
            options["blocked"]
            or ["not every observed violation has a supported original-input repair"],
        )
    intents = [option.intent.model_dump(mode="json") for option in options["options"]]
    compiler = compile_repair if len(intents) == 1 else compile_repair_sequence
    proposal = compiler(
        case,
        model,
        requirements,
        baseline,
        intents[0] if len(intents) == 1 else intents,
        minimum_radius_m=radius,
        ownership_reviews=ownership_reviews,
    )
    result["proposal"] = proposal
    result["covered_violation_ids"] = sorted(violations)
    return finish(
        "CONDITIONAL_STAGING_PROPOSAL",
        [
            "unexecuted hypothesis only; original candidate checks and all fresh SDK proof obligations remain pending"
        ],
    )


def parse_staging_documents(nav2_bytes, bringup_bytes):
    """Parse bounded concrete originals; reject YAML aliases before expansion."""
    import json

    import yaml

    if any(
        type(raw) is not bytes or not 0 < len(raw) <= 5_000_000
        for raw in (nav2_bytes, bringup_bytes)
    ):
        raise ValueError("bounded nonempty staging document bytes required")
    try:
        if any(
            isinstance(token, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken))
            for token in yaml.scan(nav2_bytes)
        ):
            raise ValueError("concrete staging YAML required; anchors and aliases are unsupported")
        nav2 = yaml.safe_load(nav2_bytes)
        bringup = json.loads(bringup_bytes)
    except (yaml.YAMLError, UnicodeError, RecursionError) as error:
        raise ValueError("bounded concrete staging documents required") from error
    if type(nav2) is not dict or type(bringup) is not dict:
        raise ValueError("Nav2 YAML and bringup JSON must contain objects")
    return {"nav2.yaml": nav2, "bringup.json": bringup}
