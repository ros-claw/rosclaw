"""Mission evidence calculations plus canonical-daemon attestation gate."""

from rosclaw.contracts.common import content_hash

from .brush_timeline import validate_brush_pair
from .coverage import CleaningPose, CoverageVerifier
from .occupancy import OccupancyAccounting


def replay_coverage(evidence):
    """Replay every paired snapshot, never a final occupancy mask over history."""
    verifier = CoverageVerifier(**evidence["grid"])
    schema = evidence.get("schema_version")
    dynamic = schema == "rosclaw.time_paired_mission_evidence.v1"
    if not dynamic and any(k in evidence for k in ("occupancy_binding", "occupancy_samples")):
        raise ValueError("dynamic evidence requires its explicit versioned schema")
    if schema not in (None, "rosclaw.time_paired_mission_evidence.v1"):
        raise ValueError("unsupported mission evidence schema")
    if any(k in evidence for k in ("brush_evidence_binding", "brush_evidence_samples")):
        binding = evidence["brush_evidence_binding"]
        brush_samples = evidence["brush_evidence_samples"]
        if type(binding) is not dict or set(binding) != {
            "run_id",
            "body_snapshot_hash",
            "attachment_hash",
            "producer_id",
        }:
            raise ValueError("frozen brush evidence binding required")
        if (
            type(brush_samples) is not list
            or not brush_samples
            or len(brush_samples) != len(evidence["trajectory"])
        ):
            raise ValueError("one brush pair per trajectory pose required")
        previous = None
        for pose, proof in zip(evidence["trajectory"], brush_samples, strict=True):
            if type(proof) is not dict or set(proof) != {
                "brush_state_pair",
                "brush_source_binding",
                "brush_source_fault",
            }:
                raise ValueError("brush proof cannot replace trajectory fields")
            previous = validate_brush_pair({**pose, **proof}, binding, previous_chain=previous)
    if not dynamic:
        for sample in evidence["trajectory"]:
            verifier.observe(CleaningPose(**sample), frame_id=evidence["frame_id"])
        return verifier.result(), None
    binding = evidence["occupancy_binding"]
    if set(binding) != {"run_id", "geometry_hash"} or evidence["frame_id"] != verifier.frame_id:
        raise ValueError("frozen dynamic binding and frame required")
    accounting = OccupancyAccounting(verifier, mission_id=evidence["mission_id"], **binding)
    samples = evidence["occupancy_samples"]
    if type(samples) is not list or not samples or len(samples) != len(evidence["trajectory"]):
        raise ValueError("one occupancy snapshot per trajectory pose required")
    for pose, occupancy in zip(evidence["trajectory"], samples, strict=True):
        if type(occupancy) is not dict or set(occupancy) != {"occupancy", "occupancy_hash"}:
            raise ValueError("occupancy record cannot replace trajectory or safety fields")
        accounting.observe_sample(
            {**pose, **occupancy, "observation_complete": True, "collision_count": 0}
        )
    result = accounting.result()
    return result["coverage"], result


def verify_mission(evidence: dict, *, daemon=None, event_bus=None) -> dict:
    mission_id, body_id = evidence["mission_id"], evidence["body_id"]
    coverage, temporal = replay_coverage(evidence)
    collision = evidence.get("collision", {})
    calculation_pass = (
        coverage["coverage_ratio"] >= 0.98
        and coverage["trace_gaps"] == 0
        and (
            collision.get("collision_count") == 0 and collision.get("observation_complete") is True
        )
    )
    evidence_hash = content_hash("rosmissionevidence", evidence)
    receipts, missing = [], []
    domains: set[str] = set()
    action_ids = evidence.get("action_ids", [])
    if not action_ids or daemon is None:
        missing.append("canonical execution receipts unavailable")
    else:
        for action_id in action_ids:
            response = daemon.get_execution_receipt(action_id)
            receipt = response.get("receipt", {})
            expected_domain = {"SIMULATION": "SIMULATION", "REAL": "HARDWARE"}.get(
                receipt.get("mode")
            )
            if expected_domain:
                domains.add(expected_domain)
            verification = receipt.get("verification_result") or {}
            # Supplied trajectory and collision claims are not measurements until
            # an independent daemon executor has bound their exact bytes.
            bound = evidence_hash in verification.get("independent_evidence_hashes", [])
            if (
                receipt.get("action_id") != action_id
                or receipt.get("body_id") != body_id
                or receipt.get("final_state") != "COMPLETED"
                or receipt.get("mode") not in {"SIMULATION", "REAL"}
                or receipt.get("evidence_domain") != expected_domain
                or receipt.get("evidence_level") not in {"TASK_VERIFIED", "PHYSICALLY_OBSERVED"}
                or verification.get("mission_id") != mission_id
                or not bound
            ):
                missing.append(f"receipt {action_id} lacks independent mission evidence binding")
            receipts.append(receipt)
        if len(domains) > 1:
            missing.append("mixed simulation/hardware receipts cannot establish one mission")
    status = "NOT_VERIFIED" if missing else "PASS" if calculation_pass else "FAIL"
    result = {
        "schema_version": "rosclaw.mission_verification.v1",
        "mission_id": mission_id,
        "body_id": body_id,
        "coverage": coverage,
        "safety": collision,
        "calculation_pass": calculation_pass,
        "verification_status": status,
        "evidence_hash": evidence_hash,
        "execution_receipts": receipts,
        "missing_evidence": missing,
        "success": status == "PASS",
        "evidence_domain": next(iter(domains))
        if len(domains) == 1 and not missing
        else "SUPPLIED_DATA",
        "hardware_verified": status == "PASS" and domains == {"HARDWARE"},
        "usable_for_real_execution": False,
    }
    if temporal is not None:
        result["time_paired_accounting"] = temporal
    from rosclaw.connectors.ros.intelligence.evidence import emit_expert_evidence

    emit_expert_evidence(
        event_bus,
        "rosclaw.ros.verification.completed",
        {"robot_id": body_id, "mission_id": mission_id, "verification": result},
    )
    return result
