"""Mission evidence calculations plus canonical-daemon attestation gate."""

from rosclaw.contracts.common import content_hash

from .coverage import CleaningPose, CoverageVerifier


def verify_mission(evidence: dict, *, daemon=None, event_bus=None) -> dict:
    mission_id, body_id = evidence["mission_id"], evidence["body_id"]
    verifier = CoverageVerifier(**evidence["grid"])
    for sample in evidence["trajectory"]:
        verifier.observe(CleaningPose(**sample), frame_id=evidence["frame_id"])
    coverage = verifier.result()
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
    from rosclaw.connectors.ros.intelligence.evidence import emit_expert_evidence

    emit_expert_evidence(
        event_bus,
        "rosclaw.ros.verification.completed",
        {"robot_id": body_id, "mission_id": mission_id, "verification": result},
    )
    return result
