"""Fail-closed bridge from measured Growth evaluations to agentd learning.

The bridge stages knowledge; it never promotes or activates a controller.  A
passing simulation may become a HOW candidate for later human review, while a
failed simulation is retained only as negative MEMORY so the same mistake can
inform the next declared learning campaign.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.agentd.context.sources import EvidenceClass
from rosclaw.agentd.learning.pipeline import LearningPipeline
from rosclaw.feedback.contracts import canonical_hash

_SUPPORTED_SCHEMAS = frozenset(
    {
        "rosclaw.growth.g1_structured_recovery_evidence.v1",
        "rosclaw.growth.g1_residual_recovery_evidence.v1",
    }
)
_FAILURE_CURRICULUM_SCHEMAS = frozenset(
    {
        "rosclaw.simforge.g1_failure_curriculum_report.v1",
        "rosclaw.simforge.g1_failure_curriculum_report.v2",
        "rosclaw.simforge.g1_failure_curriculum_report.v3",
    }
)
_MAX_EVIDENCE_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class GrowthAgentdBridgeReceipt:
    candidate_id: str
    evidence_file_hash: str
    evidence_schema: str
    evaluation_status: str
    learning_kind: str
    disposition: str
    staged: bool
    activation_authorized: bool = False
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.growth.agentd_bridge_receipt.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stage_growth_evaluation_candidate(
    *,
    evaluation_path: Path,
    connection: sqlite3.Connection,
    source_checkout: Path,
    actor_id: str = "agent:rosclaw-growth-bridge",
) -> GrowthAgentdBridgeReceipt:
    """Stage one content-addressed evaluation without authorizing deployment."""

    path = evaluation_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if path == checkout or checkout in path.parents:
        raise ValueError("Growth evaluation evidence must be outside the source checkout")
    if not path.is_file():
        raise ValueError("Growth evaluation evidence must be a regular file")
    if path.stat().st_size > _MAX_EVIDENCE_BYTES:
        raise ValueError("Growth evaluation evidence is oversized")
    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Growth evaluation evidence is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Growth evaluation evidence must be a JSON object")
    schema = str(payload.get("schema_version", ""))
    if schema not in _SUPPORTED_SCHEMAS | _FAILURE_CURRICULUM_SCHEMAS:
        raise ValueError(f"unsupported Growth evaluation schema {schema!r}")
    if schema in _FAILURE_CURRICULUM_SCHEMAS:
        _validate_failure_curriculum_evidence(payload)
        evidence = _failure_curriculum_bridge_view(payload)
    else:
        _validate_safe_sim_evidence(payload)
        evidence = payload

    file_hash = f"sha256:{hashlib.sha256(raw).hexdigest()}"
    evidence_ref = f"growth-evaluation://{file_hash.removeprefix('sha256:')}"
    existing = connection.execute(
        "SELECT candidate_id, kind FROM learning_candidates "
        "WHERE evidence_refs_json = ? ORDER BY created_at LIMIT 1",
        (json.dumps([evidence_ref], ensure_ascii=False),),
    ).fetchone()
    passed = bool(evidence["passed"])
    kind = "HOW" if passed else "MEMORY"
    disposition = "PASSING_SIM_HOW_CANDIDATE" if passed else "NEGATIVE_SIM_MEMORY"
    if existing is not None:
        return GrowthAgentdBridgeReceipt(
            candidate_id=str(existing["candidate_id"]),
            evidence_file_hash=file_hash,
            evidence_schema=schema,
            evaluation_status=str(evidence["status"]),
            learning_kind=str(existing["kind"]),
            disposition=disposition,
            staged=False,
        )

    failed_cases = [
        {
            "case_id": str(case["spec"]["case_id"]),
            "partition": str(case["spec"]["partition"]),
            "reasons": _case_failure_reasons(case),
        }
        for case in evidence["cases"]
        if not bool(case["passed"])
    ]
    content = {
        "evaluation_schema": schema,
        "evaluation_status": evidence["status"],
        "evaluation_file_hash": file_hash,
        "candidate_hash": evidence["candidate_hash"],
        "request_hash": evidence["request_hash"],
        "environment_hash": evidence["environment_hash"],
        "implementation_hash": evidence["implementation_hash"],
        "evidence_domain": evidence["evidence_domain"],
        "body_scope": "unitree_g1_sim",
        "case_count": len(evidence["cases"]),
        "failed_cases": failed_cases,
        "disposition": disposition,
        "deployable": False,
        "activation_authorized": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
    title = (
        "G1 recovery policy passed frozen SIM gates"
        if passed
        else "G1 recovery policy negative SIM evidence"
    )
    pipeline = LearningPipeline(connection, actor_id=actor_id)
    candidate_id = pipeline.propose(
        kind=kind,
        title=title,
        content=content,
        evidence_class=EvidenceClass.MEASURED,
        evidence_refs=[evidence_ref],
        body_scope="unitree_g1_sim",
    )
    return GrowthAgentdBridgeReceipt(
        candidate_id=candidate_id,
        evidence_file_hash=file_hash,
        evidence_schema=schema,
        evaluation_status=str(evidence["status"]),
        learning_kind=kind,
        disposition=disposition,
        staged=True,
    )


def _validate_safe_sim_evidence(payload: dict[str, Any]) -> None:
    required_hashes = (
        "candidate_hash",
        "request_hash",
        "environment_hash",
        "implementation_hash",
    )
    for name in required_hashes:
        if not str(payload.get(name, "")).startswith("sha256:"):
            raise ValueError(f"Growth evaluation evidence requires {name}")
    if payload.get("evidence_domain") != "SIM":
        raise ValueError("Growth bridge accepts SIM evidence only")
    if payload.get("activation_ceiling") != "SIM_ONLY":
        raise ValueError("Growth bridge requires a SIM_ONLY activation ceiling")
    for name in (
        "activation_authorized",
        "promotion_authorized",
        "hardware_authorized",
        "hardware_command_sent",
    ):
        if payload.get(name) is not False:
            raise ValueError(f"Growth bridge requires {name}=false")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 8:
        raise ValueError("G1 Growth evaluation requires exactly eight committed cases")
    if any(
        not isinstance(case, dict)
        or not isinstance(case.get("spec"), dict)
        or not isinstance(case.get("passed"), bool)
        or case.get("parent_strict_replay") is not True
        or case.get("candidate_strict_replay") is not True
        for case in cases
    ):
        raise ValueError("every Growth case requires typed status and strict replay")
    passed = payload.get("passed")
    status = payload.get("status")
    if not isinstance(passed, bool):
        raise ValueError("Growth evaluation passed must be boolean")
    computed_passed = all(bool(case["passed"]) for case in cases)
    if payload.get("schema_version") == "rosclaw.growth.g1_residual_recovery_evidence.v1":
        development_gate = payload.get("development_aggregate_gate")
        if not isinstance(development_gate, dict) or not isinstance(
            development_gate.get("passed"), bool
        ):
            raise ValueError("residual Growth evidence requires a development aggregate gate")
        computed_passed = computed_passed and bool(development_gate["passed"])
    if passed != computed_passed:
        raise ValueError("Growth evaluation passed disagrees with committed gates")
    expected_status = "SIM_GATE_PASS" if passed else "REJECTED_BY_SIM_GATE"
    if status != expected_status:
        raise ValueError("Growth evaluation status disagrees with gate result")


def _validate_failure_curriculum_evidence(payload: dict[str, Any]) -> None:
    schema = str(payload.get("schema_version", ""))
    if payload.get("activation_ceiling") != "SIM_ONLY":
        raise ValueError("failure-curriculum bridge requires SIM_ONLY")
    if payload.get("evidence_domain") != "SHADOW":
        raise ValueError("failure-curriculum bridge requires SHADOW evidence")
    for name in (
        "body_hash",
        "kick_prior_hash",
        "frozen_policy_hash",
        "curriculum_commitment",
        "report_hash",
    ):
        if not str(payload.get(name, "")).startswith("sha256:"):
            raise ValueError(f"failure-curriculum evidence requires {name}")
    unsigned = dict(payload)
    claimed_report_hash = str(unsigned.pop("report_hash"))
    if canonical_hash(unsigned) != claimed_report_hash:
        raise ValueError("failure-curriculum report hash mismatch")
    validation = payload.get("validation")
    holdout = payload.get("holdout")
    if not isinstance(validation, list) or not validation or not isinstance(holdout, list) or not holdout:
        raise ValueError("failure-curriculum evidence requires sealed validation and holdout")
    sealed = [*validation, *holdout]
    if any(
        not isinstance(row, dict)
        or row.get("strict_replay") is not True
        or row.get("quality_accepted") is not True
        or not isinstance(row.get("critical"), bool)
        or not isinstance(row.get("success"), bool)
        for row in sealed
    ):
        raise ValueError("failure-curriculum sealed rows require typed gates and strict replay")
    if schema.endswith(".v3"):
        if any(not isinstance(row.get("abstained"), bool) for row in sealed):
            raise ValueError("v3 failure-curriculum rows require typed abstention")
        for row in sealed:
            result = row.get("result")
            if not isinstance(result, dict):
                raise ValueError("v3 failure-curriculum rows require results")
            if row["abstained"] and (
                result.get("status") != "ROBOT_NOT_STABLE"
                or result.get("physics_executed") is not False
                or row["success"] is not False
            ):
                raise ValueError("failure-curriculum abstention semantics are inconsistent")
            if not row["abstained"] and result.get("physics_executed") is not True:
                raise ValueError("failure-curriculum execution semantics are inconsistent")
        calibration = payload.get("calibration_contract")
        if not isinstance(calibration, dict) or any(
            calibration.get(name) is not expected
            for name, expected in (
                ("exact_hidden_dynamics_exposed", False),
                ("abstention_counts_as_success", False),
            )
        ):
            raise ValueError("failure-curriculum calibration isolation is invalid")
    elif payload.get("decision") == "SIM_CANDIDATE":
        raise ValueError("only v3 failure-curriculum evidence may stage a HOW candidate")
    critical = any(bool(row["critical"]) for row in sealed)
    success_rate = sum(bool(row["success"]) for row in validation) / len(validation)
    coverage = sum(not bool(row.get("abstained", False)) for row in validation) / len(
        validation
    )
    reasons: list[str] = []
    if critical:
        reasons.append("new_critical_failure")
    if success_rate < 0.5:
        reasons.append("validation_success_rate_below_50_percent")
    if schema.endswith(".v3") and coverage < 2.0 / 3.0:
        reasons.append("validation_execution_coverage_below_two_thirds")
    expected_decision = "SIM_CANDIDATE" if not reasons else "REJECTED"
    if payload.get("decision") != expected_decision or payload.get("gate_reasons") != reasons:
        raise ValueError("failure-curriculum decision disagrees with recomputed gates")
    learning = payload.get("learning_contract")
    if not isinstance(learning, dict) or any(
        learning.get(name) is not expected
        for name, expected in (
            ("failed_rollouts_train_actor", False),
            ("sealed_cases_changed_policy", False),
        )
    ):
        raise ValueError("failure-curriculum learning isolation is invalid")


def _failure_curriculum_bridge_view(payload: dict[str, Any]) -> dict[str, Any]:
    passed = payload["decision"] == "SIM_CANDIDATE"
    cases: list[dict[str, Any]] = []
    for row in [*payload["validation"], *payload["holdout"]]:
        case_passed = bool(
            row["strict_replay"]
            and row["quality_accepted"]
            and not row["critical"]
            and (row["success"] or row.get("abstained", False))
        )
        cases.append(
            {
                "spec": {
                    "case_id": str(row["case_id"]),
                    "partition": str(row["purpose"]),
                },
                "passed": case_passed,
                "absolute_gate": {
                    "reasons": []
                    if case_passed
                    else [str(row["result"]["status"]).lower()]
                },
            }
        )
    return {
        "passed": passed,
        "status": "SIM_GATE_PASS" if passed else "REJECTED_BY_SIM_GATE",
        "candidate_hash": payload["frozen_policy_hash"],
        "request_hash": payload["curriculum_commitment"],
        "environment_hash": payload["body_hash"],
        "implementation_hash": payload["report_hash"],
        "evidence_domain": payload["evidence_domain"],
        "cases": cases,
    }


def _case_failure_reasons(case: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for gate_name in ("non_regression_gate", "naturalness_gate", "absolute_gate"):
        gate = case.get(gate_name)
        if isinstance(gate, dict):
            value = gate.get("reasons")
            if isinstance(value, list):
                reasons.extend(str(reason) for reason in value)
    return sorted(set(reasons)) or ["case_gate_rejected"]


__all__ = ["GrowthAgentdBridgeReceipt", "stage_growth_evaluation_candidate"]
