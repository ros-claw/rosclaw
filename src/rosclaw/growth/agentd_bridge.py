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

_SUPPORTED_SCHEMAS = frozenset(
    {
        "rosclaw.growth.g1_structured_recovery_evidence.v1",
        "rosclaw.growth.g1_residual_recovery_evidence.v1",
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
    if schema not in _SUPPORTED_SCHEMAS:
        raise ValueError(f"unsupported Growth evaluation schema {schema!r}")
    _validate_safe_sim_evidence(payload)

    file_hash = f"sha256:{hashlib.sha256(raw).hexdigest()}"
    evidence_ref = f"growth-evaluation://{file_hash.removeprefix('sha256:')}"
    existing = connection.execute(
        "SELECT candidate_id, kind FROM learning_candidates "
        "WHERE evidence_refs_json = ? ORDER BY created_at LIMIT 1",
        (json.dumps([evidence_ref], ensure_ascii=False),),
    ).fetchone()
    passed = bool(payload["passed"])
    kind = "HOW" if passed else "MEMORY"
    disposition = "PASSING_SIM_HOW_CANDIDATE" if passed else "NEGATIVE_SIM_MEMORY"
    if existing is not None:
        return GrowthAgentdBridgeReceipt(
            candidate_id=str(existing["candidate_id"]),
            evidence_file_hash=file_hash,
            evidence_schema=schema,
            evaluation_status=str(payload["status"]),
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
        for case in payload["cases"]
        if not bool(case["passed"])
    ]
    content = {
        "evaluation_schema": schema,
        "evaluation_status": payload["status"],
        "evaluation_file_hash": file_hash,
        "candidate_hash": payload["candidate_hash"],
        "request_hash": payload["request_hash"],
        "environment_hash": payload["environment_hash"],
        "implementation_hash": payload["implementation_hash"],
        "evidence_domain": payload["evidence_domain"],
        "body_scope": "unitree_g1_sim",
        "case_count": len(payload["cases"]),
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
        evaluation_status=str(payload["status"]),
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
