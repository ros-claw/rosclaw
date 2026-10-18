"""New RK06 frozen D/A/B/C/E pilot allocation and fail-closed entry gate.

Pure original-file review only. This module never launches a model, ROS, World,
Runtime or action, and never modifies the frozen G5 protocol or experiment rows.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rosclaw.contracts.common import content_hash

ARMS = ("D", "A", "B", "C", "E")
STRATA = ("known_mechanism", "new_root_cause", "multi_node_chain", "new_body_task", "physical")
GATES = {
    "source_main_delivery": "PASS_RK05_EXACT_SOURCE_MAIN_DELIVERY",
    "frozen_core_new_body": "PASS_NEW_BODY_NAVIGATION_AFTER_UNCHANGED_CORE_FREEZE",
    "memory_transfer_counterexample": "PASS_CROSS_BODY_MEMORY_TRANSFER_AND_WRONG_CAUSE_COUNTEREXAMPLE",
    "post_freeze_case_registry": "PASS_NEW_RSK_CASES_FROZEN_AFTER_HARNESS",
}


class Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PilotCaseV1(Closed):
    case_id: str = Field(pattern=r"^[a-z0-9_]{1,64}$")
    stratum: Literal[
        "known_mechanism", "new_root_cause", "multi_node_chain", "new_body_task", "physical"
    ]
    facts_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_episode_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RskPilotProtocolV1(Closed):
    schema_version: Literal["rosclaw.rsk_pilot_protocol.v1"] = "rosclaw.rsk_pilot_protocol.v1"
    harness_source: str = Field(pattern=r"^[0-9a-f]{40}$")
    benchmark_code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_identity: str = Field(min_length=1, max_length=128)
    provider_settings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    memory_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: list[PilotCaseV1] = Field(min_length=5, max_length=10)
    seeds: list[int] = Field(min_length=1, max_length=3)
    allocation_seed: int = Field(ge=0)
    max_model_decisions: int = Field(ge=3, le=12)
    max_read_only_probes: int = Field(ge=1, le=8)
    max_new_worlds_per_episode: Literal[1] = 1
    max_goals_per_world: Literal[2] = 2
    first_known_valid_proposal_decisions: Literal[3] = 3
    identical_original_fact_access: Literal[True] = True
    action_path: Literal["MCP_REQUEST_ACTION_ROSCLAWD_ONLY"] = "MCP_REQUEST_ACTION_ROSCLAWD_ONLY"
    baseline_claim: Literal["SAME_MODEL_TOOLS_NOT_CODEX_PRODUCT"] = (
        "SAME_MODEL_TOOLS_NOT_CODEX_PRODUCT"
    )
    old_g5: Literal["FROZEN_UNCHANGED"] = "FROZEN_UNCHANGED"

    @model_validator(mode="after")
    def complete_split(self):
        if (
            {case.stratum for case in self.cases} != set(STRATA)
            or len({case.case_id for case in self.cases}) != len(self.cases)
            or len({case.evaluation_episode_sha256 for case in self.cases}) != len(self.cases)
            or len(set(self.seeds)) != len(self.seeds)
            or any(seed < 0 for seed in self.seeds)
        ):
            raise ValueError("five strata and distinct evaluation cases/episodes/seeds required")
        return self

    @property
    def protocol_hash(self):
        return content_hash("rskpilotprotocol", self.model_dump(mode="json"))


def allocate_pilot(protocol: RskPilotProtocolV1) -> list[dict]:
    rng = random.Random(protocol.allocation_seed)
    pairs = [(case, seed) for case in protocol.cases for seed in protocol.seeds]
    rng.shuffle(pairs)
    result = []
    for pair, (case, seed) in enumerate(pairs, 1):
        arms = list(ARMS)
        rng.shuffle(arms)
        for arm in arms:
            result.append(
                {
                    "ordinal": len(result) + 1,
                    "pair": pair,
                    "case_id": case.case_id,
                    "stratum": case.stratum,
                    "seed": seed,
                    "arm": arm,
                    "facts_manifest_sha256": case.facts_manifest_sha256,
                    "model_identity": None if arm == "D" else protocol.model_identity,
                    "provider_settings_sha256": None
                    if arm == "D"
                    else protocol.provider_settings_sha256,
                    "typed_ACI": arm in {"B", "C", "E"},
                    "active_hypotheses_and_probes": arm in {"C", "E"},
                    "mechanism_Memory": arm == "E",
                    "execution_authority": "NONE",
                }
            )
    return result


def review_pilot_entry(protocol: RskPilotProtocolV1, evidence: dict, *, root: Path) -> dict:
    """Host SHA references gate eligibility; absence/corruption cannot become PASS."""
    if set(evidence) != set(GATES):
        raise ValueError("all four named gate slots required, absent references must be null")
    root = Path(root).resolve(strict=True)
    reasons, checks = [], []
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != protocol.benchmark_code_sha256:
        reasons.append("benchmark implementation differs from frozen protocol SHA")
    for name, expected_status in GATES.items():
        ref = evidence[name]
        check = {"gate": name, "passed": False}
        try:
            if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
                raise ValueError("original SHA reference missing")
            path = (root / ref["path"]).resolve(strict=True)
            path.relative_to(root)
            if not path.is_file() or not 0 < path.stat().st_size <= 1_000_000:
                raise ValueError("bounded original gate review required")
            raw = path.read_bytes()
            if len(raw) > 1_000_000 or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
                raise ValueError("original gate review SHA mismatch")
            review = json.loads(raw)
            if review.get("status") != expected_status:
                raise ValueError("required independent acceptance status missing")
            if review.get("harness_source") != protocol.harness_source:
                raise ValueError("gate does not bind frozen harness source")
            if review.get("protocol_hash") != protocol.protocol_hash:
                raise ValueError("gate does not bind this frozen pilot split")
            check.update(passed=True, original_ref=ref)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            check["reason"] = str(exc)
            reasons.append(name + ": " + str(exc))
        checks.append(check)
    payload = {
        "schema_version": "rosclaw.rsk_pilot_entry.v1",
        "status": "ELIGIBLE_FOR_BOUNDED_PILOT_ONLY" if not reasons else "NO_GO",
        "protocol_hash": protocol.protocol_hash,
        "harness_source": protocol.harness_source,
        "planned_allocation": allocate_pilot(protocol),
        "gate_checks": checks,
        "reasons": reasons,
        "model_calls": 0,
        "new_Worlds": 0,
        "causal_Memory_benefit": "NOT_ESTABLISHED",
        "execution_authority": "NONE",
        "formal_study_authorized_by_this_review": False,
    }
    return {"review_hash": content_hash("rskpilotentry", payload), **payload}


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def bounded(path):
        if not path.is_file() or not 0 < path.stat().st_size <= 1_000_000:
            raise ValueError("bounded original protocol/evidence file required")
        raw = path.read_bytes()
        if len(raw) > 1_000_000:
            raise ValueError("protocol/evidence changed size during read")
        return json.loads(raw)

    protocol = RskPilotProtocolV1.model_validate(bounded(args.protocol))
    review = review_pilot_entry(protocol, bounded(args.evidence), root=args.root)
    # Allocation is a plan only, not executed episodes or a source of authority.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(review, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": review["status"], "model_calls": 0, "new_Worlds": 0}))


if __name__ == "__main__":
    main()
