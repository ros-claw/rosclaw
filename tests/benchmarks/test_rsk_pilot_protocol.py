"""New pilot splits stay fair and cannot pass missing or corrupt entry evidence."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pytest
from pydantic import ValidationError

import benchmarks.ros_expert_ablation.rsk_pilot_protocol as implementation
from benchmarks.ros_expert_ablation.rsk_pilot_protocol import (
    ARMS,
    GATES,
    STRATA,
    RskPilotProtocolV1,
    allocate_pilot,
    review_pilot_entry,
)


def protocol():
    return RskPilotProtocolV1(
        harness_source="a" * 40,
        benchmark_code_sha256=hashlib.sha256(
            Path(implementation.__file__).read_bytes()
        ).hexdigest(),
        model_identity="fixture:same-frozen-model",
        provider_settings_sha256="b" * 64,
        memory_policy_sha256="c" * 64,
        cases=[
            {
                "case_id": f"case_{i}",
                "stratum": stratum,
                "facts_manifest_sha256": str(i + 1) * 64,
                "evaluation_episode_sha256": str(i + 1) * 64,
            }
            for i, stratum in enumerate(STRATA)
        ],
        seeds=[101],
        allocation_seed=1018,
        max_model_decisions=12,
        max_read_only_probes=8,
    )


def evidence(tmp_path, frozen):
    refs = {}
    for name, status in GATES.items():
        p = tmp_path / (name + ".json")
        p.write_text(
            json.dumps(
                {
                    "status": status,
                    "harness_source": frozen.harness_source,
                    "protocol_hash": frozen.protocol_hash,
                }
            )
        )
        refs[name] = {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    return refs


def test_deterministic_pairs_share_original_facts_and_same_model_settings():
    frozen = protocol()
    rows = allocate_pilot(frozen)
    assert rows == allocate_pilot(frozen) and len(rows) == 25
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["pair"]].append(row)
    for values in pairs.values():
        assert {v["arm"] for v in values} == set(ARMS)
        assert len({v["facts_manifest_sha256"] for v in values}) == 1
        assert len({v["model_identity"] for v in values if v["arm"] != "D"}) == 1
        assert next(v for v in values if v["arm"] == "D")["model_identity"] is None
        assert all(v["execution_authority"] == "NONE" for v in values)
        assert sum(v["mechanism_Memory"] for v in values) == 1


def test_missing_evidence_stops_before_any_calls_or_worlds(tmp_path):
    frozen = protocol()
    review = review_pilot_entry(frozen, dict.fromkeys(GATES), root=tmp_path)
    assert review["status"] == "NO_GO" and len(review["reasons"]) == 4
    assert review["model_calls"] == review["new_Worlds"] == 0
    assert review["formal_study_authorized_by_this_review"] is False


@pytest.mark.parametrize("fault", ["sha", "source", "split", "status", "outside_root"])
def test_original_review_corruption_or_wrong_split_cannot_open_gate(tmp_path, fault):
    frozen = protocol()
    refs = evidence(tmp_path, frozen)
    name = next(iter(GATES))
    p = tmp_path / refs[name]["path"]
    if fault == "sha":
        refs[name]["sha256"] = "0" * 64
    elif fault == "outside_root":
        refs[name]["path"] = "../outside.json"
    else:
        value = json.loads(p.read_text())
        value[{"source": "harness_source", "split": "protocol_hash", "status": "status"}[fault]] = (
            "wrong"
        )
        p.write_text(json.dumps(value))
        refs[name]["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    review = review_pilot_entry(frozen, refs, root=tmp_path)
    assert review["status"] == "NO_GO" and len(review["reasons"]) == 1


def test_all_synthetic_entry_reviews_only_allow_bounded_pilot_not_study(tmp_path):
    frozen = protocol()
    review = review_pilot_entry(frozen, evidence(tmp_path, frozen), root=tmp_path)
    assert review["status"] == "ELIGIBLE_FOR_BOUNDED_PILOT_ONLY"
    assert review["formal_study_authorized_by_this_review"] is False
    assert review["causal_Memory_benefit"] == "NOT_ESTABLISHED"


def test_incomplete_strata_and_relaxed_budgets_reject():
    value = protocol().model_dump(mode="json")
    for changed in [
        {"max_model_decisions": 13},
        {"max_new_worlds_per_episode": 2},
        {"identical_original_fact_access": False},
        {"cases": value["cases"][:-1]},
    ]:
        with pytest.raises(ValidationError):
            RskPilotProtocolV1.model_validate({**value, **changed})


def test_review_hash_covers_all_returned_fields_including_allocation(tmp_path):
    frozen = protocol()
    review = review_pilot_entry(frozen, evidence(tmp_path, frozen), root=tmp_path)
    digest = review.pop("review_hash")
    from rosclaw.contracts.common import content_hash

    assert digest == content_hash("rskpilotentry", review)
    assert len(review["planned_allocation"]) == 25


def test_changed_benchmark_code_pin_rejects_otherwise_valid_gate_reviews(tmp_path):
    value = protocol().model_dump(mode="json")
    value["benchmark_code_sha256"] = "0" * 64
    frozen = RskPilotProtocolV1.model_validate(value)
    assert (
        review_pilot_entry(frozen, evidence(tmp_path, frozen), root=tmp_path)["status"] == "NO_GO"
    )


def test_published_schema_matches_closed_protocol():
    schema = (
        Path(__file__).resolve().parents[2]
        / "docs/reports/ros-expert-harness/RSK/RskPilotProtocolV1.schema.json"
    )
    assert json.loads(schema.read_text()) == RskPilotProtocolV1.model_json_schema()
