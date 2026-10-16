"""Pure compound compilation; unit fixtures are not physical evaluation evidence."""

import copy
import json

import pytest

from benchmarks.ros_expert_ablation.aci_schema import decision_schema
from benchmarks.ros_expert_ablation.physical_aci import PhysicalACI
from benchmarks.ros_expert_ablation.task_tools import TaskTools
from rosclaw.connectors.ros.engineering import compile_engineering_case
from rosclaw.connectors.ros.engineering.repair_compiler import (
    compile_repair_sequence,
    proposal_staging,
)
from rosclaw.contracts.common import content_hash
from tests.benchmarks.test_physical_aci import fixture
from tests.connectors.ros.test_engineering_case_compiler import NOW


def compound():
    aci = fixture("tf")
    next(s for s in aci.model.signals if s.topic == "/sample/scan2").last_message_age_ms = 700
    aci.model.seal()
    files = copy.deepcopy(aci.tools.files)
    files["bringup.json"] = json.dumps({"scan_bridge_state": "paused_after_initialization"})
    tools = TaskTools(
        files, arm="B", reviewed_at=NOW.isoformat(), minimum_radius_m=aci.tools.minimum_radius_m
    )
    case = compile_engineering_case(
        aci.model,
        files["robot.urdf"].encode(),
        aci.requirements,
        source_ref="fixture:compound",
        now=NOW,
        evidence_domain="SEALED_REPLAY",
    )
    return PhysicalACI(case, aci.model, tools)


def compile_for(aci, intents):
    return compile_repair_sequence(
        aci.case,
        aci.model,
        aci.requirements,
        aci.tools.staging,
        intents,
        minimum_radius_m=aci.tools.minimum_radius_m,
    )


def test_two_repairs_original_baseline_and_exact_rollback():
    aci = compound()
    old = copy.deepcopy(aci.tools.staging)
    intents = [{"kind": "resume_scan_bridge"}, {"kind": "enable_localization_tf"}]
    p = compile_for(aci, intents)
    assert p == compile_for(aci, list(reversed(intents)))
    assert len(p["changes"]) == 2 and len(p["component_proposal_ids"]) == 2
    staged = proposal_staging(p, old)
    assert staged["bringup.json"]["scan_bridge_state"] == "running"
    assert staged["nav2.yaml"]["/sample"]["amcl"]["ros__parameters"]["tf_broadcast"] is True
    assert (
        aci.tools.staging == old
        and p["current_authority"] == "NONE"
        and p["admission"] == "NOT_ADMITTED"
    )
    restored = copy.deepcopy(staged)
    for row in p["rollback"]:
        parts = row["pointer"].split("/")[1:]
        parent = restored[row["file"]]
        for part in parts[:-1]:
            parent = parent[part.replace("~1", "/").replace("~0", "~")]
        parent[parts[-1]] = row["value"]
    assert restored == old
    p["changes"][0]["value"] = "tampered"
    with pytest.raises(ValueError, match="proposal/base changed"):
        proposal_staging(p, old)


@pytest.mark.parametrize(
    "intents",
    [
        [],
        [{"kind": "resume_scan_bridge"}],
        [{"kind": "resume_scan_bridge"}] * 2,
        [{"kind": "resume_scan_bridge"}, {"kind": "cover_urdf_envelope", "radius_m": 0.4}],
        [{"kind": "enable_localization_tf"}, {"kind": "arbitrary_shell"}],
    ],
)
def test_invalid_or_healthy_component_rejects_atomically(intents):
    aci = compound()
    old = copy.deepcopy(aci.tools.staging)
    with pytest.raises(ValueError):
        compile_for(aci, intents)
    assert aci.tools.staging == old and not aci.proposals


def test_common_typed_compound_facade_preserves_tasktools_policy():
    aci = compound()
    old = copy.deepcopy(aci.tools.staging)
    out = aci.call(
        {
            "operation": "ros_engineer.propose_compound_repair",
            "arguments": {
                "case_hash": aci.case.case_hash,
                "repairs": [{"kind": "enable_localization_tf"}, {"kind": "resume_scan_bridge"}],
            },
        }
    )
    assert (
        out["status"] == "TYPED_STAGING_PROPOSAL_NOT_EXECUTED"
        and len(out["proposal"]["changes"]) == 2
    )
    assert aci.tools.staging == old and aci.validation_count == 0
    assert "CompoundRepairCall" in decision_schema("A")["$defs"]


def test_changed_base_and_mismatched_source_stay_blocked():
    aci = compound()
    p = compile_for(aci, [{"kind": "resume_scan_bridge"}, {"kind": "enable_localization_tf"}])
    changed = copy.deepcopy(aci.tools.staging)
    changed["bringup.json"]["scan_bridge_state"] = "running"
    with pytest.raises(ValueError, match="proposal/base changed"):
        proposal_staging(p, changed)
    data = aci.case.model_dump(mode="json")
    data["source_status"] = "MISMATCH"
    data["case_hash"] = content_hash("roscase", {k: v for k, v in data.items() if k != "case_hash"})
    aci.case = type(aci.case).from_dict(data)
    with pytest.raises(ValueError, match="mismatched source"):
        compile_for(aci, [{"kind": "resume_scan_bridge"}, {"kind": "enable_localization_tf"}])
    # Independently reject a changed original snapshot, even for an otherwise valid case.
    aci = compound()
    aci.model.snapshot_hash = "rossnap_" + "f" * 64
    with pytest.raises(ValueError, match="snapshot mismatch"):
        compile_for(aci, [{"kind": "resume_scan_bridge"}, {"kind": "enable_localization_tf"}])


def test_expert_card_exposes_conditional_semantics_without_granting_admission():
    aci = compound()
    card = aci.call({"operation": "ros_engineer.inspect_task", "arguments": {}})["task"]
    issues = {row["id"]: row for row in card["constraints"]}
    assert issues["F-TF-001"]["invariant"] == "localization.map_to_odom"
    assert issues["F-TF-001"]["hypotheses"] == ["localization_tf_broadcast_disabled"]
    candidates = {row["semantic_field"]: row for row in card["conditional_candidates"]}
    tf = candidates["localization.tf_broadcast"]
    assert tf["proposed_value"] is True and tf["classification"] == "INFERRED_CANDIDATE"
    assert tf["applicable_now"] is False and tf["stage_only"] is True
    assert tf["preconditions"] and all(ref in aci.case.evidence_index for ref in tf["refs"])
    aci.tools.arm = "A"
    common = aci.call({"operation": "ros_engineer.inspect_task", "arguments": {}})["task"]
    assert "conditional_candidates" not in common and "constraints" not in common


def test_provider_phase_schema_binds_actual_case_and_issued_references():
    aci = compound()
    before = decision_schema(
        "B", case_hash=aci.case.case_hash, proposal_ids=[], verification_refs=[]
    )
    assert "ValidateCall" not in before["$defs"] and "ExplainCall" not in before["$defs"]
    assert before["$defs"]["RepairArgs"]["properties"]["case_hash"]["enum"] == [aci.case.case_hash]
    after = decision_schema(
        "C",
        memory_available=True,
        case_hash=aci.case.case_hash,
        proposal_ids=["actual-proposal"],
        verification_refs=["actual-proof"],
        memory_refs=["actual-memory"],
    )
    assert after["$defs"]["ProposalArgs"]["properties"]["proposal_id"]["enum"] == [
        "actual-proposal"
    ]
    assert after["$defs"]["ResultArgs"]["properties"]["verification_ref"]["enum"] == [
        "actual-proof"
    ]
    assert after["$defs"]["MemoryRepairArgs"]["properties"]["memory_ref"]["enum"] == [
        "actual-memory"
    ]
    assert after["properties"]["memory_refs_used"]["items"]["enum"] == ["actual-memory"]
    with pytest.raises(ValueError, match="case content hash"):
        decision_schema("B", case_hash="incorrectly-copied-hash")
    with pytest.raises(ValueError, match="distinct actual"):
        decision_schema("B", proposal_ids=["duplicate", "duplicate"])
