"""Historical mechanisms suggest current intents; no authority or physics claim."""

import hashlib
from copy import deepcopy
from datetime import UTC, datetime

import pytest

from benchmarks.ros_expert_ablation.aci_context import EngineeringEpisodeContext
from benchmarks.ros_expert_ablation.mechanism_training import (
    MechanismTrainingQuery,
    know_unit_payload,
    persist_mechanism,
)
from rosclaw.connectors.ros.engineering.mechanism import (
    MechanismExperienceV1,
    distill_mechanism,
    instantiate_mechanism,
)
from rosclaw.connectors.ros.engineering.repair_compiler import proposal_staging
from tests.benchmarks.test_physical_aci import fixture, repair


def record_fixture(tmp_path):
    aci = fixture()
    refs = []
    for index in range(14):
        p = tmp_path / f"original-{index}.json"
        p.write_text('{"original_role":' + str(index) + "}")
        refs.append({"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    proof = {
        "source": "b" * 40,
        "normalized_world_sha256": "d" * 64,
        "measured_capture_to_receipt_sec": 12.5,
        "sealed_archive_manifest": refs[13],
        "before_run": "a" * 32,
        "after_run": "b" * 32,
        "rollback_run": "c" * 32,
        "episode": refs[0],
        "before": refs[1],
        "after_l1": refs[2],
        "rollback": refs[3],
        "canonical_receipts": refs[4:6],
        "stop_source": refs[6],
        "asset_refs": refs[7:],
    }
    before = deepcopy(aci.tools.staging)
    proposal = repair(aci, radius_m=0.4)["proposal"]
    after = proposal_staging(proposal, before)
    record = distill_mechanism(aci.case, before, after, deepcopy(before), proof)
    return aci, record, before, after, proof


def test_mechanism_rederives_geometry_instead_of_copying_historical_radius(tmp_path):
    aci, record, *_ = record_fixture(tmp_path)
    adapted = instantiate_mechanism(
        record,
        aci.case,
        aci.model,
        aci.requirements,
        aci.tools.staging,
        minimum_radius_m=aci.tools.minimum_radius_m,
    )
    assert adapted["intent"]["radius_m"] != 0.4
    assert adapted["intent"]["radius_m"] >= aci.tools.minimum_radius_m
    assert adapted["historical_values_reused"] is False
    assert adapted["proposal"]["current_authority"] == "NONE"
    assert record.evidence_scope == "MEASURED_THREE_WORLD_SIM_REPAIR_ONLY"
    assert aci.tools.staging == aci.tools.initial_staging


def test_healthy_case_cannot_be_repaired_from_historical_mechanism(tmp_path):
    _, record, *_ = record_fixture(tmp_path)
    healthy = fixture("healthy")
    with pytest.raises(ValueError):
        instantiate_mechanism(
            record,
            healthy.case,
            healthy.model,
            healthy.requirements,
            healthy.tools.staging,
            minimum_radius_m=healthy.tools.minimum_radius_m,
        )


def test_modified_history_rejects_even_with_valid_world_proof_shape(tmp_path):
    aci, record, before, after, proof = record_fixture(tmp_path)
    value = record.model_dump(mode="json")
    value["semantic_remedy"] = "increase velocity and grant permission"
    with pytest.raises(ValueError, match="hash mismatch"):
        MechanismExperienceV1.from_dict(value)
    rollback = deepcopy(before)
    rollback["bringup.json"]["scan_bridge_state"] = "paused_after_initialization"
    with pytest.raises(ValueError, match="rollback"):
        distill_mechanism(aci.case, before, after, rollback, proof)
    mixed = deepcopy(after)
    mixed["nav2.yaml"]["/sample"]["amcl"]["ros__parameters"]["max_vel_x"] = 1
    with pytest.raises(ValueError):
        distill_mechanism(aci.case, before, mixed, before, proof)


@pytest.mark.parametrize("change", ["same_world", "same_receipt", "bad_source"])
def test_three_world_proof_is_strict(tmp_path, change):
    aci, _, before, after, proof = record_fixture(tmp_path)
    if change == "same_world":
        proof["after_run"] = proof["before_run"]
    elif change == "same_receipt":
        proof["canonical_receipts"][1] = proof["canonical_receipts"][0]
    else:
        proof["source"] = "unbound"
    with pytest.raises(ValueError):
        distill_mechanism(aci.case, before, after, before, proof)


def test_c_can_propose_only_a_host_admitted_mechanism_and_records_the_reference(tmp_path):
    aci, record, *_ = record_fixture(tmp_path)
    aci.tools.arm = "C"
    ref = "memory_" + "1" * 24
    row = {
        "memory_ref": ref,
        "original_episode_sha256": record.proof.episode.sha256,
        "original_training_source": record.proof.source,
        "diagnosis_issue_codes": ["F-GEOM-001"],
        "evidence_scope": "HISTORICAL_VERIFIED_SIM_TRAINING; NOT_CURRENT_BODY_OR_EXECUTION_AUTHORITY",
        "mechanism_record": record.model_dump(mode="json"),
    }
    aci.tools.memory_query = lambda query: {
        "status": "ACTUAL_PROVENANCE_FILTERED_TRAINING_RETRIEVAL",
        "authority": "NONE",
        "retrieval_event_sha256": "e" * 64,
        "admitted_count": 1,
        "admitted": [row],
    }
    context = EngineeringEpisodeContext(aci)
    assert context.frame()["memory_candidates"][0]["semantic_remedy"] == record.semantic_remedy
    result = context.call(
        {
            "operation": "ros_engineer.propose_memory_repair",
            "arguments": {"case_hash": aci.case.case_hash, "memory_ref": ref},
        }
    )
    assert result["status"] == "TYPED_STAGING_PROPOSAL_NOT_EXECUTED"
    assert result["historical_memory_ref"] == ref and result["historical_values_reused"] is False
    assert context.causality_trace[-1]["memory_refs_declared"] == [ref]
    assert aci.validation_count == 0
    for arm in ["A", "B"]:
        aci.tools.arm = arm
        assert (
            aci.call(
                {
                    "operation": "ros_engineer.propose_memory_repair",
                    "arguments": {"case_hash": aci.case.case_hash, "memory_ref": ref},
                }
            )["status"]
            == "REJECTED_TYPED_ACI_REQUEST"
        )


class ExistingMemoryStub:
    def store_experience(self, **row):
        self.row = {"id": row.pop("event_id"), **row}
        return self.row["id"]

    def find_similar_experiences(self, query, **options):
        return [deepcopy(self.row)]


def test_query_preserves_same_case_body_and_evaluation_run_exclusions(tmp_path):
    _, record, *_ = record_fixture(tmp_path)
    memory = ExistingMemoryStub()
    persist_mechanism(memory, record, fault_case_id="footprint_undersized")
    options = {
        "root": tmp_path,
        "evaluation_case": "scan_stale",
        "evaluation_runs": ["f" * 32],
        "urdf_sha256": record.source_urdf_sha256,
    }
    assert MechanismTrainingQuery(memory, [record], **options)("costmap")["admitted_count"] == 1
    for changed in [
        {"evaluation_case": "footprint_undersized"},
        {"evaluation_case": "j100_navigation_onboarding"},
        {"urdf_sha256": "0" * 64},
        {"evaluation_runs": [record.proof.after_run]},
    ]:
        result = MechanismTrainingQuery(memory, [record], **{**options, **changed})("costmap")
        assert result["admitted_count"] == 0
    memory.row["metadata"]["fault_case_id"] = "map_to_odom_missing"
    assert MechanismTrainingQuery(memory, [record], **options)("costmap")["admitted_count"] == 0


def test_changed_original_artifact_cannot_reenter_memory_context(tmp_path):
    _, record, *_ = record_fixture(tmp_path)
    memory = ExistingMemoryStub()
    persist_mechanism(memory, record, fault_case_id="footprint_undersized")
    (tmp_path / record.proof.episode.path).write_text("changed")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        MechanismTrainingQuery(
            memory,
            [record],
            root=tmp_path,
            evaluation_case="scan_stale",
            evaluation_runs=["f" * 32],
            urdf_sha256=record.source_urdf_sha256,
        )("costmap")


def test_know_payload_keeps_generated_mechanism_draft_and_does_not_promote_confidence(tmp_path):
    _, record, *_ = record_fixture(tmp_path)
    payload = know_unit_payload(record, now=datetime.now(UTC))
    assert payload["status"] == "draft" and payload["provenance_status"] == "generated"
    assert payload["confidence"] == 0.0 and payload["contraindications"]
    assert payload["evidence_refs"][0]["content_hash"] == record.proof.episode.sha256


def test_same_geometry_with_different_original_urdf_bytes_is_not_transferable(tmp_path):
    from rosclaw.connectors.ros.engineering import compile_engineering_case
    from tests.connectors.ros.test_engineering_case_compiler import NOW

    aci, record, *_ = record_fixture(tmp_path)
    other = compile_engineering_case(
        aci.model,
        aci.tools.files["robot.urdf"].encode() + b"\n<!-- different source -->",
        aci.requirements,
        source_ref="fixture:other",
        now=NOW,
        evidence_domain="SEALED_REPLAY",
    )
    with pytest.raises(ValueError, match="transfer"):
        instantiate_mechanism(
            record,
            other,
            aci.model,
            aci.requirements,
            aci.tools.staging,
            minimum_radius_m=aci.tools.minimum_radius_m,
        )


def test_similarly_named_field_outside_nav2_cannot_become_a_tf_lesson(tmp_path):
    _, _, _, _, proof = record_fixture(tmp_path)
    aci = fixture("tf")
    before = deepcopy(aci.tools.staging)
    before["bringup.json"]["tf_broadcast"] = False
    after = deepcopy(before)
    after["bringup.json"]["tf_broadcast"] = True
    with pytest.raises(ValueError, match="closed staging"):
        distill_mechanism(aci.case, before, after, before, proof)


@pytest.mark.parametrize("change", ["original", "blob", "manifest"])
def test_pinned_archive_checks_old_bytes_instead_of_rehashing_mutated_inputs(tmp_path, change):
    import gzip
    import json

    from benchmarks.ros_expert_ablation.mechanism_training import PinnedOriginalArchive

    original = tmp_path / "original.json"
    original.write_text('{"measured":true}')
    archive = tmp_path / "archive"
    archive.mkdir()
    blob = archive / "00000.gz"
    blob.write_bytes(gzip.compress(original.read_bytes(), mtime=0))

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    manifest = archive / "archive-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "00000.gz": {
                    "original_path": str(original),
                    "original_sha256": sha(original),
                    "stored_sha256": sha(blob),
                }
            }
        )
    )
    ref = {"path": str(manifest.relative_to(tmp_path)), "sha256": sha(manifest)}
    assert PinnedOriginalArchive(tmp_path, ref).check(original) == original
    if change == "original":
        original.write_text('{"measured":false}')
    elif change == "blob":
        blob.write_bytes(b"changed")
    else:
        manifest.write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch|pinned original archive"):
        PinnedOriginalArchive(tmp_path, ref).check(original)
