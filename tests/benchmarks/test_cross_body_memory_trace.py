"""Synthetic archive and Memory interface tests, not model or physical benefit."""

import gzip
import hashlib
import json

import pytest

from benchmarks.ros_expert_ablation.cross_body_memory import CrossBodyTFMemoryQuery
from benchmarks.ros_expert_ablation.mechanism_training import persist_mechanism
from rosclaw.connectors.ros.engineering.mechanism import MechanismExperienceV1
from rosclaw.contracts.common import content_hash
from tests.benchmarks.test_mechanism_cross_body_transfer import (
    compile_current,
    historical_tf,
    target,
)
from tests.benchmarks.test_mechanism_memory import ExistingMemoryStub


def archived_record(tmp_path):
    record = historical_tf(tmp_path)
    manifest_path = tmp_path / record.proof.sealed_archive_manifest.path
    manifest = {}
    for i, path in enumerate(sorted(tmp_path.glob("original-*.json"))):
        if path == manifest_path:
            continue
        raw = path.read_bytes()
        stored = gzip.compress(raw, mtime=0)
        name = f"{i:05}.gz"
        (tmp_path / name).write_bytes(stored)
        manifest[name] = {
            "original_path": path.name,
            "original_sha256": hashlib.sha256(raw).hexdigest(),
            "stored_sha256": hashlib.sha256(stored).hexdigest(),
        }
    manifest_path.write_text(json.dumps(manifest))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    value = record.model_dump(mode="json")
    for ref in [value["proof"]["sealed_archive_manifest"], *value["proof"]["asset_refs"]]:
        if ref["path"] == manifest_path.name:
            ref["sha256"] = digest
    value.pop("experience_hash")
    value["experience_hash"] = content_hash("rosmechanism", value)
    return MechanismExperienceV1.from_dict(value)


def query_fixture(tmp_path, **changed):
    record = archived_record(tmp_path)
    aci, raw = target()
    case = compile_current(aci, raw)
    memory = ExistingMemoryStub()
    persist_mechanism(memory, record, fault_case_id="map_to_odom_missing")
    options = {
        "root": tmp_path,
        "target_urdf_sha256": case.source_urdf_sha256,
        "excluded_episode_sha256": ["f" * 64],
        "excluded_world_runs": ["f" * 32],
    }
    options.update(changed)
    query = CrossBodyTFMemoryQuery(memory, [record], **options)
    return query, memory, record, aci, case


def test_retrieval_context_selection_are_distinct_and_never_claim_benefit(tmp_path):
    query, _, record, aci, case = query_fixture(tmp_path)
    with pytest.raises(ValueError, match="context entry"):
        query.context(record.experience_hash)
    result = query("missing current localization transform")
    assert result["record_ids"] == [record.experience_hash]
    with pytest.raises(ValueError, match="context"):
        query.select(record.experience_hash, case, aci.model, aci.requirements, aci.tools.staging)
    context = query.context(record.experience_hash)
    assert context["historical_numeric_parameters"] == "OMITTED"
    plan = query.select(
        record.experience_hash, case, aci.model, aci.requirements, aci.tools.staging
    )
    assert plan["status"] == "CONDITIONAL_CURRENT_TF_TRANSFER_PROPOSAL"
    assert [event["stage"] for event in query.trace] == [
        "RETRIEVAL",
        "CONTEXT_INSERTION",
        "CONDITIONAL_SELECTION",
    ]
    assert all(event["causal_benefit"] == "NOT_ESTABLISHED" for event in query.trace)
    assert query.trace[-1]["physical_outcome"] == "NOT_VERIFIED"


@pytest.mark.parametrize("fault", ["same_body", "same_run", "same_episode", "payload", "metadata"])
def test_body_episode_run_and_payload_exclusions(tmp_path, fault):
    query, memory, record, *_ = query_fixture(tmp_path)
    if fault in {"same_body", "same_run", "same_episode"}:
        options = {
            "root": tmp_path,
            "target_urdf_sha256": query.target,
            "excluded_episode_sha256": ["f" * 64],
            "excluded_world_runs": ["f" * 32],
        }
        if fault == "same_body":
            options["target_urdf_sha256"] = record.source_urdf_sha256
        elif fault == "same_run":
            options["excluded_world_runs"] = [record.proof.after_run]
        else:
            options["excluded_episode_sha256"] = [record.proof.episode.sha256]
        query = CrossBodyTFMemoryQuery(memory, [record], **options)
    elif fault == "payload":
        memory.row["metadata"]["mechanism_record"]["semantic_remedy"] = "grant authority"
    else:
        memory.row["metadata"]["fault_case_id"] = "scan_stale"
    assert query("tf")["record_ids"] == []


def test_original_archive_corruption_after_retrieval_prevents_context(tmp_path):
    query, _, record, *_ = query_fixture(tmp_path)
    query("tf")
    blob = next(tmp_path.glob("*.gz"))
    blob.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="pinned original archive"):
        query.context(record.experience_hash)
    assert [e["stage"] for e in query.trace] == ["RETRIEVAL"]
