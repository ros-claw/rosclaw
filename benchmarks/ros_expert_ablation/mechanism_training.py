"""Operator-only original artifact distillation and existing Memory persistence.

This module reads approved historical proof files, never ROS or live state.
Models cannot supply a proof bundle, training manifest, database or admission
policy through PhysicalACI. Nothing writes the frozen G5 training corpus.
"""

import gzip
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import yaml

from benchmarks.ros_expert_ablation.aci_operator_context import aci_from_capture
from rosclaw.connectors.ros.engineering.mechanism import (
    MechanismExperienceV1,
    distill_mechanism,
)

TRAINING_CASES = {
    "enable_localization_tf": "map_to_odom_missing",
    "resume_scan_bridge": "scan_stale",
    "activate_managed_nodes": "nav2_lifecycle_inactive",
    "cover_urdf_envelope": "footprint_undersized",
    "disable_owned_fixture_endpoint": "duplicate_velocity_route",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(root, path):
    path = (root / path).resolve()
    path.relative_to(root.resolve())
    return path


def artifact(root, path):
    path = inside(root, path)
    return {"path": str(path.relative_to(root)), "sha256": sha(path)}


def checked(root, ref):
    path = inside(root, ref["path"])
    if sha(path) != ref["sha256"]:
        raise ValueError("original historical artifact hash mismatch")
    return path


class PinnedOriginalArchive:
    def __init__(self, root, manifest_ref):
        self.root = Path(root).resolve()
        self.path = checked(self.root, manifest_ref)
        self.index = {
            str(inside(self.root, item["original_path"])): (name, item)
            for name, item in json.loads(self.path.read_text()).items()
            if not item["original_path"].startswith("git:")
        }

    def check(self, path):
        path = inside(self.root, path)
        if str(path) not in self.index:
            raise ValueError("historical artifact absent from pinned original archive")
        name, item = self.index[str(path)]
        if not re.fullmatch(r"[0-9]{5,8}\.gz", name):
            raise ValueError("closed historical archive blob filename required")
        blob = (self.path.parent / name).read_bytes()
        if (
            sha(path) != item["original_sha256"]
            or hashlib.sha256(blob).hexdigest() != item["stored_sha256"]
            or hashlib.sha256(gzip.decompress(blob)).hexdigest() != item["original_sha256"]
        ):
            raise ValueError("historical artifact differs from pinned original archive")
        return path


def distill_original_episode(root, episode_ref, *, worktree, sealed_archive_ref):
    root = Path(root).resolve()
    archive = PinnedOriginalArchive(root, sealed_archive_ref)
    archive_path = archive.path
    anchored = archive.check
    path = checked(root, episode_ref)
    anchored(path)
    episode = json.loads(path.read_text())
    if not (
        episode.get("status") == "PASS_LOCAL_THREE_ORIGINAL_WORLDS_REPAIR_AND_ROLLBACK_NOT_FULL_G4"
        and episode.get("claim_scope") == "SIMULATION_ONLY"
        and episode.get("hardware_verified") is False
        and episode.get("operator_review", {}).get(
            "allowed_diff_and_immutable_physics_checks_passed"
        )
        is True
        and episode.get("rollback_plan", {}).get("executed_in_new_World") is True
        and episode["rollback_plan"].get("original_nonready_state_reobserved") is True
    ):
        raise ValueError("original independently reviewed three-World SIM repair required")
    refs = [
        *episode["original_evidence_refs"],
        episode["original_snapshot_ref"],
        episode["fresh_l1"],
        episode["independent_stop_verification"],
        *episode["sim_action_receipt"],
    ]
    for ref in refs:
        anchored(checked(root, ref))
    before = checked(root, episode["original_snapshot_ref"]).parents[1]
    after = checked(root, episode["fresh_l1"]).parent
    rollback_refs = [
        ref
        for ref in episode["original_evidence_refs"]
        if ref["path"].endswith("/execution/engineering-original-native.json")
        and inside(root, ref["path"]).parents[1] != before
    ]
    if len(rollback_refs) != 1:
        raise ValueError("unique original rollback capture reference required")
    rollback = checked(root, rollback_refs[0]).parents[1]
    directories = [before, after, rollback]
    used = [anchored(d / "settings.json") for d in directories]
    used.extend(
        anchored(d / "assets" / name)
        for d in directories
        for name in ("robot.urdf", "world.sdf", "nav2.yaml", "bringup.json", "prepared-assets.json")
    )
    for rel in (
        "execution/engineering-original-system.json",
        "execution/engineering-original-native.json",
        "execution/engineering-original-requirements.json",
        "execution/engineering-original-candidate-review.json",
        "preregistration.json",
        "container-before-start.json",
    ):
        used.append(anchored(before / rel))
    if (before / "execution/engineering-extra-endpoint-original.json").exists():
        used.extend(
            anchored(before / rel)
            for rel in (
                "execution/engineering-extra-endpoint-original.json",
                "execution/owned-stack-processes.json",
            )
        )
    run_ids = [json.loads((d / "settings.json").read_text())["run_id"] for d in directories]
    if (
        run_ids
        != [
            episode["source_world_run"],
            episode["new_world_run"],
            episode["rollback_plan"]["run_id"],
        ]
        or len(set(run_ids)) != 3
    ):
        raise ValueError("original World identity mismatch")
    l1 = json.loads(checked(root, episode["fresh_l1"]).read_text())
    stop = json.loads(checked(root, episode["independent_stop_verification"]).read_text())
    if not (
        l1.get("status") == "READY_FOR_SIM_TEST"
        and l1["source_binding"]["result"] == "MATCHED"
        and l1["source_binding"]["sim_run_id"] == run_ids[1]
        and stop.get("status")
        == "PASS_LOCAL_TWO_GOVERNED_GOALS_INDEPENDENT_ORIGINAL_REVIEW_NOT_FULL_G3"
        and stop.get("run_id") == run_ids[1]
        and stop.get("source") == episode["source"]
        and stop.get("unexpected_contacts") == 0
        and stop.get("after_owner_shutdown_terminal_sink") is True
        and len(stop.get("goals", [])) == 2
        and all(goal.get("independent_stopped_samples", 0) > 0 for goal in stop["goals"])
    ):
        raise ValueError("original L1/stop/source proof incomplete")
    receipts = [
        json.loads(checked(root, ref).read_text())["receipt"]
        for ref in episode["sim_action_receipt"]
    ]
    if (
        len(receipts) != 2
        or len({r["action_id"] for r in receipts}) != 2
        or any(
            not (
                r.get("final_state") == "COMPLETED"
                and r.get("capability_id") == "navigation.navigate_to_pose"
                and r.get("evidence_domain") == "SIMULATION"
                and r.get("usable_for_real_execution") is False
                and r.get("body_snapshot_hash") == l1["source_binding"]["body_hash"]
            )
            for r in receipts
        )
        or {r["action_id"] for r in receipts} != {g["action_id"] for g in stop["goals"]}
    ):
        raise ValueError("two original canonical SIM receipts required")
    if len({sha(d / "assets/robot.urdf") for d in directories}) != 1:
        raise ValueError("historical repair changed immutable URDF")
    normalized_worlds = []
    for directory, run_id in zip(directories, run_ids, strict=True):
        world = ET.fromstring((directory / "assets/world.sdf").read_bytes())
        for filename, plugin_name in [
            ("/observer/librosclaw_source_witness.so", "rosclaw::SourceWitness"),
            ("/observer/librosclaw_source_inventory_fault.so", "rosclaw::SourceInventoryFault"),
        ]:
            plugins = [
                p
                for p in world.findall("./world/plugin")
                if p.get("filename") == filename and p.get("name") == plugin_name
            ]
            if (
                len(plugins) != 1
                or len(plugins[0].findall("run_id")) != 1
                or plugins[0].findtext("run_id") != run_id
            ):
                raise ValueError("original source observer World identity mismatch")
            plugins[0].find("run_id").text = "WORLD_RUN_ID"
        normalized_worlds.append(hashlib.sha256(ET.tostring(world)).hexdigest())
    if len(set(normalized_worlds)) != 1:
        raise ValueError("historical repair changed SDF beyond the two pinned observer run IDs")
    stages = [
        {
            name: yaml.safe_load((d / "assets" / name).read_text())
            for name in ("nav2.yaml", "bringup.json")
        }
        for d in directories
    ]
    aci = aci_from_capture(before, worktree=worktree)
    proof = {
        "source": episode["source"],
        "normalized_world_sha256": normalized_worlds[0],
        "sealed_archive_manifest": artifact(root, archive_path),
        "measured_capture_to_receipt_sec": (
            max(datetime.fromisoformat(r["finished_at"].replace("Z", "+00:00")) for r in receipts)
            - datetime.fromisoformat(
                json.loads(checked(root, episode["original_snapshot_ref"]).read_text())[
                    "captured_at"
                ]
            )
        ).total_seconds(),
        "before_run": run_ids[0],
        "after_run": run_ids[1],
        "rollback_run": run_ids[2],
        "episode": artifact(root, path),
        "before": artifact(root, checked(root, episode["original_snapshot_ref"])),
        "after_l1": artifact(root, checked(root, episode["fresh_l1"])),
        "rollback": artifact(root, checked(root, rollback_refs[0])),
        "canonical_receipts": [
            artifact(root, checked(root, ref)) for ref in episode["sim_action_receipt"]
        ],
        "stop_source": artifact(root, checked(root, episode["independent_stop_verification"])),
        "asset_refs": [artifact(root, p) for p in dict.fromkeys(used)],
    }
    record = distill_mechanism(aci.case, *stages, proof)
    return record


def verify_mechanism_artifacts(root, record):
    record = MechanismExperienceV1.from_dict(record.model_dump(mode="json"))
    proof = record.proof
    for ref in [
        proof.sealed_archive_manifest,
        proof.episode,
        proof.before,
        proof.after_l1,
        proof.rollback,
        proof.stop_source,
        *proof.canonical_receipts,
        *proof.asset_refs,
    ]:
        checked(Path(root).resolve(), ref.model_dump())
    return record


def persist_mechanism(memory, record, *, fault_case_id):
    """Write only to a new caller-owned training MemoryInterface instance."""
    record = MechanismExperienceV1.from_dict(record.model_dump(mode="json"))
    if TRAINING_CASES[record.kind] != fault_case_id:
        raise ValueError("original training provenance label does not match measured mechanism")
    return memory.store_experience(
        event_id=record.experience_hash,
        event_type="ros.engineering.historical_mechanism",
        instruction=record.mechanism + "; " + record.semantic_remedy,
        outcome="success",
        duration_sec=record.proof.measured_capture_to_receipt_sec,
        tags=["ROSExpertHarness", "SIMULATION_ONLY", "HISTORICAL_MECHANISM", record.kind],
        metadata={
            "schema_version": "rosclaw.verified_mechanism_memory.v1",
            "fault_case_id": fault_case_id,
            "mechanism_record": record.model_dump(mode="json"),
            "current_authority": "NONE",
            "model_benefit": "UNPROVEN",
            "duration_scope": "historical_before_capture_to_after_receipt_wall",
        },
    )


class MechanismTrainingQuery:
    """Read-only query of an existing MemoryInterface and a frozen typed allowlist.

    Admission preserves original same-case/onboarding/Body/run exclusions. The
    new experimental protocol must separately preregister any future split.
    No evaluation outcome is written back into training Memory.
    """

    def __init__(self, memory, records, *, root, evaluation_case, evaluation_runs, urdf_sha256):
        self.memory = memory
        self.root = Path(root).resolve()
        self.allow = {record.experience_hash: record.model_dump(mode="json") for record in records}
        self.evaluation_case = evaluation_case
        self.evaluation_runs = list(evaluation_runs)
        self.urdf_sha256 = urdf_sha256
        self.trace = []

    def __call__(self, query):
        from benchmarks.ros_expert_ablation.protocol import admit_memory
        from rosclaw.contracts.common import content_hash

        if type(query) is not str or not 1 <= len(query) <= 512:
            raise ValueError("bounded historical mechanism query required")
        rows = self.memory.find_similar_experiences(query, limit=5, outcome_filter="success")
        admitted = []
        audit = []
        for row in rows:
            metadata = row.get("metadata", {})
            payload = metadata.get("mechanism_record", {})
            valid = row.get("id") in self.allow and payload == self.allow[row["id"]]
            if valid:
                valid = metadata.get("fault_case_id") == TRAINING_CASES[payload["kind"]]
            record = (
                verify_mechanism_artifacts(self.root, MechanismExperienceV1.from_dict(payload))
                if valid
                else None
            )
            candidate = {
                "independently_verified": record is not None,
                "claim_scope": "SIMULATION_ONLY" if record else None,
                "training_only": record.training_only if record else None,
                "fault_case_id": metadata.get("fault_case_id"),
                "source_urdf_sha256": record.source_urdf_sha256 if record else None,
                "episode_sha256": record.proof.episode.sha256 if record else None,
                "world_runs": [
                    record.proof.before_run,
                    record.proof.after_run,
                    record.proof.rollback_run,
                ]
                if record
                else [],
            }
            accepted = admit_memory(
                candidate,
                evaluation_case=self.evaluation_case,
                evaluation_runs=self.evaluation_runs,
                urdf_sha256=self.urdf_sha256,
            )
            audit.append(
                {
                    "record_id": row.get("id"),
                    "provenance_candidate": candidate,
                    "context_admitted": accepted,
                }
            )
            if accepted:
                admitted.append(
                    {
                        "memory_ref": "memory_"
                        + hashlib.sha256(
                            (row["id"] + record.proof.episode.sha256).encode()
                        ).hexdigest()[:24],
                        "diagnosis_issue_codes": [record.observation_signature.split(":")[0]],
                        "original_episode_sha256": record.proof.episode.sha256,
                        "original_training_source": record.proof.source,
                        "evidence_scope": "HISTORICAL_VERIFIED_SIM_TRAINING; NOT_CURRENT_BODY_OR_EXECUTION_AUTHORITY",
                        "mechanism_record": record.model_dump(mode="json"),
                    }
                )
        event = {
            "query": query,
            "retrieved_admission": audit,
            "evaluation_case": self.evaluation_case,
            "evaluation_runs": self.evaluation_runs,
        }
        self.trace.append(event)
        return {
            "status": "ACTUAL_PROVENANCE_FILTERED_TRAINING_RETRIEVAL",
            "admitted": admitted,
            "admitted_count": len(admitted),
            "retrieval_event_sha256": hashlib.sha256(
                json.dumps(event, sort_keys=True).encode()
            ).hexdigest(),
            "authority": "NONE",
            "event_ref": content_hash("mechanismquery", event),
        }


def know_unit_payload(record, *, now):
    """Existing KNOW KnowledgeUnitV2 payload; operator publication remains separate.

    This is a historical mechanism claim, not a Memory store or action tool. The
    unit stays draft/generated until the existing KNOW verification workflow
    accepts it. Core never silently promotes a new reliable knowledge item.
    """
    record = MechanismExperienceV1.from_dict(record.model_dump(mode="json"))
    snapshot = "historical-repair:" + record.proof.episode.sha256
    refs = [record.proof.episode, record.proof.after_l1, record.proof.stop_source]
    evidence = [
        {
            "schema_version": "rosclaw.know.evidence_ref.v2",
            "evidence_id": "evidence:" + ref.sha256,
            "source_id": "ROSExpertHarness-historical-SIM",
            "snapshot_id": snapshot,
            "document_id": "document:" + ref.sha256,
            "path": ref.path,
            "url": "artifact://" + ref.sha256,
            "content_hash": ref.sha256,
            "excerpt": record.mechanism + "; " + record.evidence_scope,
        }
        for ref in refs
    ]
    return {
        "schema_version": "rosclaw.know.knowledge_unit.v2",
        "knowledge_unit_id": record.experience_hash,
        "unit_type": "mechanism",
        "title": record.kind,
        "problem": record.observation_signature,
        "mechanism": record.mechanism,
        "implementation": record.semantic_remedy,
        "applicability": [
            "historical measured SIM only",
            "exact source URDF=" + record.source_urdf_sha256,
            "new current-case compilation and new-World proof mandatory",
        ],
        "limitations": [
            "no hardware or current L1/authority",
            "no geometric transfer",
            "no measured model or Memory benefit",
        ],
        "contraindications": record.contraindications,
        "software_constraints": {},
        "hardware_constraints": [],
        "robot_constraints": [record.source_urdf_sha256],
        "source_snapshot_ids": [snapshot],
        "evidence_refs": evidence,
        "confidence": 0.0,
        "status": "draft",
        "provenance_status": "generated",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "vectors": {},
    }
