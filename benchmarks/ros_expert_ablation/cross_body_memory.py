"""New RK05 read-only Memory split; never changes frozen G5 admission.

Only operator-allowlisted TF concepts with original archive checks may enter
context. Retrieval, context insertion and selection are separately recorded;
none establishes a causal benefit or current physical success.
"""

import re
from pathlib import Path

from benchmarks.ros_expert_ablation.mechanism_training import (
    PinnedOriginalArchive,
    verify_mechanism_artifacts,
)
from rosclaw.connectors.ros.engineering.mechanism import MechanismExperienceV1
from rosclaw.connectors.ros.engineering.mechanism_transfer import propose_tf_mechanism_transfer
from rosclaw.contracts.common import content_hash


class CrossBodyTFMemoryQuery:
    def __init__(
        self,
        memory,
        records,
        *,
        root,
        target_urdf_sha256,
        excluded_episode_sha256,
        excluded_world_runs,
    ):
        self.memory = memory
        self.root = Path(root).resolve()
        self.target = target_urdf_sha256
        self.excluded_episodes = frozenset(excluded_episode_sha256)
        self.excluded_runs = frozenset(excluded_world_runs)
        if (
            re.fullmatch("[0-9a-f]{64}", self.target) is None
            or not self.excluded_episodes
            or not self.excluded_runs
            or any(re.fullmatch("[0-9a-f]{64}", value) is None for value in self.excluded_episodes)
            or any(re.fullmatch("[0-9a-f]{32}", value) is None for value in self.excluded_runs)
        ):
            raise ValueError("frozen target SHA, excluded original episodes and World IDs required")
        self.allow = {}
        for record in records:
            record = MechanismExperienceV1.from_dict(record.model_dump(mode="json"))
            if record.kind != "enable_localization_tf":
                raise ValueError("only the frozen TF concept belongs in this split")
            if record.experience_hash in self.allow:
                raise ValueError("duplicate frozen mechanism")
            self.allow[record.experience_hash] = record.model_dump(mode="json")
        self.policy = {
            "schema_version": "rosclaw.cross_body_tf_memory_policy.v1",
            "target_urdf_sha256": self.target,
            "excluded_episode_sha256": sorted(self.excluded_episodes),
            "excluded_world_runs": sorted(self.excluded_runs),
            "allowlist": self.allow,
            "legacy_G5_policy": "UNCHANGED",
            "authority": "NONE",
        }
        self.policy_hash = content_hash("crossbodymemorypolicy", self.policy)
        self.trace = []
        self.retrieved = {}
        self.in_context = set()

    def _event(self, stage, **fields):
        event = {
            "schema_version": "rosclaw.cross_body_memory_use.v1",
            "stage": stage,
            "policy_hash": self.policy_hash,
            "authority": "NONE",
            "causal_benefit": "NOT_ESTABLISHED",
            **fields,
        }
        event["event_hash"] = content_hash("crossbodymemoryuse", event)
        self.trace.append(event)
        return event

    def _verified(self, record):
        record = verify_mechanism_artifacts(self.root, record)
        proof = record.proof
        archive = PinnedOriginalArchive(self.root, proof.sealed_archive_manifest.model_dump())
        for ref in [
            proof.episode,
            proof.before,
            proof.after_l1,
            proof.rollback,
            proof.stop_source,
            *proof.canonical_receipts,
            *proof.asset_refs,
        ]:
            if ref == proof.sealed_archive_manifest:
                continue
            archive.check(ref.path)
        return record

    def __call__(self, query):
        if type(query) is not str or not 1 <= len(query) <= 512:
            raise ValueError("bounded historical concept query required")
        rows = self.memory.find_similar_experiences(query, limit=5, outcome_filter="success")
        if not isinstance(rows, list) or len(rows) > 5:
            raise ValueError("bounded existing Memory result required")
        audit, admitted = [], []
        for row in rows:
            key = row.get("id")
            payload = row.get("metadata", {}).get("mechanism_record")
            reason = "NOT_IN_FROZEN_ALLOWLIST_OR_PAYLOAD_CHANGED"
            record = None
            metadata = row.get("metadata", {})
            if (
                key in self.allow
                and payload == self.allow[key]
                and metadata.get("schema_version") == "rosclaw.verified_mechanism_memory.v1"
                and metadata.get("fault_case_id") == "map_to_odom_missing"
            ):
                record = MechanismExperienceV1.from_dict(payload)
                runs = {record.proof.before_run, record.proof.after_run, record.proof.rollback_run}
                if record.source_urdf_sha256 == self.target:
                    reason = "SAME_BODY_NOT_CROSS_BODY"
                elif (
                    record.proof.episode.sha256 in self.excluded_episodes
                    or runs & self.excluded_runs
                ):
                    reason = "EVALUATION_EPISODE_OR_WORLD_LEAKAGE"
                else:
                    self._verified(record)
                    self.retrieved[key] = record
                    admitted.append(key)
                    reason = "ADMITTED_HISTORICAL_CONCEPT_ONLY"
            audit.append({"record_id": key, "reason": reason})
        event = self._event("RETRIEVAL", query=query, audit=audit, admitted=admitted)
        return {"record_ids": admitted, "event_hash": event["event_hash"], "authority": "NONE"}

    def context(self, record_id):
        if record_id not in self.retrieved:
            raise ValueError("context entry requires actual admitted retrieval")
        record = self._verified(self.retrieved[record_id])
        concept = {
            "record_id": record_id,
            "mechanism": record.mechanism,
            "semantic_remedy": record.semantic_remedy,
            "observation_signature": record.observation_signature,
            "contraindications": record.contraindications,
            "historical_numeric_parameters": "OMITTED",
            "current_authority": "NONE",
            "new_World_required": True,
        }
        self.in_context.add(record_id)
        self._event(
            "CONTEXT_INSERTION",
            record_id=record_id,
            context_hash=content_hash("crossbodymemorycontext", concept),
        )
        return concept

    def select(self, record_id, case, model, requirements, staging):
        if record_id not in self.in_context:
            raise ValueError("selection trace requires admitted concept in context")
        if case.source_urdf_sha256 != self.target:
            raise ValueError("current target differs from frozen Memory policy")
        record = self._verified(self.retrieved[record_id])
        plan = propose_tf_mechanism_transfer(record, case, model, requirements, staging)
        self._event(
            "CONDITIONAL_SELECTION",
            record_id=record_id,
            target_case_hash=case.case_hash,
            plan_hash=plan["transfer_plan_hash"],
            status=plan["status"],
            physical_outcome="NOT_VERIFIED",
        )
        return plan
