"""Current-state context and automatic historical priors for a future protocol.

No model runner is changed. Full requests/results stay in PhysicalACI.trace;
frames contain current decisions and content references rather than transcript.
The host supplies the existing provenance-filtered training Memory callback.
"""

import copy
import json
import re
from collections import Counter

from benchmarks.ros_expert_ablation.physical_aci import Empty, InspectCall
from rosclaw.agentd.context.tokens import estimate_tokens
from rosclaw.connectors.ros.engineering import RosEngineeringCaseV1
from rosclaw.connectors.ros.engineering.mechanism import MechanismExperienceV1
from rosclaw.contracts.common import content_hash


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value, length):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value)


class EngineeringEpisodeContext:
    """External benchmark adapter, not another agent runtime or Memory store.

    Construction in C performs one host retrieval, including an empty admission.
    A/B never retrieve. Model Memory citations are optional and separately audited;
    retrieval alone is not usage, benefit, instruction-following or permission.
    """

    def __init__(self, aci):
        self.aci = aci
        self.revision = 0
        self.stages = Counter()
        self.last_result = None
        self.causality_trace = []
        self.memory_candidates = []
        self.memory_retrieval = None
        self.frame_metrics = []
        if aci.tools.arm == "C":
            if aci.tools.memory_query is None:
                raise ValueError("C automatic retrieval requires the host training Memory backend")
            signature = (
                ";".join(c.invariant for c in aci.case.constraints if c.status == "VIOLATED")
                or "navigation onboarding; no observed violated constraint"
            )
            result = aci.tools.call("query_memory", {"query": signature[:512]})
            self.memory_retrieval = copy.deepcopy(result)
            self.stages["host_memory_retrieval"] += 1
            self.memory_candidates = self._project_admitted(result)
            projected = {row["memory_ref"] for row in self.memory_candidates}
            for row in result["admitted"]:
                if row["memory_ref"] in projected and "mechanism_record" in row:
                    self.aci.admitted_mechanisms[row["memory_ref"]] = (
                        MechanismExperienceV1.from_dict(row["mechanism_record"])
                    )

    @staticmethod
    def _project_admitted(result):
        # The callback is host-owned and checks the actual frozen database, record,
        # original episode bytes, same-case exclusion, Body and evaluation run IDs.
        # Reject an incompatible callback instead of treating arbitrary text as KNOW.
        if not (
            isinstance(result, dict)
            and result.get("status") == "ACTUAL_PROVENANCE_FILTERED_TRAINING_RETRIEVAL"
            and result.get("authority") == "NONE"
            and digest(result.get("retrieval_event_sha256"), 64)
            and isinstance(result.get("admitted"), list)
            and type(result.get("admitted_count")) is int
            and result["admitted_count"] == len(result["admitted"])
        ):
            raise ValueError("host provenance admission receipt required")
        candidates = []
        refs = set()
        for row in result["admitted"]:
            if not isinstance(row, dict):
                raise ValueError("invalid admitted historical Memory projection")
            ref = row.get("memory_ref", "")
            if not (
                isinstance(ref, str)
                and re.fullmatch(r"memory_[0-9a-f]{24}", ref)
                and ref not in refs
                and row.get("evidence_scope")
                == "HISTORICAL_VERIFIED_SIM_TRAINING; NOT_CURRENT_BODY_OR_EXECUTION_AUTHORITY"
                and digest(row.get("original_episode_sha256"), 64)
                and digest(row.get("original_training_source"), 40)
                and isinstance(row.get("diagnosis_issue_codes"), list)
                and len(row["diagnosis_issue_codes"]) <= 32
                and all(
                    isinstance(code, str) and len(code) <= 128
                    for code in row["diagnosis_issue_codes"]
                )
            ):
                raise ValueError("invalid admitted historical Memory projection")
            refs.add(ref)
            candidates.append(
                {
                    "memory_ref": ref,
                    "diagnosis_issue_codes": row["diagnosis_issue_codes"],
                    "original_episode_sha256": row["original_episode_sha256"],
                    "role": "HISTORICAL_PRIOR",
                    "current_authority": "NONE",
                }
            )
            if "mechanism_record" in row:
                record = MechanismExperienceV1.from_dict(row["mechanism_record"])
                if (
                    record.proof.episode.sha256 != row["original_episode_sha256"]
                    or record.proof.source != row["original_training_source"]
                ):
                    raise ValueError("historical mechanism/admission provenance mismatch")
                candidates[-1].update(
                    {
                        "experience_ref": record.experience_hash,
                        "mechanism": record.mechanism,
                        "semantic_remedy": record.semantic_remedy,
                        "contraindications": record.contraindications,
                    }
                )
        # Preserve backend retrieval ordering, not an answer derived from case labels.
        return candidates[:2]

    def frame(self):
        RosEngineeringCaseV1.from_dict(self.aci.case.model_dump(mode="json"))
        self.aci._bound(self.aci.case.case_hash, read_only=True)
        # Pure inspection; this is not an additional model operation.
        card = self.aci._dispatch(
            InspectCall(operation="ros_engineer.inspect_task", arguments=Empty()),
            remaining_sec=None,
        )
        phase = (
            "FINISHED"
            if self.aci.finished
            else "VERIFIED_OTHER_WORLD"
            if any(r.get("accepted_success") is True for r in self.aci.results.values())
            else "VALIDATION_REVIEW"
            if self.aci.results
            else "PROPOSED"
            if self.aci.proposals
            else "INSPECTED"
        )
        payload = {
            "schema_version": "rosclaw.engineering_episode_context.v1",
            "state_revision": self.revision,
            "phase": phase,
            "task": card,
            "proposals": [
                {"proposal_id": key, "changes": len(value["changes"]), "stage_only": True}
                for key, value in self.aci.proposals.items()
            ],
            "verification_refs": list(self.aci.results),
            "recent_result": self.last_result,
            "memory_candidates": copy.deepcopy(self.memory_candidates),
            "current_authority": "NONE",
            "current_l1": "NOT_EVALUATED",
            "scope": "SEALED_EPISODE_CONTEXT_NOT_LIVE_SELF",
        }
        payload["state_hash"] = content_hash("acistate", payload)
        text = encoded(payload)
        # Never silently truncate constraints, source bindings or permission facts.
        if len(text) > 10000:
            raise ValueError(
                "current decision state exceeds frame budget; operator review required"
            )
        self.frame_metrics.append(
            {
                "state_revision": self.revision,
                "characters": len(text),
                "utf8_bytes": len(text.encode()),
                "estimated_tokens": estimate_tokens(text),
                "token_measurement": "ESTIMATE_NOT_PROVIDER_USAGE",
            }
        )
        return copy.deepcopy(payload)

    def call(self, request, *, memory_refs_used=(), remaining_sec=None):
        admitted = {row["memory_ref"] for row in self.memory_candidates}
        if (
            not isinstance(memory_refs_used, (list, tuple))
            or any(type(ref) is not str for ref in memory_refs_used)
            or not set(memory_refs_used).issubset(admitted)
        ):
            raise ValueError("only actually admitted episode Memory references may be cited")
        operation = request.get("operation", "INVALID") if isinstance(request, dict) else "INVALID"
        if not isinstance(operation, str):
            operation = "INVALID"
        before = list(self.aci.proposals)
        if operation == "ros_engineer.propose_memory_repair":
            ref = (
                request.get("arguments", {}).get("memory_ref")
                if isinstance(request.get("arguments"), dict)
                else None
            )
            if isinstance(ref, str) and ref in admitted and ref not in memory_refs_used:
                memory_refs_used = [*memory_refs_used, ref]
        result = self.aci.call(request, remaining_sec=remaining_sec)
        self.stages[operation] += 1
        self.revision += 1
        result_ref = content_hash("aciresult", result)
        self.last_result = {"status": result["status"], "result_ref": result_ref}
        if result["status"] == "REJECTED_TYPED_ACI_REQUEST":
            error = result.get("error", "")
            self.last_result["error_excerpt"] = error[:400]
            self.last_result["error_excerpt_truncated"] = len(error) > 400
            self.last_result["legal_example"] = copy.deepcopy(result["legal_example"])
        if "verification" in result:
            self.last_result["verification"] = {
                key: result["verification"][key]
                for key in [
                    "acceptance_status",
                    "accepted_success",
                    "evidence_complete",
                    "in_budget",
                ]
            }
        self.causality_trace.append(
            {
                "state_revision": self.revision,
                "memory_refs_declared": list(memory_refs_used),
                "operation": operation,
                "new_proposal_ids": [key for key in self.aci.proposals if key not in before],
                "result_ref": result_ref,
                "status": result["status"],
                "original_result_utf8_bytes": len(encoded(result).encode()),
                "verification_refs": list(self.aci.results),
                "causal_benefit": "NOT_ESTABLISHED_BY_CITATION",
                "selection_effect": "EXPLICIT_MEMORY_REPAIR_INTENT"
                if operation == "ros_engineer.propose_memory_repair"
                and result["status"] == "TYPED_STAGING_PROPOSAL_NOT_EXECUTED"
                else "DECLARED_REFERENCE_ONLY_OR_NO_OBSERVED_SELECTION",
                "priority_change_counterfactual": "UNKNOWN_WITHOUT_COMPARISON",
            }
        )
        return result
