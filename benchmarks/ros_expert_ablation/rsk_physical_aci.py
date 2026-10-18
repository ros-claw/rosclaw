"""Separate RSK B/C/E facade; old G5 ACI, arm meanings and admission stay intact.

Only operator-injected audited Memory can enter E. New calls form conditional
staging proposals; explicit validation delegates to the existing SDK boundary.
"""

from __future__ import annotations

import copy
from typing import Annotated, Literal

from pydantic import Field, StrictStr, TypeAdapter, ValidationError

from benchmarks.ros_expert_ablation.aci_schema import _project, normalize_decision
from benchmarks.ros_expert_ablation.cross_body_memory import CrossBodyTFMemoryQuery
from benchmarks.ros_expert_ablation.physical_aci import (
    Args,
    CaseArgs,
    PhysicalACI,
    RepairArgs,
    RepairCall,
)
from benchmarks.ros_expert_ablation.rsk_pilot_protocol import RskPilotProtocolV2, review_pilot_entry


class CrossBodyQueryArgs(CaseArgs):
    query: StrictStr = Field(min_length=1, max_length=512)


class CrossBodyProposalArgs(CaseArgs):
    memory_ref: StrictStr = Field(min_length=1, max_length=256)


class CrossBodyQueryCall(Args):
    operation: Literal["ros_engineer.query_cross_body_memory"]
    arguments: CrossBodyQueryArgs


class CrossBodyProposalCall(Args):
    operation: Literal["ros_engineer.propose_cross_body_memory_repair"]
    arguments: CrossBodyProposalArgs


CROSS_BODY_CALLS = TypeAdapter(
    Annotated[CrossBodyQueryCall | CrossBodyProposalCall, Field(discriminator="operation")]
)


class RskPhysicalACI:
    def __init__(
        self,
        base: PhysicalACI,
        *,
        arm,
        memory_query=None,
        pilot_protocol=None,
        pilot_evidence=None,
        evidence_root=None,
    ):
        if (
            arm not in {"B", "C", "E"}
            or base.tools.arm != "B"
            or base.max_validations != 1
            or (arm == "B" and base.probe_evaluator is not None)
            or (arm != "E" and memory_query is not None)
        ):
            raise ValueError("separate RSK B/C/E arm, legacy B tools and one validation required")
        if arm == "E" and (
            not isinstance(memory_query, CrossBodyTFMemoryQuery)
            or memory_query.target != base.case.source_urdf_sha256
        ):
            raise ValueError("E requires operator-injected audited current-Body Memory policy")
        self.pilot_entry_review = None
        if base.evaluator is not None or base.probe_evaluator is not None:
            if not isinstance(pilot_protocol, RskPilotProtocolV2) or evidence_root is None:
                raise ValueError("RSK pilot entry proof required before SDK/probe callbacks")
            entry = review_pilot_entry(pilot_protocol, pilot_evidence, root=evidence_root)
            if entry["status"] != "ELIGIBLE_FOR_BOUNDED_PILOT_ONLY":
                raise ValueError("RSK pilot entry is NO_GO; SDK/probe callbacks disabled")
            if base.operator_source != pilot_protocol.harness_source:
                raise ValueError("SDK/probe operator source differs from frozen harness")
            self.pilot_entry_review = entry
        self.base = base
        self.arm = arm
        self.memory_query = memory_query
        self.trace: list[dict] = []
        self.world_budget = {"read_only_probe": 1, "canonical_validation": 1, "total": 2}

    def request_schema(self):
        old, new = PhysicalACI.request_schema(), CROSS_BODY_CALLS.json_schema()
        definitions = copy.deepcopy(old["$defs"])
        for key, value in new["$defs"].items():
            if key in definitions and definitions[key] != value:
                raise ValueError("ambiguous typed operation definition")
            definitions[key] = value
        disabled = "ros_engineer.propose_memory_repair"
        mapping = {k: v for k, v in old["discriminator"]["mapping"].items() if k != disabled}
        if self.arm == "E":
            mapping.update(new["discriminator"]["mapping"])
        return {
            "$defs": definitions,
            "oneOf": [{"$ref": ref} for ref in mapping.values()],
            "discriminator": {"propertyName": "operation", "mapping": mapping},
        }

    def decision_schema(self):
        """Provider-friendly phase-bound schema for a NEW RSK client, not G5 runner."""
        schema = self.request_schema()
        definitions = schema["$defs"]
        refs = sorted(self.memory_query.in_context) if self.memory_query is not None else []
        mapping = schema["discriminator"]["mapping"]
        if not refs:
            mapping.pop("ros_engineer.propose_cross_body_memory_repair", None)
        if not self.base.proposals:
            mapping.pop("ros_engineer.validate_proposal", None)
        if not self.base.results:
            mapping.pop("ros_engineer.explain_result", None)
        for definition in definitions.values():
            properties = definition.get("properties", {})
            for field, values in [
                ("case_hash", [self.base.case.case_hash]),
                ("proposal_id", list(self.base.proposals)),
                ("verification_ref", list(self.base.results)),
                ("memory_ref", refs),
            ]:
                if field in properties and values:
                    properties[field]["enum"] = values
        return _project(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "decision": {"oneOf": [{"$ref": value} for value in mapping.values()]},
                    "decision_summary": {"type": "string", "maxLength": 2000},
                    "memory_refs_used": {
                        "type": "array",
                        "maxItems": len(refs),
                        "items": {"type": "string", **({"enum": refs} if refs else {})},
                    },
                },
                "required": ["decision", "decision_summary", "memory_refs_used"],
                "$defs": definitions,
            }
        )

    def call_model_decision(self, value, *, remaining_sec=None):
        """Validate the new client's envelope; this never invokes a model itself."""
        if not isinstance(value, dict) or set(value) != {
            "decision",
            "decision_summary",
            "memory_refs_used",
        }:
            raise ValueError("closed RSK decision envelope required")
        if type(value["decision_summary"]) is not str or len(value["decision_summary"]) > 2000:
            raise ValueError("bounded public decision summary required")
        refs = value["memory_refs_used"]
        admitted = self.memory_query.in_context if self.memory_query is not None else set()
        if (
            type(refs) is not list
            or any(type(ref) is not str for ref in refs)
            or not set(refs) <= admitted
        ):
            raise ValueError("only actually admitted context references may be cited")
        decision = value["decision"]
        operation = decision.get("operation") if isinstance(decision, dict) else None
        if operation not in self.request_schema()["discriminator"]["mapping"]:
            raise ValueError("operation unavailable in current RSK arm")
        if operation in {
            "ros_engineer.query_cross_body_memory",
            "ros_engineer.propose_cross_body_memory_repair",
        }:
            CROSS_BODY_CALLS.validate_python(decision)
        else:
            decision = normalize_decision(value)["decision"]
        return self.call(decision, remaining_sec=remaining_sec)

    def call(self, request, *, remaining_sec=None):
        operation = request.get("operation") if isinstance(request, dict) else None
        if operation not in {
            "ros_engineer.query_cross_body_memory",
            "ros_engineer.propose_cross_body_memory_repair",
            "ros_engineer.propose_memory_repair",
        }:
            result = self.base.call(request, remaining_sec=remaining_sec)
        else:
            try:
                call = CROSS_BODY_CALLS.validate_python(request)
                if self.base.finished:
                    raise ValueError("episode finished")
                self.base._bound(call.arguments.case_hash, read_only=True)
                if self.arm != "E":
                    raise ValueError("audited cross-Body Memory is exclusive to new RSK E")
                if isinstance(call, CrossBodyQueryCall):
                    retrieval = self.memory_query(call.arguments.query)
                    concepts = [self.memory_query.context(key) for key in retrieval["record_ids"]]
                    result = {
                        "status": "HISTORICAL_CONCEPTS_ENTERED_CONTEXT_NO_AUTHORITY",
                        "concepts": concepts,
                        "retrieval": retrieval,
                        "causal_Memory_benefit": "NOT_ESTABLISHED",
                        "current_authority": "NONE",
                    }
                else:
                    self.base._bound(call.arguments.case_hash)
                    plan = self.memory_query.select(
                        call.arguments.memory_ref,
                        self.base.case,
                        self.base.model,
                        self.base.requirements,
                        self.base.tools.staging,
                    )
                    result = {
                        "status": plan["status"],
                        "transfer_plan": plan,
                        "current_authority": "NONE",
                    }
                    if plan["status"] == "CONDITIONAL_CURRENT_TF_TRANSFER_PROPOSAL":
                        previous_ids = set(self.base.proposals)
                        proposal = self.base._dispatch(
                            RepairCall(
                                operation="ros_engineer.propose_repair",
                                arguments=RepairArgs(
                                    case_hash=call.arguments.case_hash,
                                    repair={"kind": "enable_localization_tf"},
                                ),
                            ),
                            remaining_sec=remaining_sec,
                        )
                        if proposal["proposal"] != plan["proposal"]:
                            key = proposal["proposal"]["proposal_id"]
                            if key not in previous_ids:
                                self.base.proposals.pop(key, None)
                            raise ValueError("independent current staging compilation differs")
                        result.update(
                            status="TYPED_CROSS_BODY_STAGING_PROPOSAL_NOT_EXECUTED",
                            proposal=proposal["proposal"],
                        )
            except (ValidationError, ValueError, KeyError, TypeError, IndexError) as exc:
                result = {
                    "status": "REJECTED_TYPED_RSK_ACI_REQUEST",
                    "error": str(exc),
                    "current_authority": "NONE",
                }
        if operation == "ros_engineer.validate_proposal" and "verification" in result:
            record = result["verification"]
            validation_run = record.get("original_result", {}).get("new_world_run")
            probe_runs = {
                row.get("new_world_run")
                for row in self.base.probe_results
                if row.get("fresh_probe_performed") is True
            }
            if validation_run is not None and validation_run in probe_runs:
                record.update(
                    accepted_success=False,
                    evidence_complete=False,
                    acceptance_status="REJECTED_REUSED_READ_ONLY_PROBE_WORLD",
                )
                self.base.results[result["verification_ref"]] = copy.deepcopy(record)
        self.trace.append({"request": copy.deepcopy(request), "result": copy.deepcopy(result)})
        return copy.deepcopy(result)
