"""Operational hypotheses/probes composed with the new RSK interfaces, not G5.

Graph plans are read-only and unexecuted. A new owned probe needs original entry,
source/run binding, a shared budget slot and independent original-file review.
An observational assessment never grants navigation success, L1 or Memory rights.
"""

from __future__ import annotations

import copy
import json
import math
import re
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, StrictStr, TypeAdapter

from benchmarks.ros_expert_ablation.physical_aci import Args, CaseArgs
from benchmarks.ros_expert_ablation.rsk_physical_aci import RskPhysicalACI
from benchmarks.ros_expert_ablation.rsk_pilot_protocol import review_pilot_entry
from rosclaw.connectors.ros.engineering.diagnostic_experiment import RosDiagnosticExperimentV1
from rosclaw.connectors.ros.engineering.probe_planner import plan_diagnostic_probes
from rosclaw.connectors.ros.intelligence.graph_contracts import RosOperationalGraphV1
from rosclaw.connectors.ros.intelligence.operational_graph import compile_operational_graph


class GraphPageArgs(CaseArgs):
    cursor: int = Field(default=0, ge=0)
    limit: int = Field(default=8, ge=1, le=16)


class GraphPlanArgs(CaseArgs):
    issue_ids: list[StrictStr] = Field(min_length=1, max_length=8)
    budget_sec: int = Field(default=20, ge=10, le=30)
    max_probes: int = Field(default=4, ge=1, le=8)


class RunGraphProbeArgs(CaseArgs):
    experiment_hash: StrictStr = Field(min_length=1, max_length=128)
    probe_id: StrictStr = Field(min_length=1, max_length=128)


class InspectOperationalCall(Args):
    operation: Literal["ros_engineer.inspect_operational_graph"]
    arguments: GraphPageArgs


class PlanOperationalCall(Args):
    operation: Literal["ros_engineer.plan_diagnostic_experiment"]
    arguments: GraphPlanArgs


class RunOperationalProbeCall(Args):
    operation: Literal["ros_engineer.run_diagnostic_probe"]
    arguments: RunGraphProbeArgs


GRAPH_CALLS = TypeAdapter(
    Annotated[
        InspectOperationalCall | PlanOperationalCall | RunOperationalProbeCall,
        Field(discriminator="operation"),
    ]
)


def original_operational_graph(base):
    """Host fact binding shared by all arms; this grants no model-facing tool."""
    base._bound(base.case.case_hash, read_only=True)
    return compile_operational_graph(
        base.model,
        base.tools.files["robot.urdf"].encode(),
        base.requirements,
        source_ref="sealed-episode-native:" + base.model.snapshot_hash,
        now=datetime.fromisoformat(base.case.reviewed_at),
        evidence_domain="SEALED_REPLAY",
        source_review=base.case.derived_documents["source_review"],
        expected_run_id=base.case.run_id,
        expected_body_hash=base.case.body_hash,
    )


class OperationalRskACI(RskPhysicalACI):
    def __init__(
        self,
        delegate,
        *,
        diagnostic_probe_provider=None,
        pilot_protocol=None,
        pilot_evidence=None,
        evidence_root=None,
    ):
        if not isinstance(delegate, RskPhysicalACI) or delegate.arm not in {"B", "C", "E"}:
            raise ValueError("operational diagnostics compose only new RSK B/C/E")
        self.delegate = delegate
        self.base, self.arm, self.memory_query = delegate.base, delegate.arm, delegate.memory_query
        self.world_budget = copy.deepcopy(delegate.world_budget)
        self.trace = []
        self.diagnostic_plans = {}
        self.diagnostic_results = []
        self.provider = diagnostic_probe_provider
        self._diagnostic_only_provider = False
        self._entry_args = (pilot_protocol, pilot_evidence, evidence_root)
        self.pilot_entry_review = delegate.pilot_entry_review
        if self.provider is not None:
            from benchmarks.ros_expert_ablation.rsk_diagnostic_provider import (
                DiagnosticProbeProvider,
            )

            if not isinstance(self.provider, DiagnosticProbeProvider) or self.arm not in {"C", "E"}:
                raise ValueError("only C/E may bind the closed owned diagnostic provider")
            self._provider_entry()
            if self.base.probe_evaluator is None:

                def diagnostic_only_marker(*args, **kwargs):
                    raise ValueError("diagnostic provider requires an explicit operational plan")

                self.base.probe_evaluator = diagnostic_only_marker
                self._diagnostic_only_provider = True
        graph = original_operational_graph(self.base)
        self._graph_json = json.dumps(graph.model_dump(mode="json"), separators=(",", ":"))
        self._graph_hash = graph.graph_hash

    def _graph(self):
        self.base._bound(self.base.case.case_hash, read_only=True)
        graph = RosOperationalGraphV1.from_dict(json.loads(self._graph_json))
        if (
            graph.graph_hash != self._graph_hash
            or graph.snapshot_hash != self.base.case.snapshot_hash
        ):
            raise ValueError("original operational graph changed")
        return graph

    def _provider_entry(self):
        protocol, evidence, root = self._entry_args
        if protocol is None or root is None:
            raise ValueError("original pilot entry required before diagnostic SDK binding")
        entry = review_pilot_entry(protocol, evidence, root=root)
        if entry["status"] != "ELIGIBLE_FOR_BOUNDED_PILOT_ONLY":
            raise ValueError("original diagnostic pilot entry is NO_GO")
        if (
            self.base.operator_source != protocol.harness_source
            or self.provider.policy.source_revision != protocol.harness_source
            or self.provider.policy.original_run_id != self.base.original_run_id
            or not isinstance(self.base.original_run_id, str)
            or re.fullmatch(r"[0-9a-f]{32}", self.base.original_run_id) is None
        ):
            raise ValueError("diagnostic provider source/run differs from frozen original operator")
        self.pilot_entry_review = entry

    @property
    def registered_application_inputs(self):
        return getattr(self.delegate, "registered_application_inputs", None)

    def registered_diagnostic_inputs(self):
        return {
            "schema_version": "rosclaw.rsk_diagnostic_inputs.v1",
            "graph": self._graph().model_dump(mode="json"),
            "provider_policy": None
            if self.provider is None
            else self.provider.policy.model_dump(mode="json"),
        }

    def _extra_original_files(self):
        return self.delegate._extra_original_files()

    def context_projection(self):
        original = getattr(self.delegate, "context_projection", None)
        projection = original() if callable(original) else {}
        graph = self._graph()
        projection["operational_diagnostics"] = {
            "graph_hash": graph.graph_hash,
            "snapshot_hash": graph.snapshot_hash,
            "findings_total": len(graph.findings),
            "inspect_operation": "ros_engineer.inspect_operational_graph",
            "plan_operation": "ros_engineer.plan_diagnostic_experiment"
            if self.arm in {"C", "E"}
            else None,
            "explicit_probe_available": self.arm in {"C", "E"} and self.provider is not None,
            "registered_probe_scope": None if self.provider is None else self.provider.policy.scope,
            "replica_observation_role": None
            if self.provider is None
            else getattr(self.provider.policy, "observation_role", None),
            "minimum_host_probe_budget_sec": None
            if self.provider is None
            else self.provider.policy.minimum_wall_sec,
            "owned_probe_physics_World_cost": None
            if self.provider is None
            else int(self.provider.policy.scope == "OWNED_SOURCE_ONLY_NAV2_PHYSICS_NO_BODY"),
            "registered_experiments": list(self.diagnostic_plans),
            "scope": "SEALED_ORIGINAL_DIAGNOSIS_NOT_CURRENT_L1_OR_CAUSAL_VERIFICATION",
            "current_authority": "NONE",
        }
        return projection

    def request_schema(self):
        schema, extra = self.delegate.request_schema(), GRAPH_CALLS.json_schema()
        if self._diagnostic_only_provider:
            removed = schema["discriminator"]["mapping"].pop("ros_engineer.investigate", None)
            if removed is not None:
                schema["oneOf"] = [
                    choice for choice in schema["oneOf"] if choice.get("$ref") != removed
                ]

        for key, value in extra["$defs"].items():
            if key in schema["$defs"] and schema["$defs"][key] != value:
                raise ValueError("ambiguous operational argument schema")
            schema["$defs"][key] = value
        for operation, ref in extra["discriminator"]["mapping"].items():
            if self.arm == "B" and operation in {
                "ros_engineer.plan_diagnostic_experiment",
                "ros_engineer.run_diagnostic_probe",
            }:
                continue
            schema["discriminator"]["mapping"][operation] = ref
            schema["oneOf"].append({"$ref": ref})
        return schema

    def decision_schema(self):
        schema = super().decision_schema()
        if not self.diagnostic_plans:
            schema["properties"]["decision"]["anyOf"] = [
                choice
                for choice in schema["properties"]["decision"]["anyOf"]
                if not choice.get("$ref", "").endswith("/RunOperationalProbeCall")
            ]
        else:
            ids = list(self.diagnostic_plans)
            probes = [
                p["request"]["id"]
                for plan in self.diagnostic_plans.values()
                for p in plan["probes"]
            ]
            properties = schema["$defs"]["RunGraphProbeArgs"]["properties"]
            properties["experiment_hash"]["enum"] = ids
            if probes:
                properties["probe_id"]["enum"] = probes
        return schema

    def call_model_decision(self, value, *, remaining_sec=None):
        request = value.get("decision") if isinstance(value, dict) else None
        operation = request.get("operation") if isinstance(request, dict) else None
        if operation not in self.request_schema()["discriminator"]["mapping"]:
            raise ValueError("operation unavailable in this source-bound operational interface")
        if operation not in GRAPH_CALLS.json_schema()["discriminator"]["mapping"]:
            result = self.delegate.call_model_decision(value, remaining_sec=remaining_sec)
            self.trace.append({"request": copy.deepcopy(request), "result": copy.deepcopy(result)})
            return result
        if set(value) != {"decision", "decision_summary", "memory_refs_used"} or (
            type(value["decision_summary"]) is not str or len(value["decision_summary"]) > 2000
        ):
            raise ValueError("closed public operational decision required")
        refs = value["memory_refs_used"]
        admitted = self.memory_query.in_context if self.memory_query is not None else set()
        if (
            type(refs) is not list
            or any(type(ref) is not str for ref in refs)
            or not set(refs) <= admitted
        ):
            raise ValueError("only actually admitted Memory references may be cited")
        request = copy.deepcopy(request)
        if isinstance(request.get("arguments"), dict):
            for name in ["cursor", "limit", "budget_sec", "max_probes"]:
                if request["arguments"].get(name, False) is None:
                    request["arguments"].pop(name)
        GRAPH_CALLS.validate_python(request)
        return self.call(request, remaining_sec=remaining_sec)

    def _run_probe(self, args, remaining_sec):
        if self.arm not in {"C", "E"}:
            raise ValueError("active diagnostic probes unavailable in B")
        plan = RosDiagnosticExperimentV1.from_dict(self.diagnostic_plans[args.experiment_hash])
        expected = plan_diagnostic_probes(
            self._graph(),
            issue_ids=sorted({h.issue_id for h in plan.hypotheses}),
            budget_sec=plan.budget_sec,
            max_probes=plan.max_probes,
        )
        if plan.experiment_hash != args.experiment_hash or plan != expected:
            raise ValueError("diagnostic request differs from the original deterministic plan")
        selected = next(p for p in plan.probes if p.request.id == args.probe_id)
        if plan.graph_hash != self._graph().graph_hash or plan.status != "NEEDS_READ_ONLY_PROBE":
            raise ValueError("current bound diagnostic plan required")
        if self.provider is None:
            return {
                "status": "NEEDS_OPERATOR_OWNED_DIAGNOSTIC_PROVIDER",
                "starts_World": False,
                "probe_performed": False,
                "current_authority": "NONE",
            }
        self._provider_entry()
        if self.base.probe_count >= 1:
            raise ValueError("shared one-probe budget exhausted")
        if (
            selected.request.provider != "EXISTING_NATIVE_READ_ONLY_PROBE"
            or selected.request.operation not in self.provider.policy.operations
        ):
            return {
                "status": "MEASUREMENT_PROVIDER_UNAVAILABLE_REMAINS_UNKNOWN",
                "starts_World": False,
                "probe_performed": False,
                "current_authority": "NONE",
            }
        self.provider.check_context(self.base)
        if (
            type(remaining_sec) not in (int, float)
            or not math.isfinite(remaining_sec)
            or not self.provider.policy.minimum_wall_sec <= remaining_sec <= 300
        ):
            raise ValueError("sufficient host wall budget required before consuming a probe slot")
        self.base.probe_count += 1
        result = self.provider.run_and_review(
            plan, selected, self.base, original_graph=self._graph(), remaining_sec=remaining_sec
        )
        self.diagnostic_results.append(copy.deepcopy(result))
        self.base.probe_results.append(copy.deepcopy(result))
        return result

    def call(self, request, *, remaining_sec=None):
        operation = request.get("operation") if isinstance(request, dict) else None
        if operation == "ros_engineer.investigate" and self._diagnostic_only_provider:
            return {
                "status": "NEEDS_LEGACY_READ_ONLY_PROVIDER_NOT_CONSUMED",
                "probe_performed": False,
                "current_authority": "NONE",
            }

        if operation not in GRAPH_CALLS.json_schema()["discriminator"]["mapping"]:
            result = self.delegate.call(request, remaining_sec=remaining_sec)
            self.trace.append({"request": copy.deepcopy(request), "result": copy.deepcopy(result)})
            return result
        try:
            if operation not in self.request_schema()["discriminator"]["mapping"]:
                raise ValueError("operational hypothesis/probe operation unavailable in this arm")
            call = GRAPH_CALLS.validate_python(request)
            self.base._bound(call.arguments.case_hash, read_only=True)
            if self.base.finished:
                raise ValueError("episode finished")
            graph = self._graph()
            if isinstance(call, InspectOperationalCall):
                args = call.arguments
                if args.cursor > len(graph.findings):
                    raise ValueError("finding cursor outside original graph")
                rows = graph.findings[args.cursor : args.cursor + args.limit]
                result = {
                    "status": "SEALED_OPERATIONAL_GRAPH_PAGE_NOT_ADMISSION",
                    "graph_hash": graph.graph_hash,
                    "source_status": graph.source_status,
                    "snapshot_freshness": graph.snapshot_freshness,
                    "findings_total": len(graph.findings),
                    "cursor": args.cursor,
                    "next_cursor": args.cursor + len(rows)
                    if args.cursor + len(rows) < len(graph.findings)
                    else None,
                    "findings": [
                        {
                            "id": f.id,
                            "invariant": f.invariant,
                            "status": f.status,
                            "targets": f.targets[:8],
                            "targets_omitted": max(0, len(f.targets) - 8),
                            "evidence_ref_count": len(f.evidence_refs),
                            "scope": f.scope,
                        }
                        for f in rows
                    ],
                    "current_authority": "NONE",
                }
            elif isinstance(call, PlanOperationalCall):
                args = call.arguments
                if len(self.diagnostic_plans) >= 8:
                    raise ValueError("bounded diagnostic plan registry exhausted")
                if len(set(args.issue_ids)) != len(args.issue_ids):
                    raise ValueError("distinct original graph issue IDs required")
                plan = plan_diagnostic_probes(
                    graph,
                    issue_ids=args.issue_ids,
                    budget_sec=args.budget_sec,
                    max_probes=args.max_probes,
                )
                self.diagnostic_plans[plan.experiment_hash] = plan.model_dump(mode="json")
                result = {
                    "status": "DIAGNOSTIC_PLAN_PROPOSED_NOT_EXECUTED",
                    "experiment": plan.model_dump(mode="json"),
                    "probe_performed": False,
                    "current_authority": "NONE",
                }
            else:
                result = self._run_probe(call.arguments, remaining_sec)
        except (ValueError, TypeError, KeyError, IndexError, StopIteration) as error:
            result = {
                "status": "REJECTED_OPERATIONAL_DIAGNOSTIC_REQUEST",
                "error": str(error),
                "current_authority": "NONE",
            }
        self.trace.append({"request": copy.deepcopy(request), "result": copy.deepcopy(result)})
        return copy.deepcopy(result)
