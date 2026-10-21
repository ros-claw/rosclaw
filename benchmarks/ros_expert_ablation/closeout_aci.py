"""Thin final-closeout report facade over existing proposal-only PhysicalACI.

Reports are model assertions, independently scored later. They never establish
current L1, execution authority, or success. No evaluator may be attached.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

from pydantic import Field, TypeAdapter

from benchmarks.ros_expert_ablation.aci_schema import _project
from benchmarks.ros_expert_ablation.physical_aci import Args, CaseArgs, PhysicalACI
from benchmarks.ros_expert_ablation.rsk_physical_aci import RskPhysicalACI


class ReportArgs(CaseArgs):
    decision: Literal["PROPOSE", "ABSTAIN", "NEEDS_EVIDENCE", "NO_REPAIR_NEEDED"]
    reason: Literal[
        "CONDITIONAL_PROPOSAL",
        "SOURCE_MISMATCH",
        "MISSING_OR_STALE_EVIDENCE",
        "AUTHORITY_CONFLICT",
        "UNSUPPORTED_OR_AMBIGUOUS",
        "MULTIPLE_BLOCKERS",
        "NO_VIOLATIONS",
    ]
    evidence_refs: list[str] = Field(min_length=1, max_length=16)
    proposal_id: str | None = None


class ReportCall(Args):
    operation: Literal["ros_engineer.report_decision"]
    arguments: ReportArgs


REPORT = TypeAdapter(ReportCall)
ALLOWED = {
    "finish",
    "ros_engineer.inspect_task",
    "ros_engineer.read_task_evidence",
    "ros_engineer.expand_task_evidence",
    "ros_engineer.propose_repair",
    "ros_engineer.propose_compound_repair",
    "ros_engineer.select_staging_options",
    "files.list",
    "files.read",
}


class CloseoutACI:
    def __init__(self, base: PhysicalACI):
        if type(base) is not PhysicalACI or base.tools.arm != "B":
            raise ValueError("existing B PhysicalACI required")
        self.base = base
        self.delegate = RskPhysicalACI(base, arm="B")
        self.trace: list[dict] = []
        self.report = None
        self.visible_refs = {ref for issue in base.case.constraints for ref in issue.evidence_refs}
        self._bound()

    def _bound(self):
        if self.base.evaluator is not None or self.base.probe_evaluator is not None:
            raise ValueError("closeout decision facade cannot attach physical/probe evaluators")
        if self.base.validation_count or self.base.probe_count or self.base.results:
            raise ValueError("decision-only closeout cannot contain physical outcomes")
        self.base._bound(self.base.case.case_hash, read_only=True)

    def decision_schema(self):
        self._bound()
        schema = self.delegate.decision_schema()
        choices = schema["properties"]["decision"]["anyOf"]
        choices[:] = [
            choice
            for choice in choices
            if schema["$defs"][choice["$ref"].rsplit("/", 1)[1]]["properties"]["operation"]["enum"][
                0
            ]
            in ALLOWED
        ]
        report = _project(REPORT.json_schema())
        definitions = report.pop("$defs")
        definitions["ReportArgs"]["properties"]["case_hash"]["enum"] = [self.base.case.case_hash]
        definitions["ReportArgs"]["properties"]["evidence_refs"]["items"]["enum"] = sorted(
            self.visible_refs
        )
        definitions["ReportCall"] = report
        schema["$defs"].update(definitions)
        choices.append({"$ref": "#/$defs/ReportCall"})
        return schema

    def call_model_decision(self, value):
        self._bound()
        if self.report is not None:
            raise ValueError("closeout report is terminal")
        if type(value) is not dict or set(value) != {
            "decision",
            "decision_summary",
            "memory_refs_used",
        }:
            raise ValueError("closed decision envelope required")
        if (
            type(value["decision_summary"]) is not str
            or len(value["decision_summary"]) > 2000
            or value["memory_refs_used"] != []
        ):
            raise ValueError("bounded public summary and no Memory citations required")
        request = value["decision"]
        if type(request) is not dict:
            raise ValueError("typed decision object required")
        operation = request.get("operation")
        if operation == "ros_engineer.report_decision":
            call = REPORT.validate_python(request)
            args = call.arguments
            if args.case_hash != self.base.case.case_hash:
                raise ValueError("current case required")
            if not set(args.evidence_refs) <= self.visible_refs:
                raise ValueError("only actually exposed evidence refs may be cited")
            if args.decision == "PROPOSE":
                if args.proposal_id not in self.base.proposals:
                    raise ValueError("PROPOSE report requires a host-accepted actual proposal")
            elif args.proposal_id is not None:
                raise ValueError("non-proposal reports cannot cite a proposal id")
            self.report = args.model_dump(mode="json")
            result = {
                "status": "MODEL_REPORT_NOT_VERIFICATION",
                "report": self.report,
                "current_authority": "NONE",
                "current_l1": "NOT_EVALUATED",
                "verified_success": False,
            }
        elif operation in ALLOWED:
            result = self.delegate.call_model_decision(value)
            self.visible_refs.update(row["evidence_ref"] for row in result.get("records", []))
        else:
            raise ValueError("operation outside closeout proposal-only allowlist")
        self._bound()
        self.trace.append({"request": deepcopy(request), "result": deepcopy(result)})
        return deepcopy(result)
