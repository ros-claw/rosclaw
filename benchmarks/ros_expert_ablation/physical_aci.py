"""Externally mediated typed ACI; physical exit stays in existing SDK operator.

No model, ROS, Runtime or process is started here. Validation is an explicit,
budgeted call to the host-supplied existing canonical SDK evaluator.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from typing import Annotated, Literal

import yaml
from pydantic import Field, StrictStr, TypeAdapter, ValidationError

from benchmarks.ros_expert_ablation.common_evidence import CommonEvidence, EvidenceView
from benchmarks.ros_expert_ablation.task_tools import TaskTools
from rosclaw.connectors.ros.engineering import RosEngineeringCaseV1, summarize_case
from rosclaw.connectors.ros.engineering.mechanism import instantiate_mechanism
from rosclaw.connectors.ros.engineering.repair_compiler import (
    Args,
    SemanticRepair,
    compile_repair,
    compile_repair_sequence,
    proposal_staging,
)
from rosclaw.connectors.ros.engineering.staging_options import compile_staging_options
from rosclaw.contracts.common import content_hash


class Empty(Args):
    pass


class CaseArgs(Args):
    case_hash: StrictStr


class InvestigateArgs(CaseArgs):
    issue_id: StrictStr
    check_id: StrictStr


class EvidenceArgs(CaseArgs):
    view: EvidenceView
    cursor: int = Field(default=0, ge=0)


class ExpandArgs(CaseArgs):
    evidence_ref: StrictStr = Field(min_length=1, max_length=256)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=1200, ge=1, le=1600)


class RepairArgs(CaseArgs):
    repair: SemanticRepair


class CompoundRepairArgs(CaseArgs):
    repairs: list[SemanticRepair] = Field(min_length=2, max_length=5)


class SelectOptionsArgs(CaseArgs):
    option_refs: list[StrictStr] = Field(min_length=1, max_length=5)


class MemoryRepairArgs(CaseArgs):
    memory_ref: StrictStr = Field(min_length=1, max_length=256)


class Interfaces(Args):
    mapping: StrictStr
    odometry: StrictStr
    lidar: StrictStr
    navigate_to_pose: StrictStr


class OnboardingArgs(CaseArgs):
    interfaces: Interfaces


class ProposalArgs(CaseArgs):
    proposal_id: StrictStr


class ResultArgs(Args):
    verification_ref: StrictStr


class InspectCall(Args):
    operation: Literal["ros_engineer.inspect_task"]
    arguments: Empty


class InvestigateCall(Args):
    operation: Literal["ros_engineer.investigate"]
    arguments: InvestigateArgs


class EvidenceCall(Args):
    operation: Literal["ros_engineer.read_task_evidence"]
    arguments: EvidenceArgs


class ExpandCall(Args):
    operation: Literal["ros_engineer.expand_task_evidence"]
    arguments: ExpandArgs


class RepairCall(Args):
    operation: Literal["ros_engineer.propose_repair"]
    arguments: RepairArgs


class CompoundRepairCall(Args):
    operation: Literal["ros_engineer.propose_compound_repair"]
    arguments: CompoundRepairArgs


class SelectOptionsCall(Args):
    operation: Literal["ros_engineer.select_staging_options"]
    arguments: SelectOptionsArgs


class MemoryRepairCall(Args):
    operation: Literal["ros_engineer.propose_memory_repair"]
    arguments: MemoryRepairArgs


class OnboardingCall(Args):
    operation: Literal["ros_engineer.propose_onboarding"]
    arguments: OnboardingArgs


class ValidateCall(Args):
    operation: Literal["ros_engineer.validate_proposal"]
    arguments: ProposalArgs


class ExplainCall(Args):
    operation: Literal["ros_engineer.explain_result"]
    arguments: ResultArgs


class FinishCall(Args):
    operation: Literal["finish"]
    arguments: Empty


CALL_ADAPTER = TypeAdapter(
    Annotated[
        InspectCall
        | EvidenceCall
        | ExpandCall
        | InvestigateCall
        | RepairCall
        | CompoundRepairCall
        | SelectOptionsCall
        | MemoryRepairCall
        | OnboardingCall
        | ValidateCall
        | ExplainCall
        | FinishCall,
        Field(discriminator="operation"),
    ]
)


def _staging_files(tools):
    files = dict(tools.files)
    for name, value in tools.staging.items():
        files[name] = (
            yaml.safe_dump(value, sort_keys=False) if name.endswith(".yaml") else json.dumps(value)
        )
    return files


class PhysicalACI:
    """Episode-scoped facade; history and original generic tools remain separate.

    An operator creates this object with the original case/model/files and its
    evaluator. Model arguments cannot provide an evaluator, permission or budget.
    """

    def __init__(
        self,
        case,
        model,
        tools,
        *,
        evaluator=None,
        ownership_reviews=None,
        max_validations=2,
        clock=time.monotonic,
        operator_source=None,
        original_run_id=None,
        probe_evaluator=None,
    ):
        self.case = RosEngineeringCaseV1.from_dict(case.model_dump(mode="json"))
        self.model = type(model).from_dict(model.to_dict())
        if self.case.snapshot_hash != self.model.snapshot_hash:
            raise ValueError("case/model mismatch")
        if (
            hashlib.sha256(tools.files["robot.urdf"].encode()).hexdigest()
            != case.source_urdf_sha256
        ):
            raise ValueError("original URDF differs")
        if type(max_validations) is not int or not 0 <= max_validations <= 2:
            raise ValueError("bounded validation budget required")
        self.tools = tools
        self.requirements = json.loads(tools.files["requirements.json"])
        self.evaluator = evaluator
        self.probe_evaluator = probe_evaluator
        self.probe_count = 0
        self.probe_results = []
        self.operator_source = operator_source
        self.original_run_id = original_run_id
        if (evaluator is not None or probe_evaluator is not None) and not (
            isinstance(operator_source, str)
            and re.fullmatch(r"[0-9a-f]{40}", operator_source)
            and isinstance(original_run_id, str)
            and re.fullmatch(r"[0-9a-f]{32}", original_run_id)
        ):
            raise ValueError("pinned host operator source and original run identity required")
        if (
            case.run_id is not None
            and original_run_id is not None
            and case.run_id != original_run_id
        ):
            raise ValueError("operator/case original run mismatch")
        self.ownership_reviews = copy.deepcopy(ownership_reviews or {})
        self.max_validations = max_validations
        self.proposals = {}
        self.results = {}
        self.validation_count = 0
        self.finished = False
        self.trace = []
        self.clock = clock
        self.original_file_names = tuple(tools.files)
        self.original_files_hash = content_hash(
            "acifiles", {name: tools.files[name] for name in self.original_file_names}
        )
        expected = self.case.derived_documents["navigation_observation_review"]["requirements_hash"]
        if expected != content_hash("navrequirements", self.requirements):
            raise ValueError("case/original requirements mismatch")
        self.common_evidence = CommonEvidence(self)
        self.admitted_mechanisms = {}

    @staticmethod
    def request_schema():
        return CALL_ADAPTER.json_schema()

    def _bound(self, case_hash, *, read_only=False):
        RosEngineeringCaseV1.from_dict(self.case.model_dump(mode="json"))
        if (
            case_hash != self.case.case_hash
            or self.model.compute_snapshot_hash() != self.case.snapshot_hash
        ):
            raise ValueError("stale case revision; inspect current task")
        if (
            content_hash(
                "acifiles", {name: self.tools.files[name] for name in self.original_file_names}
            )
            != self.original_files_hash
        ):
            raise ValueError("original task files changed")
        if self.case.source_status == "MISMATCH" and not read_only:
            raise ValueError("source mismatch; no proposal or validation")

    def call(self, request, *, remaining_sec=None):
        """JSON object arguments, actionable rejection and no implicit test on finish."""
        try:
            call = CALL_ADAPTER.validate_python(request)
            if self.finished:
                raise ValueError("episode finished; no new validation or proposal")
            if isinstance(call.arguments, CaseArgs):
                self._bound(
                    call.arguments.case_hash,
                    read_only=isinstance(call, (EvidenceCall, ExpandCall, InvestigateCall)),
                )
            result = self._dispatch(call, remaining_sec=remaining_sec)
        except (ValidationError, ValueError, KeyError, IndexError, TypeError) as error:
            result = {
                "status": "REJECTED_TYPED_ACI_REQUEST",
                "error": str(error),
                "legal_example": {"operation": "ros_engineer.inspect_task", "arguments": {}},
                "contract": "request_schema() defines operation-specific object arguments",
                "current_authority": "NONE",
            }
        self.trace.append({"request": copy.deepcopy(request), "result": copy.deepcopy(result)})
        return copy.deepcopy(result)

    def conditional_staging_options(self):
        self._bound(self.case.case_hash, read_only=True)
        return compile_staging_options(
            self.case,
            self.model,
            self.requirements,
            self.tools.staging,
            minimum_radius_m=self.tools.minimum_radius_m,
            ownership_reviews=self.ownership_reviews,
        )

    def _dispatch(self, call, *, remaining_sec):
        args = call.arguments
        if isinstance(call, SelectOptionsCall):
            if self.tools.arm not in {"B", "C"}:
                raise ValueError("Expert staging options unavailable in A")
            if len(set(args.option_refs)) != len(args.option_refs):
                raise ValueError("distinct current option references required")
            available = {
                option.option_ref: option
                for option in self.conditional_staging_options()["options"]
            }
            if any(ref not in available for ref in args.option_refs):
                raise ValueError("select an actual current statically compiled staging option")
            repairs = [available[ref].intent.model_dump(mode="json") for ref in args.option_refs]
            selected = (
                RepairCall(
                    operation="ros_engineer.propose_repair",
                    arguments=RepairArgs(case_hash=args.case_hash, repair=repairs[0]),
                )
                if len(repairs) == 1
                else CompoundRepairCall(
                    operation="ros_engineer.propose_compound_repair",
                    arguments=CompoundRepairArgs(case_hash=args.case_hash, repairs=repairs),
                )
            )
            result = self._dispatch(selected, remaining_sec=remaining_sec)
            result["selected_staging_option_refs"] = args.option_refs
            return result
        if isinstance(call, MemoryRepairCall):
            if self.tools.arm != "C" or args.memory_ref not in self.admitted_mechanisms:
                raise ValueError("select an actually admitted C historical mechanism reference")
            adapted = instantiate_mechanism(
                self.admitted_mechanisms[args.memory_ref],
                self.case,
                self.model,
                self.requirements,
                self.tools.staging,
                minimum_radius_m=self.tools.minimum_radius_m,
                ownership_reviews=self.ownership_reviews,
            )
            result = self._dispatch(
                RepairCall(
                    operation="ros_engineer.propose_repair",
                    arguments=RepairArgs(case_hash=args.case_hash, repair=adapted["intent"]),
                ),
                remaining_sec=remaining_sec,
            )
            result["historical_memory_ref"] = args.memory_ref
            result["historical_experience_ref"] = adapted["historical_experience_ref"]
            result["historical_values_reused"] = False
            return result
        if isinstance(call, EvidenceCall):
            return self.common_evidence.read(args.view, args.cursor)
        if isinstance(call, ExpandCall):
            return self.common_evidence.expand(args.evidence_ref, args.offset, args.limit)
        if isinstance(call, InspectCall):
            card = (
                summarize_case(self.case)
                if self.tools.arm in {"B", "C"}
                else {
                    "case_hash": self.case.case_hash,
                    "scope": self.case.evidence_domain,
                    "source_status": self.case.source_status,
                    "authority": "NONE",
                    "L1": "NOT_EVALUATED",
                    "task": "propose a minimal staged repair or observed interface onboarding and explicitly request independent validation",
                }
            )
            if self.tools.arm in {"B", "C"}:
                compiled = self.conditional_staging_options()
                card["staging_options"] = [
                    {
                        "option_ref": option.option_ref,
                        "issue_id": option.issue_id,
                        "intent": option.intent.model_dump(mode="json"),
                        "original_staging_values": [change.before for change in option.changes],
                        **(
                            {"pending_conditional_checks": option.original_candidate_preconditions}
                            if option.case_candidate_id is None
                            and option.original_candidate_preconditions
                            else {}
                        ),
                    }
                    for option in compiled["options"]
                ]
                if compiled["options"]:
                    card["proof_obligations"] = list(
                        dict.fromkeys(
                            proof
                            for option in compiled["options"]
                            for proof in option.required_new_world_proofs
                        )
                    )
                card["unsupported_staging_mechanisms"] = compiled["blocked"]
                card["staging_phase"] = (
                    "STATIC_DIFF_CHECKED_CONDITIONAL_SELECTION"
                    if compiled["options"]
                    else "NO_SUPPORTED_STAGING_OPTION"
                )
                card["physical_phase"] = "NOT_ADMITTED_REQUIRES_EXPLICIT_NEW_WORLD_VALIDATION"
            return {
                "status": "ENGINEERING_DOSSIER_NOT_ADMISSION",
                "task": card,
                "common_evidence": self.common_evidence.catalog(),
                "base_staging_hash": content_hash("acistaging", self.tools.staging),
                "operator_owned_staging_refs": list(self.ownership_reviews),
                "validation_cost": "HIGH; fresh SDK World, full L1 and canonical goals",
                "needs_operator": self.evaluator is None,
                "current_authority": "NONE",
                "finish_semantics": "report only; never starts a World",
            }
        if isinstance(call, InvestigateCall):
            if self.tools.arm not in {"B", "C"}:
                raise ValueError("expert investigation unavailable in generic arm")
            issue = next((c for c in self.case.constraints if c.id == args.issue_id), None)
            if issue is None or args.check_id not in issue.next_checks:
                raise ValueError("select a declared issue and its allowed read-only check")
            if self.probe_evaluator is not None:
                if self.probe_count >= 1:
                    raise ValueError("host read-only fresh probe budget exhausted")
                if (
                    type(remaining_sec) not in (int, float)
                    or not math.isfinite(remaining_sec)
                    or not 0 < remaining_sec <= 300
                ):
                    raise ValueError("positive host-owned remaining wall budget required")
                self.probe_count += 1
                started = self.clock()
                result = self.probe_evaluator(
                    args.issue_id, args.check_id, remaining_sec=remaining_sec
                )
                if result.get("fresh_probe_performed") is True:
                    ref = result.get("original_capture_ref", {})
                    if not (
                        result.get("source") == self.operator_source
                        and isinstance(result.get("new_world_run"), str)
                        and re.fullmatch(r"[0-9a-f]{32}", result["new_world_run"])
                        and result["new_world_run"] != self.original_run_id
                        and result.get("check_id") == args.check_id
                        and isinstance(ref.get("path"), str)
                        and bool(ref["path"])
                        and isinstance(ref.get("sha256"), str)
                        and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"])
                    ):
                        raise ValueError("fresh read-only probe source/run/original proof mismatch")
                result = copy.deepcopy(result)
                result.update(
                    current_authority="NONE",
                    current_case_L1="NOT_EVALUATED",
                    probe_in_budget=self.clock() - started <= remaining_sec,
                )
                self.probe_results.append(result)
                return result
            return {
                "status": "NEEDS_OPERATOR_READ_ONLY_PROBE",
                "scope": "SEALED_EVIDENCE_ONLY_NOT_NEW_PROBE",
                "issue": issue.model_dump(mode="json"),
                "evidence_refs": {
                    key: self.case.evidence_index[key].model_dump(mode="json")
                    for key in issue.evidence_refs
                },
                "check_id": args.check_id,
                "fresh_probe_performed": False,
                "current_authority": "NONE",
            }
        if isinstance(call, (RepairCall, CompoundRepairCall)):
            compiler = (
                compile_repair_sequence if isinstance(call, CompoundRepairCall) else compile_repair
            )
            intent = (
                [repair.model_dump(mode="json") for repair in args.repairs]
                if isinstance(call, CompoundRepairCall)
                else args.repair.model_dump(mode="json")
            )
            proposal = compiler(
                self.case,
                self.model,
                self.requirements,
                self.tools.staging,
                intent,
                minimum_radius_m=self.tools.minimum_radius_m,
                ownership_reviews=self.ownership_reviews,
            )
            # Preserve the existing operator's actual whitelist and typed field checks.
            trial = TaskTools(
                _staging_files(self.tools),
                arm=self.tools.arm,
                reviewed_at=self.tools.reviewed_at,
                minimum_radius_m=self.tools.minimum_radius_m,
                memory_query=self.tools.memory_query,
            )
            trial.memory_trace = copy.deepcopy(self.tools.memory_trace)
            trial.call(
                "propose_patch",
                {
                    "changes": [
                        {k: change[k] for k in ["file", "pointer", "value"]}
                        for change in proposal["changes"]
                    ]
                },
            )
            if trial.staging != proposal_staging(proposal, self.tools.staging):
                raise ValueError("independent staging policy mismatch")
            self.proposals[proposal["proposal_id"]] = copy.deepcopy(proposal)
            return {
                "status": "TYPED_STAGING_PROPOSAL_NOT_EXECUTED",
                "proposal": proposal,
                "current_authority": "NONE",
            }
        if isinstance(call, OnboardingCall):
            names = {
                role: getattr(args.interfaces, key)
                for role, key in [
                    ("mapping.occupancy_map", "mapping"),
                    ("state.odometry", "odometry"),
                    ("sensing.lidar", "lidar"),
                    ("navigation.navigate_to_pose", "navigate_to_pose"),
                ]
            }
            candidate = self.case.derived_documents["navigation_observation_review"]["candidate"]
            for role, name in names.items():
                observations = candidate["interfaces"].get(role, [])
                if len(observations) != 1 or observations[0]["name"] != name:
                    raise ValueError("four unique original observed interfaces required")
            payload = {
                "case_hash": self.case.case_hash,
                "snapshot_hash": self.case.snapshot_hash,
                "source_urdf_sha256": self.case.source_urdf_sha256,
                "base_staging_hash": content_hash("acistaging", self.tools.staging),
                "operator_owned_staging_refs": list(self.ownership_reviews),
                "requirements_hash": content_hash("acirequirements", self.requirements),
                "changes": [],
                "rollback": [],
                "interface_names": names,
                "stage_only": True,
                "current_authority": "NONE",
                "admission": "NOT_ADMITTED",
            }
            proposal = {"proposal_id": content_hash("aciproposal", payload), **payload}
            self.proposals[proposal["proposal_id"]] = copy.deepcopy(proposal)
            return {
                "status": "OBSERVED_ONBOARDING_PROPOSAL_NOT_BOUND",
                "proposal": proposal,
                "current_authority": "NONE",
            }
        if isinstance(call, ValidateCall):
            if self.evaluator is None:
                raise ValueError("operator SDK evaluator unavailable; proposal remains unexecuted")
            if (
                type(remaining_sec) not in (int, float)
                or not math.isfinite(remaining_sec)
                or not 0 < remaining_sec <= 300
            ):
                raise ValueError("positive host-owned remaining wall budget required")
            if self.validation_count >= self.max_validations:
                raise ValueError("fresh-World validation budget exhausted")
            proposal = copy.deepcopy(self.proposals[args.proposal_id])
            if proposal["case_hash"] != self.case.case_hash or proposal[
                "requirements_hash"
            ] != content_hash("acirequirements", self.requirements):
                raise ValueError("proposal context changed")
            staged = proposal_staging(proposal, self.tools.staging)
            index = self.validation_count
            self.validation_count += 1
            started = self.clock()
            result = self.evaluator(
                staged, proposal.get("interface_names"), index, remaining_sec=remaining_sec
            )

            def valid_ref(ref):
                return (
                    isinstance(ref, dict)
                    and isinstance(ref.get("path"), str)
                    and bool(ref["path"])
                    and isinstance(ref.get("sha256"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]) is not None
                )

            complete = (
                result.get("verification_source") == "INDEPENDENT_ORIGINAL_SDK_REVIEW"
                and result.get("source") == self.operator_source
                and isinstance(result.get("new_world_run"), str)
                and re.fullmatch(r"[0-9a-f]{32}", result["new_world_run"]) is not None
                and result["new_world_run"] != self.original_run_id
                and all(
                    valid_ref(result.get(key))
                    for key in ["fresh_l1_ref", "independent_stop_ref", "strict_source_envelope"]
                )
                and isinstance(result.get("canonical_receipts"), list)
                and len(result["canonical_receipts"]) == 2
                and all(valid_ref(ref) for ref in result["canonical_receipts"])
            )
            in_budget = self.clock() - started <= remaining_sec
            accepted = result.get("verified_success") is True and bool(complete) and in_budget
            status = (
                "ACCEPTED_INDEPENDENT_SIM_RESULT"
                if accepted
                else "UNVERIFIED_OR_FAILED_OR_OUT_OF_BUDGET"
            )
            ref = content_hash(
                "aciverification",
                {"proposal_id": args.proposal_id, "index": index, "result": result},
            )
            record = {
                "original_result": copy.deepcopy(result),
                "acceptance_status": status,
                "accepted_success": accepted,
                "in_budget": in_budget,
                "evidence_complete": bool(complete),
            }
            self.results[ref] = record
            return {
                "status": "EXTERNAL_SDK_VERIFICATION_RECORDED",
                "verification_ref": ref,
                "verification": record,
                "current_authority": "NONE",
                "case_L1": "NOT_EVALUATED",
                "scope": "result belongs to a different SDK World; not reusable authority",
            }
        if isinstance(call, ExplainCall):
            result = self.results[args.verification_ref]
            return {
                "status": "RECORDED_SDK_RESULT_EXPLANATION",
                "verification_ref": args.verification_ref,
                "outcome": copy.deepcopy(result),
                "current_authority": "NONE",
                "missing_phase_details": "UNKNOWN unless present in the original external result",
            }
        self.finished = True
        return {
            "status": "FINISHED_REPORT_ONLY",
            "recorded_verification_refs": list(self.results),
            "validation_calls": self.validation_count,
            "current_authority": "NONE",
            "physical_success_claimed": False,
        }
