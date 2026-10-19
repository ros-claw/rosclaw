"""Application synthesis through the existing Core compiler and SDK boundary.

No ROS, Runtime, driver or executor is created. Four-file application proposals
remain separate from five-mechanism repair materialization; their validation
shares the existing episode's one-World counter and independent evidence rules.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Literal

from pydantic import Field

from benchmarks.ros_expert_ablation.physical_aci import Args, CaseArgs
from benchmarks.ros_expert_ablation.rsk_physical_aci import RskPhysicalACI
from benchmarks.ros_expert_ablation.rsk_sensor_composition import (
    INPUTS_V2,
    SourceSensorComponentV1,
    compose_source_sensor_component,
)
from rosclaw.connectors.ros.engineering.application_bootstrap import compile_driver_bootstrap
from rosclaw.connectors.ros.engineering.staging_plan import parse_staging_documents
from rosclaw.connectors.ros.intelligence.operational_graph import RosOperationalGraphV1
from rosclaw.contracts.common import content_hash

APPLICATION_FILES = frozenset(
    {"nav2.yaml", "bringup.json", "requirements.json", "navigation-policy.json"}
)


class ApplicationBootstrapCall(Args):
    operation: Literal["ros_engineer.propose_application_bootstrap"]
    arguments: CaseArgs


class ApplicationInputArgs(CaseArgs):
    file: Literal[
        "software-template.json",
        "component-manifests.json",
        "driver-bootstrap-request.json",
        "original/nav2.yaml",
        "original/bringup.json",
        "original/requirements.json",
        "original/navigation-policy.json",
    ]
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=12000, ge=1, le=16000)


class ApplicationInputCall(Args):
    operation: Literal["ros_engineer.read_application_input"]
    arguments: ApplicationInputArgs


class ApplicationInputArgsV2(CaseArgs):
    file: Literal[
        "software-template.json",
        "component-manifests.json",
        "driver-bootstrap-request.json",
        "original/nav2.yaml",
        "original/bringup.json",
        "original/requirements.json",
        "original/navigation-policy.json",
        "optional-sensor-component.json",
    ]
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=12000, ge=1, le=16000)


class ApplicationInputCallV2(Args):
    operation: Literal["ros_engineer.read_application_input"]
    arguments: ApplicationInputArgsV2


class ApplicationPhysicalACI(RskPhysicalACI):
    """Optional host-bound application inputs; model supplies only case_hash."""

    def __init__(self, base, *, application_inputs, application_validator=None, **kwargs):
        original_keys = {
            "graph",
            "driver_bringup",
            "software_template",
            "component_manifests",
            "request",
            "original_files",
        }
        self._sensor_composition = (
            type(application_inputs) is dict
            and application_inputs.get("schema_version") == INPUTS_V2
        )
        expected = (
            original_keys | {"schema_version", "sensor_processing"}
            if self._sensor_composition
            else original_keys
        )
        if type(application_inputs) is not dict or set(application_inputs) != expected:
            raise ValueError("closed original application input bundle required")
        raw = json.dumps(
            application_inputs, sort_keys=False, separators=(",", ":"), allow_nan=False
        )
        if not 0 < len(raw.encode()) <= 16_000_000:
            raise ValueError("bounded original application inputs required")
        self._inputs_json = raw
        self._inputs_bytes_sha256 = hashlib.sha256(raw.encode()).hexdigest()
        self._inputs_hash = content_hash("rskappinputs", application_inputs)
        self.application_validator = application_validator
        self.application_proposals = {}
        if application_validator is not None:
            if not (
                isinstance(base.operator_source, str)
                and re.fullmatch(r"[0-9a-f]{40}", base.operator_source)
                and isinstance(base.original_run_id, str)
                and re.fullmatch(r"[0-9a-f]{32}", base.original_run_id)
            ):
                raise ValueError("pinned existing SDK operator source and original run required")
            if base.evaluator is None:
                # Advertise the real validation callback to the inherited entry
                # gate and contract-client guard. Legacy repair dispatch cannot
                # accidentally consume this application-specific callback.
                def application_only_marker(*args, **options):
                    raise ValueError(
                        "application validator requires its typed application proposal"
                    )

                base.evaluator = application_only_marker
        super().__init__(base, **kwargs)
        self._inputs()

    def _inputs(self):
        if hashlib.sha256(self._inputs_json.encode()).hexdigest() != self._inputs_bytes_sha256:
            raise ValueError("original application input bytes/order changed")
        data = json.loads(self._inputs_json)
        if content_hash("rskappinputs", data) != self._inputs_hash:
            raise ValueError("original application inputs changed")
        if self._sensor_composition:
            SourceSensorComponentV1.model_validate(data["sensor_processing"])
        originals = data["original_files"]
        if (
            type(originals) is not dict
            or set(originals) != APPLICATION_FILES
            or any(
                type(value) is not str or not 0 < len(value.encode()) <= 5_000_000
                for value in originals.values()
            )
        ):
            raise ValueError("four exact bounded original application files required")
        staged = parse_staging_documents(
            originals["nav2.yaml"].encode(), originals["bringup.json"].encode()
        )
        if (
            staged != self.base.tools.initial_staging
            or data["driver_bringup"] != staged["bringup.json"]
            or json.loads(originals["requirements.json"]) != self.base.requirements
        ):
            raise ValueError("application originals differ from current original task")
        policy = json.loads(originals["navigation-policy.json"])
        request = data["request"]
        if (
            policy.get("schema_version") != "rosclaw.sdk_navigation_policy_input.v1"
            or policy.get("scope") != "OPERATOR_OWNED_DISPOSABLE_SIM_NEW_BODY"
            or policy.get("source_urdf_sha256") != self.base.case.source_urdf_sha256
            or policy.get("navigation_requirements_hash")
            != content_hash("navrequirements", self.base.requirements)
            or policy.get("navigation_limits") != request.get("navigation_limits")
            or policy.get("body_id") != request.get("body_id")
        ):
            raise ValueError("original Body policy/limits cannot be replaced by a proposal")
        graph = RosOperationalGraphV1.from_dict(data["graph"])
        if graph.snapshot_hash != self.base.case.snapshot_hash:
            raise ValueError("application graph differs from current case snapshot")
        return data, graph

    def context_projection(self):
        data, graph = self._inputs()
        return {
            "schema_version": "rosclaw.rsk_application_context.v1",
            "case_hash": self.base.case.case_hash,
            "snapshot_hash": graph.snapshot_hash,
            "application_inputs_hash": self._inputs_hash,
            "application_input_bytes_sha256": self._inputs_bytes_sha256,
            "proposal_operation": "ros_engineer.propose_application_bootstrap",
            "target_capabilities": [
                "mapping.occupancy_map",
                "localization.pose_estimate",
                "navigation.navigate_to_pose",
                "safety.collision_monitor",
            ],
            "control_reference_to_verify": data["request"]["control"],
            "scope": "HOST_PINNED_APPLICATION_INPUTS_NOT_DRIVER_ACCEPTANCE_OR_BODY_ADMISSION",
            "compile_requested": False,
            "original_input_files": list(self.original_input_files()),
            "starts_World": False,
            "current_authority": "NONE",
        }

    def registered_application_inputs(self):
        return self._inputs()[0]

    def compile_plan(self):
        data, graph = self._inputs()
        raw = self.base.tools.files["robot.urdf"].encode()
        plan = compile_driver_bootstrap(
            graph,
            self.base.model,
            raw,
            data["driver_bringup"],
            data["software_template"],
            data["component_manifests"],
            data["request"],
        )
        return (
            compose_source_sensor_component(plan, self.base.model, raw, data["sensor_processing"])
            if self._sensor_composition
            else plan
        )

    def _extra_original_files(self):
        return self.original_input_files()

    def original_input_files(self):
        data, _ = self._inputs()
        files = {
            **{"original/" + name: text for name, text in data["original_files"].items()},
            **{
                name: json.dumps(data[key], ensure_ascii=False, separators=(",", ":"))
                for name, key in [
                    ("software-template.json", "software_template"),
                    ("component-manifests.json", "component_manifests"),
                    ("driver-bootstrap-request.json", "request"),
                ]
            },
        }
        if self._sensor_composition:
            files["optional-sensor-component.json"] = json.dumps(
                data["sensor_processing"], ensure_ascii=False, separators=(",", ":")
            )
        return files

    def read_application_input(self, request):
        reader = ApplicationInputCallV2 if self._sensor_composition else ApplicationInputCall
        call = reader.model_validate(request)
        self.base._bound(call.arguments.case_hash, read_only=True)
        if self.base.finished:
            raise ValueError("episode finished")
        text = self.original_input_files()[call.arguments.file]
        offset, limit = call.arguments.offset, call.arguments.limit
        return {
            "status": "ORIGINAL_APPLICATION_INPUT_SLICE_NOT_DIAGNOSIS",
            "file": call.arguments.file,
            "offset": offset,
            "total_characters": len(text),
            "text": text[offset : offset + limit],
            "truncated": offset + limit < len(text),
            "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "current_authority": "NONE",
        }

    def request_schema(self):
        schema = super().request_schema()
        reader = ApplicationInputCallV2 if self._sensor_composition else ApplicationInputCall
        for model in [ApplicationBootstrapCall, reader]:
            name = model.__name__
            definition = model.model_json_schema()
            for key, value in definition.pop("$defs", {}).items():
                if key in schema["$defs"] and schema["$defs"][key] != value:
                    raise ValueError("ambiguous application argument definition")
                schema["$defs"][key] = value
            schema["$defs"][name] = definition
            ref = "#/$defs/" + name
            schema["oneOf"].append({"$ref": ref})
            operation = definition["properties"]["operation"]["const"]
            schema["discriminator"]["mapping"][operation] = ref
        return schema

    def call_model_decision(self, value, *, remaining_sec=None):
        if not (
            isinstance(value, dict)
            and isinstance(value.get("decision"), dict)
            and value["decision"].get("operation")
            in {"ros_engineer.propose_application_bootstrap", "ros_engineer.read_application_input"}
        ):
            return super().call_model_decision(value, remaining_sec=remaining_sec)
        if set(value) != {"decision", "decision_summary", "memory_refs_used"} or (
            type(value["decision_summary"]) is not str or len(value["decision_summary"]) > 2000
        ):
            raise ValueError("closed bounded public application decision required")
        refs = value["memory_refs_used"]
        admitted = self.memory_query.in_context if self.memory_query is not None else set()
        if (
            type(refs) is not list
            or any(type(ref) is not str for ref in refs)
            or not set(refs) <= admitted
        ):
            raise ValueError("only actually returned context references may be cited")
        model = (
            (ApplicationInputCallV2 if self._sensor_composition else ApplicationInputCall)
            if value["decision"]["operation"] == "ros_engineer.read_application_input"
            else ApplicationBootstrapCall
        )
        request = copy.deepcopy(value["decision"])
        if model in {ApplicationInputCall, ApplicationInputCallV2} and isinstance(
            request.get("arguments"), dict
        ):
            for name in ["offset", "limit"]:
                if request["arguments"].get(name, False) is None:
                    request["arguments"].pop(name)
        model.model_validate(request)
        return self.call(request, remaining_sec=remaining_sec)

    def _propose(self, request):
        call = ApplicationBootstrapCall.model_validate(request)
        self.base._bound(call.arguments.case_hash)
        if self.base.finished:
            raise ValueError("episode finished")
        data, _ = self._inputs()
        plan = self.compile_plan()
        if plan["status"] != "CONDITIONAL_DRIVER_BOOTSTRAP_STAGING":
            return {
                "status": plan["status"],
                "application_plan": plan,
                "proposal_registered": False,
                "current_authority": "NONE",
            }
        assert set(plan["files"]) == APPLICATION_FILES
        for name, text in plan["files"].items():
            if hashlib.sha256(text.encode()).hexdigest() != plan["file_sha256"][name]:
                raise ValueError("compiled application file hash mismatch")
        original = data["original_files"]
        changes = [
            {
                "file": name,
                "pointer": "",
                "operation": "REPLACE_WHOLE_ORIGINAL_FILE",
                "before": original[name],
                "value": plan["files"][name],
            }
            for name in sorted(APPLICATION_FILES)
            if original[name] != plan["files"][name]
        ]
        payload = {
            "schema_version": "rosclaw.rsk_application_proposal.v1",
            "case_hash": self.base.case.case_hash,
            "snapshot_hash": self.base.case.snapshot_hash,
            "source_urdf_sha256": self.base.case.source_urdf_sha256,
            "original_application_files_hash": content_hash("rskappfiles", original),
            "application_inputs_hash": self._inputs_hash,
            "application_input_bytes_sha256": self._inputs_bytes_sha256,
            "requirements_hash": content_hash("acirequirements", self.base.requirements),
            "intent": {"kind": "compile_driver_application"},
            "changes": changes,
            "rollback": [
                {
                    "file": c["file"],
                    "pointer": "",
                    "operation": c["operation"],
                    "value": c["before"],
                }
                for c in reversed(changes)
            ],
            "files": plan["files"],
            "file_sha256": plan["file_sha256"],
            "compiled_plan_hash": plan["plan_hash"],
            "validation_plan": plan["validation_plan"],
            "interface_names": plan["requirements"]["interface_names"],
            "interface_status": "PLANNED_NOT_OBSERVED_OR_ADMITTED",
            "stage_only": True,
            "current_authority": "NONE",
            "admission": "NOT_ADMITTED",
        }
        proposal = {"proposal_id": content_hash("aciproposal", payload), **payload}
        self.application_proposals[proposal["proposal_id"]] = copy.deepcopy(proposal)
        # Existing context rendering and phase-bound reference schemas see this
        # explicit four-file proposal. Validation is intercepted below; it never
        # reaches five-mechanism JSON-pointer materialization.
        self.base.proposals[proposal["proposal_id"]] = copy.deepcopy(proposal)
        return {
            "status": "CONDITIONAL_APPLICATION_STAGING_PROPOSAL_NOT_EXECUTED",
            "proposal": proposal,
            "current_authority": "NONE",
        }

    def _validate_application(self, request, remaining_sec):
        if self.application_validator is None:
            raise ValueError("operator SDK application evaluator unavailable; proposal unexecuted")
        args = request["arguments"]
        if set(args) != {"case_hash", "proposal_id"}:
            raise ValueError("closed current application validation request required")
        self.base._bound(args["case_hash"])
        if self.base.finished:
            raise ValueError("episode finished")
        if (
            type(remaining_sec) not in (int, float)
            or not math.isfinite(remaining_sec)
            or not 0 < remaining_sec <= 300
        ):
            raise ValueError("positive existing host remaining wall budget required")
        if self.base.validation_count >= 1:
            raise ValueError("shared fresh-World validation budget exhausted")
        data, _ = self._inputs()
        proposal = copy.deepcopy(self.application_proposals[args["proposal_id"]])
        payload = {key: value for key, value in proposal.items() if key != "proposal_id"}
        if (
            proposal["proposal_id"] != content_hash("aciproposal", payload)
            or proposal != self.base.proposals[args["proposal_id"]]
            or proposal["original_application_files_hash"]
            != content_hash("rskappfiles", data["original_files"])
        ):
            raise ValueError("application proposal/original files changed")
        index = self.base.validation_count
        self.base.validation_count += 1
        started = self.base.clock()
        try:
            result = self.application_validator(
                copy.deepcopy(proposal["files"]),
                copy.deepcopy(proposal["interface_names"]),
                index,
                remaining_sec=remaining_sec,
            )
        except Exception as exc:
            raise ValueError(
                "existing SDK application callback failed without retry: "
                + type(exc).__name__
                + ": "
                + str(exc)[:1000]
            ) from exc
        if type(result) is not dict:
            raise ValueError("existing SDK application callback returned no verification record")

        def valid_ref(value):
            return (
                isinstance(value, dict)
                and type(value.get("path")) is str
                and bool(value["path"])
                and type(value.get("sha256")) is str
                and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None
            )

        complete = (
            result.get("verification_source") == "INDEPENDENT_ORIGINAL_SDK_REVIEW"
            and result.get("source") == self.base.operator_source
            and type(result.get("new_world_run")) is str
            and re.fullmatch(r"[0-9a-f]{32}", result["new_world_run"]) is not None
            and result["new_world_run"] != self.base.original_run_id
            and result["new_world_run"]
            not in {
                row.get("new_world_run")
                for row in self.base.probe_results
                if row.get("fresh_probe_performed") is True
            }
            and all(
                valid_ref(result.get(key))
                for key in ["fresh_l1_ref", "independent_stop_ref", "strict_source_envelope"]
            )
            and isinstance(result.get("canonical_receipts"), list)
            and len(result["canonical_receipts"]) == 2
            and all(valid_ref(ref) for ref in result["canonical_receipts"])
        )
        in_budget = self.base.clock() - started <= remaining_sec
        accepted = result.get("verified_success") is True and bool(complete) and in_budget
        ref = content_hash(
            "aciverification",
            {"proposal_id": args["proposal_id"], "index": index, "result": result},
        )
        record = {
            "original_result": copy.deepcopy(result),
            "acceptance_status": "ACCEPTED_INDEPENDENT_SIM_RESULT"
            if accepted
            else "UNVERIFIED_OR_FAILED_OR_OUT_OF_BUDGET",
            "accepted_success": accepted,
            "in_budget": in_budget,
            "evidence_complete": bool(complete),
        }
        self.base.results[ref] = record
        return {
            "status": "EXTERNAL_SDK_APPLICATION_VERIFICATION_RECORDED",
            "verification_ref": ref,
            "verification": record,
            "current_authority": "NONE",
        }

    def call(self, request, *, remaining_sec=None):
        operation = request.get("operation") if isinstance(request, dict) else None
        application_validation = (
            operation == "ros_engineer.validate_proposal"
            and request.get("arguments", {}).get("proposal_id") in self.application_proposals
        )
        if (
            operation
            not in {
                "ros_engineer.propose_application_bootstrap",
                "ros_engineer.read_application_input",
            }
            and not application_validation
        ):
            return super().call(request, remaining_sec=remaining_sec)
        try:
            result = (
                self._validate_application(request, remaining_sec)
                if application_validation
                else self.read_application_input(request)
                if operation == "ros_engineer.read_application_input"
                else self._propose(request)
            )
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            result = {
                "status": "REJECTED_TYPED_APPLICATION_REQUEST",
                "error": str(exc),
                "current_authority": "NONE",
            }
        self.trace.append({"request": copy.deepcopy(request), "result": copy.deepcopy(result)})
        return copy.deepcopy(result)
