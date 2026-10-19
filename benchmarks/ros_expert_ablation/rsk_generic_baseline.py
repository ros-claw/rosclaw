"""New same-model generic file baseline; no Expert diagnosis or repair selection.

Only declared original strings and bounded staging operations are exposed. The
existing staging whitelist, application compiler check and independent SDK
boundary remain host policy. This is an external-tool ablation, not Codex product.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Annotated, Literal

import yaml
from pydantic import Field, StrictStr, TypeAdapter

from benchmarks.ros_expert_ablation.aci_schema import normalize_decision
from benchmarks.ros_expert_ablation.physical_aci import Args, CaseArgs, PhysicalACI, _staging_files
from benchmarks.ros_expert_ablation.rsk_application_aci import ApplicationPhysicalACI
from benchmarks.ros_expert_ablation.rsk_original_files import FileReadCall, FilesListCall
from benchmarks.ros_expert_ablation.rsk_physical_aci import RskPhysicalACI
from benchmarks.ros_expert_ablation.task_tools import TaskTools
from rosclaw.connectors.ros.context.navigation_workspace_review import _json, _yaml
from rosclaw.connectors.ros.engineering.repair_compiler import compile_repair
from rosclaw.contracts.common import content_hash


class GenericChange(Args):
    file: Literal["nav2.yaml", "bringup.json"]
    pointer: StrictStr = Field(min_length=1, max_length=1024)
    value_json: StrictStr = Field(min_length=1, max_length=8192)


class PatchArgs(CaseArgs):
    changes: list[GenericChange] = Field(min_length=1, max_length=8)


class ApplicationFilesArgs(CaseArgs):
    nav2_yaml: StrictStr = Field(min_length=1, max_length=5_000_000)
    bringup_json: StrictStr = Field(min_length=1, max_length=5_000_000)
    requirements_json: StrictStr = Field(min_length=1, max_length=5_000_000)
    navigation_policy_json: StrictStr = Field(min_length=1, max_length=5_000_000)


class GenericPatchCall(Args):
    operation: Literal["staging.propose_patch"]
    arguments: PatchArgs


class GenericApplicationCall(Args):
    operation: Literal["staging.propose_application"]
    arguments: ApplicationFilesArgs


GENERIC_CALLS = TypeAdapter(
    Annotated[
        FilesListCall | FileReadCall | GenericPatchCall | GenericApplicationCall,
        Field(discriminator="operation"),
    ]
)
COMMON_CALLS = frozenset(
    {
        "ros_engineer.inspect_task",
        "ros_engineer.read_task_evidence",
        "ros_engineer.expand_task_evidence",
        "ros_engineer.propose_onboarding",
        "ros_engineer.validate_proposal",
        "ros_engineer.explain_result",
        "finish",
    }
)


class GenericRskACI(RskPhysicalACI):
    """A supplies values/files itself; no diagnosed issue, option or repair intent."""

    def __init__(self, base, *, application_inputs=None, application_validator=None, **kwargs):
        if base.tools.arm != "A":
            raise ValueError("generic baseline requires legacy A raw-fact tools")
        if application_validator is not None and application_inputs is None:
            raise ValueError("application callback requires frozen original application inputs")
        self.application = None
        if application_inputs is not None:
            self.application = ApplicationPhysicalACI(
                base,
                arm="A",
                application_inputs=application_inputs,
                application_validator=application_validator,
                **kwargs,
            )
        super().__init__(base, arm="A", **kwargs)
        self._original_files()

    @property
    def registered_application_inputs(self):
        return None if self.application is None else self.application.registered_application_inputs

    def _extra_original_files(self):
        return {} if self.application is None else self.application.original_input_files()

    def context_projection(self):
        return {
            "scope": "SAME_MODEL_GENERIC_ORIGINAL_FILES_NOT_CODEX_PRODUCT",
            "files": [
                {
                    "file": name,
                    "utf8_bytes": len(text.encode()),
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
                for name, text in sorted(self._original_files().items())
            ],
            "staging_fields": self.base.tools.call("list_files", {})["allowed_staging_fields"],
            "application_file_proposal_available": self.application is not None,
            "application_validation_contract": {
                "scope": "CURRENT_COMPILER_SEMANTIC_EQUIVALENCE_ONLY_NOT_ANY_VALID_ROS_APPLICATION",
                "host_serializes_equivalent_files": True,
                "host_seals_optional_policy_field": "navigation_requirements_hash",
                "policy_id_rule": "bootstrap-source-<source_urdf_sha256 first16 characters>",
                "numeric_types_and_list_order_must_match": True,
                "mismatch_returns_generated_answer": False,
            },
            "current_authority": "NONE",
        }

    def request_schema(self):
        old, new = PhysicalACI.request_schema(), GENERIC_CALLS.json_schema()
        mapping = {k: v for k, v in old["discriminator"]["mapping"].items() if k in COMMON_CALLS}
        mapping.update(new["discriminator"]["mapping"])
        if self.application is None:
            mapping.pop("staging.propose_application")
        definitions = copy.deepcopy(old["$defs"])
        for key, value in new["$defs"].items():
            if key in definitions and definitions[key] != value:
                raise ValueError("ambiguous generic request arguments")
            definitions[key] = value
        return {
            "$defs": definitions,
            "oneOf": [{"$ref": v} for v in mapping.values()],
            "discriminator": {"propertyName": "operation", "mapping": mapping},
        }

    def call_model_decision(self, value, *, remaining_sec=None):
        if (
            not isinstance(value, dict)
            or set(value) != {"decision", "decision_summary", "memory_refs_used"}
            or type(value["decision_summary"]) is not str
            or len(value["decision_summary"]) > 2000
        ):
            raise ValueError("closed bounded generic model decision required")
        if value["memory_refs_used"] != []:
            raise ValueError("generic baseline cannot cite Memory")
        request = copy.deepcopy(value["decision"])
        if (
            not isinstance(request, dict)
            or request.get("operation") not in self.request_schema()["discriminator"]["mapping"]
        ):
            raise ValueError("Expert operations unavailable in generic baseline")
        if request["operation"] in GENERIC_CALLS.json_schema()["discriminator"]["mapping"]:
            if request["operation"] == "files.read" and isinstance(request.get("arguments"), dict):
                for name in ["offset", "limit"]:
                    if request["arguments"].get(name, False) is None:
                        request["arguments"].pop(name)
            GENERIC_CALLS.validate_python(request)
        else:
            request = normalize_decision(value)["decision"]
        return self.call(request, remaining_sec=remaining_sec)

    def _patch(self, args):
        changes = [
            {"file": c.file, "pointer": c.pointer, "value": json.loads(c.value_json)}
            for c in args.changes
        ]
        if len({(c["file"], c["pointer"]) for c in changes}) != len(changes):
            raise ValueError("duplicate generic patch targets")
        if any(
            c["file"] == "bringup.json" and c["pointer"] == "/additional_command_endpoint/enabled"
            for c in changes
        ):
            if len(self.base.ownership_reviews) != 1:
                raise ValueError("original operator ownership proof required for endpoint mutation")
            # Enforce the same protected-route ownership checks as the typed
            # path; the model still supplies its own target and value.
            compile_repair(
                self.base.case,
                self.base.model,
                self.base.requirements,
                self.base.tools.staging,
                {
                    "kind": "disable_owned_fixture_endpoint",
                    "ownership_ref": next(iter(self.base.ownership_reviews)),
                },
                minimum_radius_m=self.base.tools.minimum_radius_m,
                ownership_reviews=self.base.ownership_reviews,
            )
        trial = TaskTools(
            _staging_files(self.base.tools),
            arm="A",
            reviewed_at=self.base.tools.reviewed_at,
            minimum_radius_m=self.base.tools.minimum_radius_m,
        )
        outcome = trial.call("propose_patch", {"changes": changes})
        actual = [
            {"file": c["file"], "pointer": c["pointer"], "before": c["before"], "value": c["after"]}
            for c in outcome["changes"]
            if c["before"] != c["after"]
        ]
        if not actual:
            raise ValueError("no effective staged change; not a valid repair proposal")
        payload = {
            "schema_version": "rosclaw.rsk_generic_staging_proposal.v1",
            "case_hash": self.base.case.case_hash,
            "snapshot_hash": self.base.case.snapshot_hash,
            "source_urdf_sha256": self.base.case.source_urdf_sha256,
            "base_staging_hash": content_hash("acistaging", self.base.tools.staging),
            "requirements_hash": content_hash("acirequirements", self.base.requirements),
            "intent": {"kind": "GENERIC_CALLER_SUPPLIED_PATCH_NOT_EXPERT_REPAIR"},
            "changes": actual,
            "rollback": [
                {"file": c["file"], "pointer": c["pointer"], "value": c["before"]}
                for c in reversed(actual)
            ],
            "stage_only": True,
            "current_authority": "NONE",
            "admission": "NOT_ADMITTED",
        }
        proposal = {"proposal_id": content_hash("aciproposal", payload), **payload}
        self.base.proposals[proposal["proposal_id"]] = copy.deepcopy(proposal)
        return {
            "status": "GENERIC_STAGING_PROPOSAL_NOT_EXECUTED",
            "proposal": proposal,
            "current_authority": "NONE",
        }

    def _application(self, args):
        if self.application is None:
            raise ValueError("no original application inputs")
        files = {
            "nav2.yaml": args.nav2_yaml,
            "bringup.json": args.bringup_json,
            "requirements.json": args.requirements_json,
            "navigation-policy.json": args.navigation_policy_json,
        }
        plan = self.application.compile_plan()
        if plan["status"] != "CONDITIONAL_DRIVER_BOOTSTRAP_STAGING":
            raise ValueError("no supported current application contract; no proposal registered")
        supplied = {
            name: _yaml(text) if name.endswith(".yaml") else _json(text)
            for name, text in files.items()
        }
        expected = {
            name: _yaml(text) if name.endswith(".yaml") else _json(text)
            for name, text in plan["files"].items()
        }
        policy = supplied["navigation-policy.json"]
        if isinstance(policy, dict) and "navigation_requirements_hash" not in policy:
            policy["navigation_requirements_hash"] = content_hash(
                "navrequirements", supplied["requirements.json"]
            )

        # Preserve scalar types (including int vs double and bool vs number),
        # while accepting dictionary order and whitespace differences. Models
        # never have to calculate a cryptographic digest or a serializer format.
        def canonical(value):
            return json.dumps(value, sort_keys=True, allow_nan=False)

        if canonical(supplied) != canonical(expected):
            raise ValueError(
                "generic application files do not satisfy the same frozen application output contract; no proposal registered"
            )
        result = self.application._propose(
            {
                "operation": "ros_engineer.propose_application_bootstrap",
                "arguments": {"case_hash": args.case_hash},
            }
        )
        result["generic_caller_supplied_files"] = True
        result["generic_original_file_sha256"] = {
            name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()
        }
        result["host_sealed_metadata_and_serialization_only"] = True
        result["compiler_role"] = "HOST_POLICY_VALIDATION_ONLY_NOT_MODEL_ANSWER"
        return result

    def call(self, request, *, remaining_sec=None):
        operation = request.get("operation") if isinstance(request, dict) else None
        if operation in COMMON_CALLS:
            if (
                self.application is not None
                and operation == "ros_engineer.validate_proposal"
                and request.get("arguments", {}).get("proposal_id")
                in self.application.application_proposals
            ):
                result = self.application.call(request, remaining_sec=remaining_sec)
                self.trace.append(
                    {"request": copy.deepcopy(request), "result": copy.deepcopy(result)}
                )
                return result
            return super().call(request, remaining_sec=remaining_sec)
        try:
            call = GENERIC_CALLS.validate_python(request)
            self.base._bound(
                call.arguments.case_hash, read_only=isinstance(call, (FilesListCall, FileReadCall))
            )
            if self.base.finished:
                raise ValueError("episode finished")
            if isinstance(call, FilesListCall):
                result = self.context_projection()
            elif isinstance(call, FileReadCall):
                args = call.arguments
                text = self._original_files()[args.file]
                result = {
                    "status": "ORIGINAL_FILE_SLICE_NOT_DIAGNOSIS",
                    "file": args.file,
                    "offset": args.offset,
                    "total_characters": len(text),
                    "text": text[args.offset : args.offset + args.limit],
                    "truncated": args.offset + args.limit < len(text),
                    "current_authority": "NONE",
                }
            elif isinstance(call, GenericPatchCall):
                result = self._patch(call.arguments)
            else:
                result = self._application(call.arguments)
        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            RecursionError,
            yaml.YAMLError,
        ) as error:
            result = {
                "status": "REJECTED_GENERIC_BASELINE_REQUEST",
                "error": str(error),
                "current_authority": "NONE",
            }
        self.trace.append({"request": copy.deepcopy(request), "result": copy.deepcopy(result)})
        return copy.deepcopy(result)
