"""Declarative manifests and cohort audits for same-model Harness ablations.

Matching declarations never proves actual inference, isolation or robot safety.
An operator must verify the referenced original evidence before dispatch/review.
This module dispatches no work and grants no action authority.
"""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import (
    ConfigDict,
    Field,
    FiniteFloat,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from rosclaw.connectors.ros.verification.model_budget import RegisteredModelBudgetV1
from rosclaw.contracts.common import ContractModel

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
GitCommit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Name = Annotated[str, Field(min_length=1, max_length=256, pattern=r"\S")]
Arm = Literal["A0", "A1", "A2", "A3"]
FEATURES = MappingProxyType(
    {
        "A0": (),
        "A1": ("system_model", "doctor", "capability_readiness"),
        "A2": (
            "system_model",
            "doctor",
            "capability_readiness",
            "resolver",
            "context_compiler",
            "coverage_mission",
        ),
        "A3": (
            "system_model",
            "doctor",
            "capability_readiness",
            "resolver",
            "context_compiler",
            "coverage_mission",
            "evidence_gated_memory",
        ),
    }
)


class PublicEvidenceRefV1(ContractModel):
    SCHEMA = "rosclaw.fair_ab_evidence.v1"
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["rosclaw.fair_ab_evidence.v1"] = "rosclaw.fair_ab_evidence.v1"
    path: Name
    sha256: Sha256
    size_bytes: StrictInt = Field(ge=0)


class NativeInferenceParamsV1(ContractModel):
    SCHEMA = "rosclaw.fair_ab_native_inference.v1"
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["rosclaw.fair_ab_native_inference.v1"] = (
        "rosclaw.fair_ab_native_inference.v1"
    )
    api: Literal["openai-codex-responses"] = "openai-codex-responses"
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"]
    reasoning_summary: Literal["auto"] = "auto"
    text_verbosity: Literal["low"] = "low"
    parallel_tool_calls: Literal[True] = True
    tool_choice: Literal["auto"] = "auto"
    store: Literal[False] = False
    stream: Literal[True] = True
    include: tuple[Literal["reasoning.encrypted_content"], ...] = ("reasoning.encrypted_content",)

    @field_validator("parallel_tool_calls", "store", "stream", mode="before")
    @classmethod
    def exact_booleans(cls, value):
        if type(value) is not bool:
            raise ValueError("exact Native inference booleans required")
        return value

    def declaration_digest(self) -> str:
        return _json_digest(self.model_dump(mode="json"))


def _json_digest(value: dict) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


class FairRunManifestV1(ContractModel):
    SCHEMA = "rosclaw.fair_ab_run.v1"
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["rosclaw.fair_ab_run.v1"] = "rosclaw.fair_ab_run.v1"
    benchmark_id: Name
    scenario_id: Literal["B1", "B2", "B3", "B4", "B5"]
    robot_id: Name
    arm: Arm
    phase: Literal["pilot", "evaluation"]
    seed: StrictInt = Field(ge=0)
    fixture_hash: Sha256
    world_hash: Sha256
    map_hash: Sha256
    start_pose: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    code_commit: GitCommit
    container_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    ROS_distro: Name
    nav2_revision: Name
    opennav_revision: Name
    model_provider: Name
    exact_model_id: Name
    inference_params: NativeInferenceParamsV1
    inference_params_hash: Sha256
    native_protocol_hash: Sha256
    task_input_hash: Sha256
    prompt_template_hash: Sha256
    initial_repository_hash: Sha256
    daemon_gateway_policy_hash: Sha256
    model_budget: RegisteredModelBudgetV1
    budget_tokens: StrictInt = Field(gt=0)
    wall_budget: StrictInt = Field(gt=0)
    tool_budget: StrictInt = Field(gt=0)
    tool_permissions: tuple[Name, ...]
    feature_flags: tuple[Name, ...]
    memory_mode: Literal["NONE", "EVIDENCE_GATED"]
    permitted_action_mode: Literal["SIM"]
    # Fresh identifiers must describe actual isolated resources, not a shared cache.
    execution_id: Name
    worker_id: Name
    memory_namespace: Name
    # UNKNOWN stays null; absent observations never become zero or PASS.
    human_interventions: StrictInt | None = Field(default=None, ge=0)
    result: Literal["PASS", "FAIL", "UNKNOWN", "NOT_RUN"] = "NOT_RUN"
    coverage_ratio: Annotated[FiniteFloat, Field(ge=0, le=1)] | None = None
    collision_count: StrictInt | None = Field(default=None, ge=0)
    failure_code: Name | None = None
    stop_verified: StrictBool | None = None
    actual_model_id: Name | None = None
    usage_complete: StrictBool | None = None
    evidence_refs: tuple[PublicEvidenceRefV1, ...] = ()

    @field_validator("coverage_ratio", mode="before")
    @classmethod
    def observed_coverage(cls, value):
        if value is not None and type(value) not in (int, float):
            raise ValueError("numeric observed coverage required")
        return value

    @field_validator("start_pose", mode="before")
    @classmethod
    def numeric_pose(cls, value):
        if type(value) not in (tuple, list) or any(type(v) not in (int, float) for v in value):
            raise ValueError("numeric initial pose required")
        return value

    @model_validator(mode="after")
    def check_declared_controls(self):
        if self.feature_flags != FEATURES[self.arm]:
            raise ValueError("arm features must match the registered A0-A3 intervention")
        expected_memory = "EVIDENCE_GATED" if self.arm == "A3" else "NONE"
        if self.memory_mode != expected_memory:
            raise ValueError("only A3 may enable evidence-gated Memory")
        budget = self.model_budget
        if self.inference_params_hash != self.inference_params.declaration_digest():
            raise ValueError("inference parameters must match their original declaration hash")
        if self.inference_params.reasoning_effort != budget.reasoning_effort:
            raise ValueError("inference effort must match the registered response budget")
        if self.inference_params.include != ("reasoning.encrypted_content",):
            raise ValueError("exact Native inference include parameters required")
        if (self.model_provider, self.exact_model_id) != (budget.provider, budget.model):
            raise ValueError("manifest and response budget must use the same exact model")
        if (self.budget_tokens, self.wall_budget, self.tool_budget) != (
            budget.max_total_tokens,
            budget.wall_budget_sec,
            budget.max_tool_calls,
        ):
            raise ValueError("manifest budget aliases cannot disagree with the release gate")
        if len(self.tool_permissions) != len(set(self.tool_permissions)):
            raise ValueError("unique tool permissions required")
        if self.result == "PASS" and (
            self.coverage_ratio is None
            or self.collision_count is None
            or self.human_interventions is None
            or self.actual_model_id is None
            or self.usage_complete is not True
            or not self.evidence_refs
            or self.failure_code is not None
            or self.stop_verified is not True
            or self.actual_model_id != self.exact_model_id
            or self.collision_count != 0
            or self.coverage_ratio < 0.98
        ):
            raise ValueError(
                "PASS requires observations, actual model and original evidence references"
            )
        if self.result == "FAIL" and (not self.failure_code or not self.evidence_refs):
            raise ValueError("failed runs must retain failure code and evidence")
        return self

    def declaration_digest(self) -> str:
        return _json_digest(self.model_dump(mode="json"))


def audit_fair_ab_cohort(
    runs: tuple[FairRunManifestV1, ...],
    *,
    expected_seeds: tuple[int, ...],
    phase: Literal["pilot", "evaluation"],
    prior_seeds: tuple[int, ...] = (),
) -> dict:
    """Require every arm/seed including failures; do not drop unsuccessful runs.

    This audits supplied declarations only. It never verifies file bytes, runtime
    feature switches, provider inference, fresh Memory or daemon receipts.
    """
    required = 5 if phase == "pilot" else 10 if phase == "evaluation" else 0
    if (
        not required
        or len(expected_seeds) != required
        or any(type(seed) is not int or seed < 0 for seed in expected_seeds)
        or len(set(expected_seeds)) != required
    ):
        raise ValueError("exactly five pilot or ten evaluation fresh unique seeds required")
    if any(type(seed) is not int or seed < 0 for seed in prior_seeds):
        raise ValueError("typed previous experimental seeds required")
    if set(expected_seeds) & set(prior_seeds):
        raise ValueError("new experimental seeds cannot reuse prior pilot or tuning seeds")
    expected = {(arm, seed) for arm in FEATURES for seed in expected_seeds}
    observed = set()
    identities: dict[str, set[str]] = {
        name: set() for name in ("execution_id", "worker_id", "memory_namespace")
    }
    shared = None
    per_seed: dict[int, dict[str, Any]] = {}
    paired_fields = {
        "fixture_hash",
        "world_hash",
        "map_hash",
        "start_pose",
        "initial_repository_hash",
        "task_input_hash",
    }
    digests = []
    failures = []
    incomplete = []
    human = []
    # Only arm switches, isolated identities and measured outcomes may differ.
    variable = {
        "arm",
        "feature_flags",
        "memory_mode",
        "execution_id",
        "worker_id",
        "memory_namespace",
        "human_interventions",
        "result",
        "coverage_ratio",
        "collision_count",
        "failure_code",
        "actual_model_id",
        "stop_verified",
        "usage_complete",
        "evidence_refs",
        "seed",
    }
    for supplied in runs:
        run = FairRunManifestV1.model_validate(supplied.model_dump())
        key = (run.arm, run.seed)
        if key not in expected or key in observed or run.phase != phase:
            raise ValueError("one unreplaced run per registered arm and seed required")
        observed.add(key)
        for name, values in identities.items():
            value = getattr(run, name)
            if value in values:
                raise ValueError("worker, execution and Memory namespace must be isolated per run")
            values.add(value)
        controls = {k: v for k, v in run.model_dump(mode="json").items() if k not in variable}
        pair_controls = {k: v for k, v in controls.items() if k in paired_fields}
        fixed_controls = {k: v for k, v in controls.items() if k not in paired_fields}
        if run.seed in per_seed and pair_controls != per_seed[run.seed]:
            raise ValueError("paired initial conditions drifted")
        per_seed[run.seed] = pair_controls
        if shared is None:
            shared = fixed_controls
        elif fixed_controls != shared:
            raise ValueError("same-model cohort controls drifted")
        if run.actual_model_id is not None and run.actual_model_id != run.exact_model_id:
            raise ValueError("actual model drift cannot be called a same-model comparison")
        if any(
            getattr(run, name).strip().upper() == "UNKNOWN"
            for name in (
                "ROS_distro",
                "nav2_revision",
                "opennav_revision",
                "model_provider",
                "exact_model_id",
            )
        ):
            incomplete.append({"arm": run.arm, "seed": run.seed, "reason": "UNKNOWN_CONTROL"})
        if run.result not in ("PASS", "FAIL") or run.human_interventions is None:
            incomplete.append(
                {"arm": run.arm, "seed": run.seed, "reason": "INCOMPLETE_OBSERVATIONS"}
            )
        human.append({"arm": run.arm, "seed": run.seed, "count": run.human_interventions})
        if run.result == "FAIL":
            failures.append({"arm": run.arm, "seed": run.seed, "failure_code": run.failure_code})
        digests.append(run.declaration_digest())
    if observed != expected:
        raise ValueError(
            "all registered arms and seeds required; missing or failed runs cannot be dropped"
        )
    return {
        "status": "DECLARATIONS_MATCH" if not incomplete else "INCOMPLETE_DECLARATIONS",
        "scope": "DECLARATIONS_ONLY_NOT_EXECUTION_ACCEPTANCE",
        "phase": phase,
        "expected_runs": len(expected),
        "recorded_failures": failures,
        "incomplete": incomplete,
        "observed_human_interventions": human,
        "run_declaration_sha256": digests,
        "evidence_bytes_verified": False,
        "causal_benefit": "NOT_MEASURED",
        "dispatch_authorized": False,
        "robot_authorization": False,
    }
