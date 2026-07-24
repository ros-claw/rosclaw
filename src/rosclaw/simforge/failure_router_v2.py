"""Evidence-bound failure classification and recovery routing."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class FailureClass(StrEnum):
    SKILL_DEFECT = "SKILL_DEFECT"
    EXECUTION_LAPSE = "EXECUTION_LAPSE"
    PERCEPTION_ERROR = "PERCEPTION_ERROR"
    BODY_MISMATCH = "BODY_MISMATCH"
    ENVIRONMENT_SHIFT = "ENVIRONMENT_SHIFT"
    RUNTIME_FAULT = "RUNTIME_FAULT"
    SAFETY_POLICY_DEFECT = "SAFETY_POLICY_DEFECT"
    IMPOSSIBLE_TASK = "IMPOSSIBLE_TASK"


class Recoverability(StrEnum):
    RECOVERABLE = "RECOVERABLE"
    RETRY_AFTER_OBSERVATION = "RETRY_AFTER_OBSERVATION"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    UNRECOVERABLE = "UNRECOVERABLE"


class FailureRoute(StrEnum):
    MEMORY = "MEMORY"
    KNOW = "KNOW"
    HOW = "HOW"
    AUTO_PARAMETER_PATCH = "AUTO_PARAMETER_PATCH"
    AUTO_TRAJECTORY_PATCH = "AUTO_TRAJECTORY_PATCH"
    SENSE = "SENSE"
    PROVIDER = "PROVIDER"
    BODY = "BODY"
    SKILL_COMPATIBILITY = "SKILL_COMPATIBILITY"
    RUNTIME = "RUNTIME"
    WORKER_MANAGER = "WORKER_MANAGER"
    SANDBOX_SHIELD = "SANDBOX_SHIELD"
    FALSIFICATION = "FALSIFICATION"
    STOP = "STOP"
    HUMAN = "HUMAN"


@dataclass(frozen=True)
class RootCauseCandidate:
    cause: str
    confidence: float

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,95}", self.cause):
            raise ValueError("root cause must be a safe identifier")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("root-cause confidence must be in [0, 1]")


@dataclass(frozen=True)
class FailureObservation:
    """Verifier output consumed by the router; no free-form model judgement."""

    task_id: str
    body_id: str
    expected_body_hash: str
    observed_body_hash: str
    action_id: str
    evidence_refs: tuple[str, ...]
    task_success: bool
    target_error_m: float | None = None
    target_tolerance_m: float | None = None
    object_overshot: bool = False
    estimated_friction: float | None = None
    peak_force_n: float | None = None
    force_limit_n: float | None = None
    execution_contract_valid: bool = True
    perception_target_match: bool = True
    observation_fresh: bool = True
    runtime_healthy: bool = True
    static_gate_allowed: bool = False
    physical_rollout_unsafe: bool = False
    reference_policy_solvable: bool = True
    retryable_environment: bool = True

    def __post_init__(self) -> None:
        if not self.task_id or not self.body_id or not self.action_id:
            raise ValueError("failure observation identity is required")
        for value in (self.expected_body_hash, self.observed_body_hash):
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
                raise ValueError("body hashes must be sha256 identifiers")
        if not self.evidence_refs or any("://" not in ref for ref in self.evidence_refs):
            raise ValueError("failure observations require replayable evidence refs")
        for value in (
            self.target_error_m,
            self.target_tolerance_m,
            self.estimated_friction,
            self.peak_force_n,
            self.force_limit_n,
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError("failure observation metrics must be finite")


@dataclass(frozen=True)
class FailureSignatureV2:
    failure_id: str
    task_id: str
    body_id: str
    action_id: str
    primary_class: FailureClass
    secondary_classes: tuple[str, ...]
    root_cause_candidates: tuple[RootCauseCandidate, ...]
    recoverability: Recoverability
    retry_budget: int
    recommended_route: tuple[FailureRoute, ...]
    evidence_refs: tuple[str, ...]
    classification_basis: tuple[str, ...]
    schema_version: str = "rosclaw.failure_signature.v2"

    def __post_init__(self) -> None:
        if not self.failure_id.startswith("failure_"):
            raise ValueError("failure_id must start with failure_")
        if not 0 <= self.retry_budget <= 3:
            raise ValueError("retry budget must be in [0, 3]")
        if not self.evidence_refs or not self.classification_basis:
            raise ValueError("failure signatures require evidence and classification basis")
        if self.recoverability in {
            Recoverability.UNRECOVERABLE,
            Recoverability.HUMAN_REVIEW_REQUIRED,
        }:
            if self.retry_budget != 0:
                raise ValueError("unrecoverable/human-review failures cannot have a retry budget")
            if FailureRoute.STOP not in self.recommended_route:
                raise ValueError("unrecoverable/human-review failures must route to STOP")
        if self.primary_class is FailureClass.IMPOSSIBLE_TASK and any(
            route.name.startswith("AUTO_") for route in self.recommended_route
        ):
            raise ValueError("impossible tasks cannot route to Auto")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "failure_id": self.failure_id,
            "task_id": self.task_id,
            "body_id": self.body_id,
            "action_id": self.action_id,
            "primary_class": self.primary_class.value,
            "secondary_classes": list(self.secondary_classes),
            "root_cause_candidates": [asdict(item) for item in self.root_cause_candidates],
            "recoverability": {
                "status": self.recoverability.value,
                "retry_budget": self.retry_budget,
            },
            "recommended_route": [route.value for route in self.recommended_route],
            "evidence_refs": list(self.evidence_refs),
            "classification_basis": list(self.classification_basis),
        }

    @property
    def signature_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


class FailureRouterV2:
    """Route verifier facts in safety-first order.

    Runtime, body, perception, and impossible-task causes are checked before
    skill/controller causes so a valid skill is not rewritten for an
    infrastructure or observation failure.
    """

    def route(self, observation: FailureObservation) -> FailureSignatureV2:
        classification = self._classify(observation)
        identity = {
            "task_id": observation.task_id,
            "body_id": observation.body_id,
            "action_id": observation.action_id,
            "primary": classification["primary"].value,
            "evidence": sorted(observation.evidence_refs),
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return FailureSignatureV2(
            failure_id="failure_" + digest[:24],
            task_id=observation.task_id,
            body_id=observation.body_id,
            action_id=observation.action_id,
            primary_class=classification["primary"],
            secondary_classes=classification["secondary"],
            root_cause_candidates=classification["causes"],
            recoverability=classification["recoverability"],
            retry_budget=classification["retry_budget"],
            recommended_route=classification["routes"],
            evidence_refs=observation.evidence_refs,
            classification_basis=classification["basis"],
        )

    def _classify(self, item: FailureObservation) -> dict[str, Any]:
        if not item.reference_policy_solvable:
            return _result(
                FailureClass.IMPOSSIBLE_TASK,
                secondary=("REFERENCE_POLICY_UNSOLVABLE",),
                causes=(RootCauseCandidate("task_not_physically_solvable", 1.0),),
                recoverability=Recoverability.UNRECOVERABLE,
                retry_budget=0,
                routes=(FailureRoute.STOP, FailureRoute.HUMAN),
                basis=("reference_policy_solvable=false",),
            )
        if not item.runtime_healthy or not item.observation_fresh:
            secondary = (
                ("OBSERVATION_STALE",) if not item.observation_fresh else ("RUNTIME_UNHEALTHY",)
            )
            return _result(
                FailureClass.RUNTIME_FAULT,
                secondary=secondary,
                causes=(
                    RootCauseCandidate(
                        "observation_channel_lost"
                        if not item.observation_fresh
                        else "runtime_process_fault",
                        1.0,
                    ),
                ),
                recoverability=Recoverability.RETRY_AFTER_OBSERVATION,
                retry_budget=1,
                routes=(FailureRoute.RUNTIME, FailureRoute.WORKER_MANAGER, FailureRoute.HOW),
                basis=(
                    f"runtime_healthy={str(item.runtime_healthy).lower()}",
                    f"observation_fresh={str(item.observation_fresh).lower()}",
                ),
            )
        if item.expected_body_hash != item.observed_body_hash:
            return _result(
                FailureClass.BODY_MISMATCH,
                secondary=("BODY_HASH_CHANGED",),
                causes=(RootCauseCandidate("effective_body_changed", 1.0),),
                recoverability=Recoverability.HUMAN_REVIEW_REQUIRED,
                retry_budget=0,
                routes=(
                    FailureRoute.STOP,
                    FailureRoute.BODY,
                    FailureRoute.SKILL_COMPATIBILITY,
                ),
                basis=("expected_body_hash!=observed_body_hash",),
            )
        if not item.perception_target_match:
            return _result(
                FailureClass.PERCEPTION_ERROR,
                secondary=("TARGET_MISMATCH",),
                causes=(RootCauseCandidate("target_grounding_mismatch", 0.98),),
                recoverability=Recoverability.RETRY_AFTER_OBSERVATION,
                retry_budget=1,
                routes=(FailureRoute.SENSE, FailureRoute.PROVIDER, FailureRoute.HOW),
                basis=("perception_target_match=false",),
            )
        if item.static_gate_allowed and item.physical_rollout_unsafe:
            return _result(
                FailureClass.SAFETY_POLICY_DEFECT,
                secondary=("STATIC_GATE_FALSE_NEGATIVE",),
                causes=(RootCauseCandidate("midpath_physics_not_checked", 1.0),),
                recoverability=Recoverability.RECOVERABLE,
                retry_budget=2,
                routes=(FailureRoute.SANDBOX_SHIELD, FailureRoute.FALSIFICATION),
                basis=("static_gate_allowed=true", "physical_rollout_unsafe=true"),
            )
        if not item.execution_contract_valid:
            return _result(
                FailureClass.EXECUTION_LAPSE,
                secondary=("EXECUTION_CONTRACT_VIOLATION",),
                causes=(RootCauseCandidate("agent_execution_lapse", 0.95),),
                recoverability=Recoverability.RECOVERABLE,
                retry_budget=1,
                routes=(FailureRoute.MEMORY, FailureRoute.KNOW, FailureRoute.HOW),
                basis=("execution_contract_valid=false",),
            )
        if item.object_overshot:
            causes = [RootCauseCandidate("push_velocity_too_high", 0.91)]
            if item.estimated_friction is not None and item.estimated_friction < 0.35:
                causes.insert(0, RootCauseCandidate("low_surface_friction", 0.96))
            causes.append(RootCauseCandidate("contact_duration_too_long", 0.78))
            return _result(
                FailureClass.ENVIRONMENT_SHIFT,
                secondary=("LOW_FRICTION", "OBJECT_OVERSHOT"),
                causes=tuple(causes),
                recoverability=(
                    Recoverability.RECOVERABLE
                    if item.retryable_environment
                    else Recoverability.HUMAN_REVIEW_REQUIRED
                ),
                retry_budget=3 if item.retryable_environment else 0,
                routes=(
                    (
                        FailureRoute.MEMORY,
                        FailureRoute.KNOW,
                        FailureRoute.HOW,
                        FailureRoute.AUTO_PARAMETER_PATCH,
                    )
                    if item.retryable_environment
                    else (FailureRoute.STOP, FailureRoute.HUMAN)
                ),
                basis=(
                    "independent_verifier.object_overshot=true",
                    f"estimated_friction={item.estimated_friction}",
                ),
            )
        return _result(
            FailureClass.SKILL_DEFECT,
            secondary=("TASK_CRITERIA_NOT_MET",),
            causes=(RootCauseCandidate("skill_policy_inadequate", 0.75),),
            recoverability=Recoverability.RECOVERABLE,
            retry_budget=2,
            routes=(FailureRoute.HOW, FailureRoute.AUTO_TRAJECTORY_PATCH),
            basis=("task_success=false", "no_higher_priority_cause=true"),
        )


@dataclass(frozen=True)
class FailureRouterAcceptanceCase:
    """One labeled, deterministic Failure Router counterfactual."""

    name: str
    expected_class: FailureClass
    observed_class: FailureClass
    expected_routes: tuple[FailureRoute, ...]
    observed_routes: tuple[FailureRoute, ...]
    retry_budget: int
    recoverability: Recoverability

    @property
    def passed(self) -> bool:
        return (
            self.observed_class is self.expected_class
            and self.observed_routes == self.expected_routes
            and 0 <= self.retry_budget <= 3
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expected_class": self.expected_class.value,
            "observed_class": self.observed_class.value,
            "expected_routes": [route.value for route in self.expected_routes],
            "observed_routes": [route.value for route in self.observed_routes],
            "retry_budget": self.retry_budget,
            "recoverability": self.recoverability.value,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class FailureRouterAcceptanceReport:
    """Aggregate acceptance metrics for all eight Phase 3 failure classes."""

    cases: tuple[FailureRouterAcceptanceCase, ...]
    schema_version: str = "rosclaw.failure_router_acceptance.v1"

    @property
    def failure_capture_rate(self) -> float:
        return len(self.cases) / len(FailureClass)

    @property
    def routing_accuracy(self) -> float:
        return sum(case.passed for case in self.cases) / len(self.cases)

    @property
    def infinite_retry_count(self) -> int:
        return sum(not 0 <= case.retry_budget <= 3 for case in self.cases)

    @property
    def unrecoverable_stop_rate(self) -> float:
        stopped = [
            FailureRoute.STOP in case.observed_routes
            for case in self.cases
            if case.recoverability
            in {
                Recoverability.HUMAN_REVIEW_REQUIRED,
                Recoverability.UNRECOVERABLE,
            }
        ]
        return sum(stopped) / len(stopped)

    @property
    def passed(self) -> bool:
        return (
            len(self.cases) == len(FailureClass)
            and self.failure_capture_rate == 1.0
            and self.routing_accuracy == 1.0
            and self.infinite_retry_count == 0
            and self.unrecoverable_stop_rate == 1.0
        )

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "predefined_cases": len(self.cases),
            "failure_capture_rate": self.failure_capture_rate,
            "routing_accuracy": self.routing_accuracy,
            "infinite_retry_count": self.infinite_retry_count,
            "unrecoverable_stop_rate": self.unrecoverable_stop_rate,
            "cases": [case.to_dict() for case in self.cases],
            "passed": self.passed,
        }
        value["report_hash"] = _hash_dict(value)
        return value


def run_failure_router_acceptance_suite() -> FailureRouterAcceptanceReport:
    """Exercise every route against a predefined verifier-fact fixture."""

    zero_hash = "sha256:" + "0" * 64
    one_hash = "sha256:" + "1" * 64
    common: dict[str, Any] = {
        "task_id": "contact_push_v3",
        "body_id": "sim_contact_pusher_v3",
        "expected_body_hash": zero_hash,
        "observed_body_hash": zero_hash,
        "action_id": "action_failure_router_acceptance",
        "evidence_refs": ("receipt://router_acceptance",),
        "task_success": False,
    }
    definitions: tuple[tuple[str, FailureClass, tuple[FailureRoute, ...], dict[str, Any]], ...] = (
        (
            "skill_defect",
            FailureClass.SKILL_DEFECT,
            (FailureRoute.HOW, FailureRoute.AUTO_TRAJECTORY_PATCH),
            {},
        ),
        (
            "execution_lapse",
            FailureClass.EXECUTION_LAPSE,
            (FailureRoute.MEMORY, FailureRoute.KNOW, FailureRoute.HOW),
            {"execution_contract_valid": False},
        ),
        (
            "perception_error",
            FailureClass.PERCEPTION_ERROR,
            (FailureRoute.SENSE, FailureRoute.PROVIDER, FailureRoute.HOW),
            {"perception_target_match": False},
        ),
        (
            "body_mismatch",
            FailureClass.BODY_MISMATCH,
            (FailureRoute.STOP, FailureRoute.BODY, FailureRoute.SKILL_COMPATIBILITY),
            {"observed_body_hash": one_hash},
        ),
        (
            "environment_shift",
            FailureClass.ENVIRONMENT_SHIFT,
            (
                FailureRoute.MEMORY,
                FailureRoute.KNOW,
                FailureRoute.HOW,
                FailureRoute.AUTO_PARAMETER_PATCH,
            ),
            {"object_overshot": True, "estimated_friction": 0.2},
        ),
        (
            "runtime_fault",
            FailureClass.RUNTIME_FAULT,
            (FailureRoute.RUNTIME, FailureRoute.WORKER_MANAGER, FailureRoute.HOW),
            {"observation_fresh": False},
        ),
        (
            "safety_policy_defect",
            FailureClass.SAFETY_POLICY_DEFECT,
            (FailureRoute.SANDBOX_SHIELD, FailureRoute.FALSIFICATION),
            {"static_gate_allowed": True, "physical_rollout_unsafe": True},
        ),
        (
            "impossible_task",
            FailureClass.IMPOSSIBLE_TASK,
            (FailureRoute.STOP, FailureRoute.HUMAN),
            {"reference_policy_solvable": False},
        ),
    )
    router = FailureRouterV2()
    cases = []
    for name, expected_class, expected_routes, updates in definitions:
        signature = router.route(FailureObservation(**{**common, **updates}))
        cases.append(
            FailureRouterAcceptanceCase(
                name=name,
                expected_class=expected_class,
                observed_class=signature.primary_class,
                expected_routes=expected_routes,
                observed_routes=signature.recommended_route,
                retry_budget=signature.retry_budget,
                recoverability=signature.recoverability,
            )
        )
    report = FailureRouterAcceptanceReport(cases=tuple(cases))
    if not report.passed:
        raise RuntimeError("Failure Router V2 acceptance suite failed")
    return report


def _result(
    primary: FailureClass,
    *,
    secondary: tuple[str, ...],
    causes: tuple[RootCauseCandidate, ...],
    recoverability: Recoverability,
    retry_budget: int,
    routes: tuple[FailureRoute, ...],
    basis: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "primary": primary,
        "secondary": secondary,
        "causes": causes,
        "recoverability": recoverability,
        "retry_budget": retry_budget,
        "routes": routes,
        "basis": basis,
    }


def _hash_dict(value: dict[str, Any]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


__all__ = [
    "FailureClass",
    "FailureObservation",
    "FailureRoute",
    "FailureRouterAcceptanceCase",
    "FailureRouterAcceptanceReport",
    "FailureRouterV2",
    "FailureSignatureV2",
    "Recoverability",
    "RootCauseCandidate",
    "run_failure_router_acceptance_suite",
]
