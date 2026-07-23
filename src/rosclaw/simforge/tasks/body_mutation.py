"""Property-style mutation validation for Effective Body simulation adapters."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rosclaw.body.schema import EffectiveBody
from rosclaw.sandbox.body_adapter import SandboxBodyAdapter
from rosclaw.simforge.budget import BudgetScope, DataBudgetManager


class BodyMutationKind(StrEnum):
    SAFE_JOINT_OFFSET = "safe_joint_offset"
    SAFE_SPEED_REDUCTION = "safe_speed_reduction"
    DISABLE_ACTUATOR = "disable_actuator"
    INVERT_JOINT_LIMIT = "invert_joint_limit"
    NONFINITE_JOINT_LIMIT = "nonfinite_joint_limit"
    NEGATIVE_VELOCITY = "negative_velocity"
    OVERSIZED_CALIBRATION = "oversized_calibration"
    RECURSIVE_RUNTIME_STATE = "recursive_runtime_state"


@dataclass(frozen=True)
class BodyMutation:
    mutation_id: str
    kind: BodyMutationKind
    seed: int
    expected_valid: bool


@dataclass(frozen=True)
class BodyMutationResult:
    mutation: BodyMutation
    accepted: bool
    invariant_errors: tuple[str, ...]
    effective_config_hash: str | None


class BodyInvariantValidator:
    def __init__(self, *, max_calibration_offset_rad: float = 0.5) -> None:
        self.max_calibration_offset_rad = max_calibration_offset_rad

    def validate(self, body: EffectiveBody) -> tuple[str, ...]:
        errors: list[str] = []
        if not body.body_instance_id or not body.eurdf_uri or not body.effective_body_hash:
            errors.append("identity")
        if len(body.joints or {}) > 4096 or len(body.actuators or {}) > 4096:
            errors.append("body_size")
        for name, joint in (body.joints or {}).items():
            if not isinstance(joint, dict):
                errors.append(f"joint:{name}:schema")
                continue
            values = [joint.get(field) for field in ("lower", "upper", "velocity", "effort")]
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            ):
                errors.append(f"joint:{name}:finite")
                continue
            lower, upper, velocity, effort = map(float, values)
            if lower >= upper:
                errors.append(f"joint:{name}:range")
            if velocity <= 0 or effort <= 0:
                errors.append(f"joint:{name}:positive_limits")
        valid_status = {"available", "unavailable", "disabled", "faulty"}
        for name, actuator in (body.actuators or {}).items():
            if not isinstance(actuator, dict) or actuator.get("status") not in valid_status:
                errors.append(f"actuator:{name}:status")
        try:
            calibration = (body.runtime_state or {}).get("calibration", {})
            offsets = calibration.get("joint_offsets", {})
            for name, value in offsets.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or abs(float(value)) > self.max_calibration_offset_rad
                ):
                    errors.append(f"calibration:{name}:offset")
        except (AttributeError, TypeError):
            errors.append("calibration:schema")
        budget = DataBudgetManager().inspect_record(
            body.runtime_state or {}, scope=BudgetScope.EVENT
        )
        if not budget.accepted:
            errors.append("runtime_state:data_budget")
        return tuple(dict.fromkeys(errors))


def generate_body_mutations(*, count: int, seed: int) -> tuple[BodyMutation, ...]:
    if not 100 <= count <= 1000:
        raise ValueError("BodyMutation suite size must be in [100, 1000]")
    rng = random.Random(seed)
    kinds = list(BodyMutationKind)
    result = []
    for index in range(count):
        kind = kinds[index % len(kinds)]
        mutation_seed = rng.randrange(2**31)
        digest = hashlib.sha256(f"{seed}:{index}:{kind.value}:{mutation_seed}".encode()).hexdigest()
        result.append(
            BodyMutation(
                mutation_id="body_mutation_" + digest[:24],
                kind=kind,
                seed=mutation_seed,
                expected_valid=kind
                in {
                    BodyMutationKind.SAFE_JOINT_OFFSET,
                    BodyMutationKind.SAFE_SPEED_REDUCTION,
                    BodyMutationKind.DISABLE_ACTUATOR,
                },
            )
        )
    return tuple(result)


def apply_and_validate_mutation(
    base: EffectiveBody,
    mutation: BodyMutation,
    *,
    validator: BodyInvariantValidator | None = None,
) -> BodyMutationResult:
    body = copy.deepcopy(base)
    if not body.joints or not body.actuators:
        return BodyMutationResult(mutation, False, ("base:missing_joint_or_actuator",), None)
    rng = random.Random(mutation.seed)
    joint_name = sorted(body.joints)[0]
    actuator_name = sorted(body.actuators)[0]
    kind = mutation.kind
    if kind is BodyMutationKind.SAFE_JOINT_OFFSET:
        body.runtime_state.setdefault("calibration", {}).setdefault("joint_offsets", {})[
            joint_name
        ] = rng.uniform(-0.1, 0.1)
    elif kind is BodyMutationKind.SAFE_SPEED_REDUCTION:
        body.joints[joint_name]["velocity"] *= rng.uniform(0.1, 0.9)
    elif kind is BodyMutationKind.DISABLE_ACTUATOR:
        body.actuators[actuator_name]["status"] = "disabled"
    elif kind is BodyMutationKind.INVERT_JOINT_LIMIT:
        body.joints[joint_name]["lower"], body.joints[joint_name]["upper"] = (1.0, -1.0)
    elif kind is BodyMutationKind.NONFINITE_JOINT_LIMIT:
        body.joints[joint_name]["upper"] = math.nan
    elif kind is BodyMutationKind.NEGATIVE_VELOCITY:
        body.joints[joint_name]["velocity"] = -1.0
    elif kind is BodyMutationKind.OVERSIZED_CALIBRATION:
        body.runtime_state.setdefault("calibration", {}).setdefault("joint_offsets", {})[
            joint_name
        ] = 100.0
    elif kind is BodyMutationKind.RECURSIVE_RUNTIME_STATE:
        recursive: dict[str, Any] = {}
        recursive["recovery_hint"] = recursive
        body.runtime_state = recursive
    errors = (validator or BodyInvariantValidator()).validate(body)
    if errors:
        return BodyMutationResult(mutation, False, errors, None)
    body.effective_body_hash = "sha256:" + body.compute_hash()
    config = SandboxBodyAdapter.from_effective_body(body).to_mujoco_config()
    adapter_errors: list[str] = []
    if kind is BodyMutationKind.DISABLE_ACTUATOR and actuator_name not in config.get(
        "disabled_actuators", []
    ):
        adapter_errors.append("adapter:disabled_actuator_not_propagated")
    if kind is BodyMutationKind.SAFE_SPEED_REDUCTION:
        configured = config.get("joint_limits", {}).get(joint_name, {}).get("velocity")
        if configured != body.joints[joint_name]["velocity"]:
            adapter_errors.append("adapter:joint_velocity_not_propagated")
    if kind is BodyMutationKind.SAFE_JOINT_OFFSET:
        configured = config.get("calibration_offsets", {}).get("joint_offsets", {}).get(joint_name)
        expected = body.runtime_state["calibration"]["joint_offsets"][joint_name]
        if configured != expected:
            adapter_errors.append("adapter:calibration_offset_not_propagated")
    if adapter_errors:
        return BodyMutationResult(mutation, False, tuple(adapter_errors), None)
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return BodyMutationResult(
        mutation,
        True,
        (),
        "sha256:" + hashlib.sha256(payload.encode()).hexdigest(),
    )


__all__ = [
    "BodyInvariantValidator",
    "BodyMutation",
    "BodyMutationKind",
    "BodyMutationResult",
    "apply_and_validate_mutation",
    "generate_body_mutations",
]
