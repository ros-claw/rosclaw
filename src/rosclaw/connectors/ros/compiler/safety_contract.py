"""ROS Connector - Safety Contract Compiler.

Classifies ROS capabilities into safety levels and produces a SafetyContract
that the runtime uses to decide ALLOW / MODIFY / BLOCK.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from rosclaw.connectors.ros.compiler.capability_manifest import (
    CapabilityManifest,
    RosCapability,
)


class SafetyLevel:
    READ_ONLY = "READ_ONLY"
    LOW_RISK = "LOW_RISK"
    MEDIUM_RISK = "MEDIUM_RISK"
    HIGH_RISK = "HIGH_RISK"
    FORBIDDEN_BY_DEFAULT = "FORBIDDEN_BY_DEFAULT"


@dataclass
class SandboxDecision:
    """Result of a safety check."""

    decision: str  # ALLOW, MODIFY, BLOCK
    risk_score: float = 0.0
    reason: str = ""
    violated_constraints: list[str] = field(default_factory=list)
    modified_args: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "risk_score": self.risk_score,
            "reason": self.reason,
            "violated_constraints": self.violated_constraints,
            "modified_args": self.modified_args,
        }


@dataclass
class SafetyRule:
    """A single rule in the safety contract."""

    capability_id: str
    level: str
    read_only: bool
    destructive: bool
    requires_sandbox: bool
    requires_runtime_guard: bool
    requires_stop_guard: bool
    max_duration_sec: float | None = None
    max_rate_hz: float | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    forbidden: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "level": self.level,
            "read_only": self.read_only,
            "destructive": self.destructive,
            "requires_sandbox": self.requires_sandbox,
            "requires_runtime_guard": self.requires_runtime_guard,
            "requires_stop_guard": self.requires_stop_guard,
            "max_duration_sec": self.max_duration_sec,
            "max_rate_hz": self.max_rate_hz,
            "constraints": self.constraints,
            "forbidden": self.forbidden,
        }


@dataclass
class SafetyContract:
    """Compiled safety contract for a capability manifest."""

    schema_version: str = "rosclaw.safety_contract.v1"
    robot_id: str = "unknown"
    generated_at: str = ""
    rules: dict[str, SafetyRule] = field(default_factory=dict)

    def get_rule(self, capability_id: str) -> SafetyRule | None:
        return self.rules.get(capability_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "robot_id": self.robot_id,
            "generated_at": self.generated_at,
            "rules": {k: v.to_dict() for k, v in self.rules.items()},
        }


class SafetyContractCompiler:
    """Compile a CapabilityManifest into a SafetyContract."""

    FORBIDDEN_PATTERNS = [
        "torque_control",
        "raw_torque",
        "disable_safety",
        "emergency_stop_disable",
        "controller_config_write",
        "unbounded_velocity",
        "motor_reset",
    ]

    def compile(self, manifest: CapabilityManifest) -> SafetyContract:
        contract = SafetyContract(
            robot_id=manifest.robot_id,
            generated_at=datetime.now(UTC).isoformat(),
        )
        for cap in manifest.capabilities:
            contract.rules[cap.id] = self._compile_rule(cap)
        return contract

    def _compile_rule(self, cap: RosCapability) -> SafetyRule:
        # Start from manifest risk metadata.
        level = self._map_level(cap.risk.level, cap.interface.name)

        forbidden = any(p in cap.interface.name.lower() for p in self.FORBIDDEN_PATTERNS)
        if forbidden:
            level = SafetyLevel.FORBIDDEN_BY_DEFAULT

        return SafetyRule(
            capability_id=cap.id,
            level=level,
            read_only=cap.risk.read_only,
            destructive=cap.risk.destructive,
            requires_sandbox=cap.risk.requires_sandbox,
            requires_runtime_guard=cap.risk.requires_runtime_guard,
            requires_stop_guard=cap.risk.requires_stop_guard,
            max_duration_sec=cap.risk.max_duration_sec,
            max_rate_hz=cap.risk.max_rate_hz,
            constraints=cap.safety.get("constraints", {}),
            forbidden=forbidden,
        )

    @staticmethod
    def _map_level(risk_level: str, ros_name: str) -> str:
        if risk_level == "high" or ros_name.lower() == "/cmd_vel":
            return SafetyLevel.HIGH_RISK
        if risk_level == "medium":
            return SafetyLevel.MEDIUM_RISK
        if risk_level == "low":
            return SafetyLevel.LOW_RISK
        return SafetyLevel.READ_ONLY

    # ------------------------------------------------------------------
    # Runtime validation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        contract: SafetyContract,
        capability_id: str,
        args: dict[str, Any],
    ) -> SandboxDecision:
        """Evaluate an execution request against the contract."""
        rule = contract.get_rule(capability_id)
        if rule is None:
            return SandboxDecision(
                decision="BLOCK",
                risk_score=1.0,
                reason=f"No safety rule for capability '{capability_id}'",
                violated_constraints=["missing_safety_rule"],
            )

        if rule.forbidden:
            return SandboxDecision(
                decision="BLOCK",
                risk_score=1.0,
                reason="Capability is forbidden by default",
                violated_constraints=["forbidden"],
            )

        if rule.read_only and not rule.destructive:
            return SandboxDecision(decision="ALLOW", risk_score=0.0, reason="Read-only capability")

        violations: list[str] = []
        modified_args = dict(args)

        # Stop guard: velocity commands require duration.
        if rule.requires_stop_guard:
            duration = self._extract_duration(args)
            if duration is None:
                violations.append("velocity_command_requires_duration")
            elif rule.max_duration_sec is not None and duration > rule.max_duration_sec:
                violations.append(f"duration {duration}s exceeds max {rule.max_duration_sec}s")
                modified_args = self._clamp_duration(modified_args, rule.max_duration_sec)

        # Constraint checks for command topics.
        for key, bounds in rule.constraints.items():
            value = self._get_nested(modified_args, key)
            if value is None:
                continue
            lo, hi = bounds
            if value < lo or value > hi:
                violations.append(f"{key}={value} outside [{lo}, {hi}]")
                modified_args = self._set_nested(modified_args, key, max(lo, min(hi, value)))

        if violations:
            if rule.level == SafetyLevel.HIGH_RISK:
                return SandboxDecision(
                    decision="BLOCK",
                    risk_score=0.9,
                    reason="High-risk capability violates safety constraints",
                    violated_constraints=violations,
                )
            return SandboxDecision(
                decision="MODIFY",
                risk_score=0.5,
                reason="Arguments modified to satisfy safety constraints",
                violated_constraints=violations,
                modified_args=modified_args,
            )

        return SandboxDecision(
            decision="ALLOW",
            risk_score=0.1,
            reason="Within safety constraints",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_duration(args: dict[str, Any]) -> float | None:
        for key in ("duration", "duration_sec", "max_duration_sec", "time"):
            value = args.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        # Nested common schemas.
        linear = args.get("linear", {})
        if isinstance(linear, dict):
            duration = linear.get("duration")
            if isinstance(duration, (int, float)):
                return float(duration)
        return None

    @staticmethod
    def _clamp_duration(args: dict[str, Any], max_duration: float) -> dict[str, Any]:
        out = dict(args)
        for key in ("duration", "duration_sec", "max_duration_sec", "time"):
            if key in out and isinstance(out[key], (int, float)):
                out[key] = min(float(out[key]), max_duration)
                return out
        return out

    @staticmethod
    def _get_nested(data: dict[str, Any], key: str) -> Any:
        """Get a possibly nested value using dot notation."""
        parts = key.split(".")
        current: Any = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    @staticmethod
    def _set_nested(data: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
        """Set a possibly nested value using dot notation (deep copy)."""
        out = copy.deepcopy(data)
        parts = key.split(".")
        current = out
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
        return out
