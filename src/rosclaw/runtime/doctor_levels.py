"""Level-based doctor checks for execution readiness."""

from __future__ import annotations

import importlib
import uuid
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class DoctorLevel(StrEnum):
    PACKAGE = "package"
    CONFIGURED = "configured"
    RUNTIME = "runtime"
    CONNECTED = "connected"
    EXECUTION = "execution"
    VERIFIED = "verified"


@dataclass
class ReadinessCheck:
    """One explicit readiness assertion."""

    id: str
    passed: bool
    detail: str
    required: bool = True
    evidence: dict[str, Any] | None = None


@dataclass
class LevelDoctorResult:
    """Truthful doctor result with physical readiness kept separate."""

    requested_level: DoctorLevel
    passed: bool
    exit_code: int
    checks: list[ReadinessCheck]
    package_healthy: bool
    configured: bool
    runtime_initialized: bool
    southbound_connected: bool
    action_dry_run: bool
    verified_action_path: bool
    verified_execution_mode: str | None
    robot_connected: bool
    real_execution_ready: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_level": self.requested_level.value,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "package_healthy": self.package_healthy,
            "configured": self.configured,
            "runtime_initialized": self.runtime_initialized,
            "southbound_connected": self.southbound_connected,
            "action_dry_run": self.action_dry_run,
            "verified_action_path": self.verified_action_path,
            "verified_execution_mode": self.verified_execution_mode,
            "robot_connected": self.robot_connected,
            "real_execution_ready": self.real_execution_ready,
            "checks": [asdict(check) for check in self.checks],
        }


class LevelDoctor:
    """Run progressively stronger checks without conflating their meaning."""

    def __init__(self, home: Path) -> None:
        self.home = home.resolve()

    def run(self, requested_level: DoctorLevel | str) -> LevelDoctorResult:
        level = DoctorLevel(requested_level)
        level_order = list(DoctorLevel)

        def requested(check_level: DoctorLevel) -> bool:
            return level_order.index(level) >= level_order.index(check_level)

        checks: list[ReadinessCheck] = []
        required_modules = [
            "rosclaw.core.runtime",
            "rosclaw.kernel",
            "rosclaw.sandbox.episode",
        ]
        package_errors: list[str] = []
        for module_name in required_modules:
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001
                package_errors.append(f"{module_name}: {exc}")
        package_healthy = not package_errors
        checks.append(
            ReadinessCheck(
                id="L0_PACKAGE",
                passed=package_healthy,
                detail="core execution modules import"
                if package_healthy
                else "; ".join(package_errors),
            )
        )

        config_candidates = [Path.cwd() / "rosclaw.yaml", self.home / "config" / "rosclaw.yaml"]
        config_path = next((path for path in config_candidates if path.is_file()), None)
        configured = config_path is not None
        checks.append(
            ReadinessCheck(
                id="L1_CONFIGURED",
                passed=configured,
                detail=str(config_path) if config_path else "no rosclaw.yaml found",
                required=requested(DoctorLevel.CONFIGURED),
            )
        )

        runtime_initialized = False
        southbound_connected = False
        action_dry_run = False
        verified_action_path = False
        verified_execution_mode: str | None = None
        runtime: Any | None = None

        if package_healthy and requested(DoctorLevel.RUNTIME):
            try:
                from rosclaw.core.runtime import Runtime, RuntimeConfig

                artifact_root = self.home / "artifacts" / "doctor"
                runtime = Runtime(
                    RuntimeConfig(
                        robot_id="sim_ur5e",
                        default_eurdf_robot="ur5e",
                        enable_event_persistence=False,
                        enable_tracing=True,
                        trace_home=str(self.home / "logs" / "doctor-trace"),
                        enable_firewall=False,
                        enable_memory=False,
                        enable_practice=False,
                        enable_skill_manager=False,
                        enable_knowledge=False,
                        enable_how=False,
                        enable_auto=False,
                        enable_provider=False,
                        enable_sense=False,
                        sandbox_engine="mujoco",
                        sandbox_world_id="tabletop",
                        sandbox_artifact_root=str(artifact_root),
                    )
                )
                runtime.initialize()
                runtime_initialized = True
                health = runtime.sandbox.health() if runtime.sandbox is not None else {}
                southbound_connected = bool(
                    health.get("status") == "healthy" and health.get("has_physics")
                )
                checks.append(
                    ReadinessCheck(
                        id="L2_RUNTIME_INITIALIZED",
                        passed=True,
                        detail="Runtime initialized with truthful action gateway",
                    )
                )
                if requested(DoctorLevel.CONNECTED):
                    checks.append(
                        ReadinessCheck(
                            id="L3_SOUTHBOUND_CONNECTED",
                            passed=southbound_connected,
                            detail=(
                                "MuJoCo model and tabletop world loaded"
                                if southbound_connected
                                else str(health.get("error") or "sandbox physics unavailable")
                            ),
                            evidence=health,
                        )
                    )

                if requested(DoctorLevel.EXECUTION):
                    action_dry_run, dry_run_evidence = self._check_dry_run(runtime)
                    checks.append(
                        ReadinessCheck(
                            id="L4_ACTION_DRY_RUN",
                            passed=action_dry_run,
                            detail=(
                                "out-of-bounds action blocked before simulation"
                                if action_dry_run
                                else "dry-run policy did not fail closed"
                            ),
                            evidence=dry_run_evidence,
                        )
                    )
                if requested(DoctorLevel.VERIFIED):
                    verified_action_path, verified_evidence = self._check_verified(runtime)
                    verified_execution_mode = "SIMULATION" if verified_action_path else None
                    checks.append(
                        ReadinessCheck(
                            id="L5_VERIFIED_EXECUTION",
                            passed=verified_action_path,
                            detail=(
                                "MuJoCo reach task verified from observed state"
                                if verified_action_path
                                else "MuJoCo reach task was not verified"
                            ),
                            evidence=verified_evidence,
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                checks.append(
                    ReadinessCheck(
                        id="L2_RUNTIME_INITIALIZED",
                        passed=False,
                        detail=str(exc),
                    )
                )
            finally:
                if runtime is not None:
                    runtime.stop()

        robot_connected = False
        real_execution_ready = False
        checks.append(
            ReadinessCheck(
                id="REAL_ROBOT_CONNECTION",
                passed=False,
                detail="No live hardware driver connection was requested or verified.",
                required=False,
            )
        )

        passed_by_level = {
            DoctorLevel.PACKAGE: package_healthy,
            DoctorLevel.CONFIGURED: package_healthy and configured,
            DoctorLevel.RUNTIME: package_healthy and configured and runtime_initialized,
            DoctorLevel.CONNECTED: package_healthy
            and configured
            and runtime_initialized
            and southbound_connected,
            DoctorLevel.EXECUTION: package_healthy
            and configured
            and runtime_initialized
            and southbound_connected
            and action_dry_run,
            DoctorLevel.VERIFIED: package_healthy
            and configured
            and runtime_initialized
            and southbound_connected
            and action_dry_run
            and verified_action_path,
        }
        passed = passed_by_level[level]
        return LevelDoctorResult(
            requested_level=level,
            passed=passed,
            exit_code=0 if passed else 1,
            checks=checks,
            package_healthy=package_healthy,
            configured=configured,
            runtime_initialized=runtime_initialized,
            southbound_connected=southbound_connected,
            action_dry_run=action_dry_run,
            verified_action_path=verified_action_path,
            verified_execution_mode=verified_execution_mode,
            robot_connected=robot_connected,
            real_execution_ready=real_execution_ready,
        )

    @staticmethod
    def _check_dry_run(runtime: Any) -> tuple[bool, dict[str, Any]]:
        from rosclaw.kernel import ActionEnvelope, ActionState, ExecutionMode

        action = ActionEnvelope(
            action_id=f"doctor_dry_{uuid.uuid4().hex}",
            actor_id="rosclaw-doctor",
            agent_framework="doctor",
            session_id="doctor",
            body_id="sim_ur5e",
            body_snapshot_hash="doctor",
            capability_id="sandbox.reach",
            arguments={"task": "reach", "target": [2.0, 0.0, 0.5]},
            execution_mode=ExecutionMode.SIMULATION,
        )
        receipt = runtime.submit_action(action)
        evidence = receipt.to_dict()
        return (
            receipt.final_state is ActionState.BLOCKED
            and receipt.policy_decision.get("reason") == "target_outside_workspace"
            and receipt.simulation_result is None,
            evidence,
        )

    @staticmethod
    def _check_verified(runtime: Any) -> tuple[bool, dict[str, Any]]:
        from rosclaw.kernel import ActionEnvelope, ActionState, EvidenceLevel, ExecutionMode

        action = ActionEnvelope(
            action_id=f"doctor_verified_{uuid.uuid4().hex}",
            actor_id="rosclaw-doctor",
            agent_framework="doctor",
            session_id="doctor",
            body_id="sim_ur5e",
            body_snapshot_hash="doctor",
            capability_id="sandbox.reach",
            arguments={"task": "reach", "seed": 0, "max_steps": 1200},
            execution_mode=ExecutionMode.SIMULATION,
        )
        receipt = runtime.submit_action(action)
        artifact_paths = [
            Path(urlparse(uri).path) for uri in receipt.artifacts if urlparse(uri).scheme == "file"
        ]
        trace_events = runtime.event_bus.get_history("rosclaw.trace.span.completed", limit=100)
        trace_observed = any(event.trace_id == receipt.trace_id for event in trace_events)
        passed = bool(
            receipt.final_state is ActionState.COMPLETED
            and receipt.evidence_level is EvidenceLevel.TASK_VERIFIED
            and receipt.verified
            and artifact_paths
            and all(path.is_file() for path in artifact_paths)
            and trace_observed
        )
        evidence = {
            "receipt": receipt.to_dict(),
            "artifact_paths_exist": bool(artifact_paths)
            and all(path.is_file() for path in artifact_paths),
            "trace_observed": trace_observed,
        }
        return passed, evidence


__all__ = [
    "DoctorLevel",
    "LevelDoctor",
    "LevelDoctorResult",
    "ReadinessCheck",
]
