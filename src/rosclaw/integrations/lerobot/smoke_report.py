"""Smoke report persistence for the LeRobot bridge.

Reports are written to ``~/.rosclaw/lerobot/smoke_reports/`` and are used by
``rosclaw lerobot doctor`` to distinguish "available" from "validated"
capabilities.

This module implements the ``rosclaw.lerobot.smoke.v1.1`` schema while
remaining tolerant of older ``v1`` reports.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from rosclaw.firstboot.workspace import get_rosclaw_home

SMOKE_REPORT_SCHEMA_VERSION = "rosclaw.lerobot.smoke.v1.1"
DEFAULT_REPORT_SUBDIR = "lerobot/smoke_reports"
PREVIEW_VALUES_LIMIT = 5
STALE_DAYS = 30


@dataclass
class SmokeReport:
    """A single real-policy smoke report."""

    schema_version: str = SMOKE_REPORT_SCHEMA_VERSION
    created_at: str = ""
    status: str = "error"
    error: dict[str, Any] | None = None

    policy: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    sample_observation: dict[str, Any] = field(default_factory=dict)
    action_proposal: dict[str, Any] | None = None
    timing: dict[str, float] = field(default_factory=dict)
    warnings: list[dict[str, str]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        # Normalize legacy v1 ``stages`` from dict[str, str] to dict[str, dict].
        if self.stages and all(isinstance(v, str) for v in self.stages.values()):
            self.stages = {k: {"status": v} for k, v in self.stages.items()}

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "status": self.status,
            "policy": self.policy,
            "runtime": self.runtime,
            "stages": self.stages,
            "features": self.features,
            "sample_observation": self.sample_observation,
            "action_proposal": self.action_proposal,
            "timing": self.timing,
            "warnings": self.warnings,
            "validation": self.validation,
        }
        if self.error is not None:
            out["error"] = self.error
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SmokeReport:
        return cls(
            schema_version=data.get("schema_version", SMOKE_REPORT_SCHEMA_VERSION),
            created_at=data.get("created_at", ""),
            status=data.get("status", "error"),
            error=data.get("error"),
            policy=data.get("policy", {}),
            runtime=data.get("runtime", {}),
            stages=data.get("stages", {}),
            features=data.get("features", {}),
            sample_observation=data.get("sample_observation", {}),
            action_proposal=data.get("action_proposal"),
            timing=data.get("timing", {}),
            warnings=data.get("warnings", []),
            validation=data.get("validation", {}),
        )


def get_smoke_report_dir() -> Path:
    """Return the directory where smoke reports are stored."""
    return get_rosclaw_home() / DEFAULT_REPORT_SUBDIR


def write_smoke_report(report: SmokeReport, *, suffix: str = "") -> Path:
    """Write a smoke report JSON and a ``latest.json`` symlink/overwrite."""
    report_dir = get_smoke_report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    name_source = report.policy.get("repo_id") or report.policy.get("local_path") or "unknown"
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(name_source))
    filename = f"{timestamp}_{safe_name}.json"
    if suffix:
        filename = f"{timestamp}_{safe_name}_{suffix}.json"

    path = report_dir / filename
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    latest_link = report_dir / "latest.json"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(path.name)
    except OSError:
        # Fallback: write a small redirect file if symlinks are not supported.
        latest_link.write_text(json.dumps({"latest_report": path.name}, indent=2), encoding="utf-8")

    return path


def read_latest_smoke_report() -> SmokeReport | None:
    """Return the most recent smoke report, or None if none exists."""
    report_dir = get_smoke_report_dir()
    latest_link = report_dir / "latest.json"
    if latest_link.is_symlink():
        target = latest_link.resolve()
        if target.exists():
            return _read_report_file(target)
        return None

    if latest_link.exists():
        try:
            redirect = json.loads(latest_link.read_text(encoding="utf-8"))
            target = report_dir / redirect.get("latest_report", "")
            if target.exists():
                return _read_report_file(target)
        except Exception:  # noqa: BLE001
            pass

    # Fallback: find the most recent report by filename.
    reports = sorted(report_dir.glob("*.json"), reverse=True)
    for candidate in reports:
        if candidate.name != "latest.json":
            return _read_report_file(candidate)
    return None


def _read_report_file(path: Path) -> SmokeReport | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return SmokeReport.from_dict(data)
    except Exception:  # noqa: BLE001
        return None


def _preview_values(values: Any, limit: int = PREVIEW_VALUES_LIMIT) -> list[float]:
    """Return the first ``limit`` scalar values as floats."""
    flat = _flatten_values(values)
    return [float(v) for v in flat[:limit]]


def _count_values(values: Any) -> int:
    """Count the total number of scalar values in a possibly nested list."""
    return len(_flatten_values(values))


def _flatten_values(values: Any) -> list[Any]:
    """Recursively flatten a list/tuple/ndarray-like structure."""
    flat: list[Any] = []
    if values is None:
        return flat
    if isinstance(values, (list, tuple)):
        for v in values:
            flat.extend(_flatten_values(v))
    else:
        flat.append(values)
    return flat


def summarize_action_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
    """Build a report-safe action proposal with limited preview values.

    Full action values are never written to the report; only a short preview and
    the shape/dtype/safety flags are persisted.
    """
    if proposal is None:
        return {
            "type": "none",
            "shape": [],
            "dtype": "float32",
            "preview_values": [],
            "num_values": 0,
            "not_executed": True,
            "requires_sandbox": True,
            "executable": False,
            "body_mapping_required": True,
            "body_compatible": False,
            "body_name": None,
        }

    values = proposal.get("values", [])
    summary: dict[str, Any] = {
        "type": proposal.get("type", "raw_lerobot_action"),
        "shape": proposal.get("shape", []),
        "dtype": proposal.get("dtype", "float32"),
        "preview_values": _preview_values(values),
        "num_values": _count_values(values),
        "not_executed": proposal.get("not_executed", True),
        "requires_sandbox": proposal.get("requires_sandbox", True),
        "executable": proposal.get("executable", False),
        "body_mapping_required": proposal.get("body_mapping_required", True),
        "body_compatible": proposal.get("body_compatible", False),
        "body_name": proposal.get("body_name"),
    }
    # Preserve chunk metadata when present.
    if "chunk_size" in proposal:
        summary["chunk_size"] = proposal["chunk_size"]
    if "action_dim" in proposal:
        summary["action_dim"] = proposal["action_dim"]
    return summary


def compute_warnings(
    timing: dict[str, float],
    action_proposal: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Generate hardening warnings based on timing and proposal safety."""
    warnings: list[dict[str, str]] = []

    infer_time = timing.get("infer_time_sec", 0.0)
    load_time = timing.get("load_time_sec", 0.0)
    total_time = timing.get("total_time_sec", 0.0)

    if total_time > 30:
        warnings.append(
            {
                "code": "slow_smoke_pipeline",
                "message": "Total smoke pipeline exceeded 30 seconds. "
                "Acceptable for validation but not for interactive rollout.",
            }
        )
    elif load_time > 10:
        warnings.append(
            {
                "code": "slow_policy_load",
                "message": "Policy load exceeded 10 seconds. "
                "Acceptable for smoke validation but not for real-time control.",
            }
        )
    elif infer_time > 5:
        warnings.append(
            {
                "code": "slow_one_shot_worker",
                "message": "Inference includes one-shot worker startup and policy loading. "
                "It is not suitable for real-time rollout.",
            }
        )

    warnings.append(
        {
            "code": "not_a_benchmark",
            "message": "Smoke validation only checks policy loading and one-step inference. "
            "It does not measure task success.",
        }
    )

    if action_proposal and action_proposal.get("body_mapping_required"):
        warnings.append(
            {
                "code": "body_mapping_required",
                "message": "LeRobot action space is not mapped to a ROSClaw body. "
                "Direct execution is not supported in P1.1.",
            }
        )

    return warnings


def build_validation_block(status: str) -> dict[str, Any]:
    """Build the validation section of a v1.1 report."""
    if status == "ok":
        return {
            "level": "validated",
            "meaning": "inspect/load-test/infer all passed on a real LeRobot policy",
            "proposal_only": True,
            "robot_execution_ready": False,
        }
    return {
        "level": "failed",
        "meaning": "Smoke validation failed; see error and stages for details",
        "proposal_only": True,
        "robot_execution_ready": False,
    }


def _parse_report_time(created_at: str) -> datetime | None:
    """Parse an ISO-8601 timestamp produced by this module."""
    if not created_at:
        return None
    try:
        if created_at.endswith("Z"):
            return datetime.fromisoformat(created_at[:-1] + "+00:00")
        return datetime.fromisoformat(created_at)
    except ValueError:
        return None


def get_stale_reasons(
    report: SmokeReport,
    current_lerobot_version: str | None = None,
    current_python_executable: str | None = None,
) -> list[str]:
    """Return human-readable reasons why a report is considered stale."""
    reasons: list[str] = []
    report_time = _parse_report_time(report.created_at)
    if report_time is not None:
        age = datetime.now(UTC) - report_time.replace(tzinfo=UTC)
        if age > timedelta(days=STALE_DAYS):
            reasons.append(f"Smoke report is {age.days} days old (threshold {STALE_DAYS} days).")

    report_version = report.runtime.get("lerobot_version")
    if current_lerobot_version and report_version and report_version != current_lerobot_version:
        reasons.append(
            f"Smoke report was created with LeRobot {report_version}, "
            f"current runtime is {current_lerobot_version}."
        )

    report_python = report.runtime.get("python_executable")
    if current_python_executable and report_python and report_python != current_python_executable:
        reasons.append(
            "Smoke report was created with a different LeRobot Python executable."
        )

    return reasons


def get_validation_status(
    report: SmokeReport | None = None,
    current_lerobot_version: str | None = None,
    current_python_executable: str | None = None,
) -> dict[str, Any]:
    """Return a summary of the latest smoke validation for doctor output.

    The returned dictionary includes a legacy ``validated`` boolean and a new
    ``state`` string so callers can render the five validation states:
    ``not_configured``, ``available_not_validated``, ``validated``, ``stale``,
    ``failed``.
    """
    if report is None:
        report = read_latest_smoke_report()

    if report is None:
        return {
            "validated": False,
            "state": "not_configured",
            "last_policy": None,
            "last_status": None,
            "policy_type": None,
            "lerobot_version": None,
            "device": None,
            "action_shape": None,
            "time": None,
            "safety": [],
            "performance_warning": None,
            "stale_reasons": [],
        }

    validated = report.status == "ok"
    stale_reasons = get_stale_reasons(report, current_lerobot_version, current_python_executable)

    if report.status == "error":
        state = "failed"
    elif stale_reasons:
        state = "stale"
    elif validated:
        state = "validated"
    else:
        state = "available_not_validated"

    proposal = report.action_proposal or {}
    safety: list[str] = []
    if proposal.get("not_executed"):
        safety.append("proposal_only")
    if proposal.get("requires_sandbox"):
        safety.append("sandbox_required")
    if proposal.get("body_mapping_required"):
        safety.append("body_mapping_required")

    performance_warning = None
    for warning in report.warnings:
        if warning.get("code") in {
            "slow_one_shot_worker",
            "slow_policy_load",
            "slow_smoke_pipeline",
        }:
            performance_warning = warning["message"]
            break

    return {
        "validated": validated and state == "validated",
        "state": state,
        "last_policy": report.policy.get("repo_id") or report.policy.get("local_path"),
        "last_status": report.status,
        "policy_type": report.policy.get("policy_type"),
        "lerobot_version": report.runtime.get("lerobot_version"),
        "device": report.runtime.get("device"),
        "action_shape": proposal.get("shape"),
        "time": report.created_at,
        "safety": safety,
        "performance_warning": performance_warning,
        "stale_reasons": stale_reasons,
    }
