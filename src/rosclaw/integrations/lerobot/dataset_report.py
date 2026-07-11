"""Dataset export report persistence for the LeRobot bridge.

Reports are written to both the dataset output directory and
``~/.rosclaw/lerobot/dataset_exports/``.  They are used by
``rosclaw lerobot doctor`` to show the latest dataset export validation state.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rosclaw.firstboot.workspace import get_rosclaw_home

DATASET_EXPORT_SCHEMA_VERSION = "rosclaw.lerobot.dataset_export.v1.1"
LEGACY_SCHEMA_VERSION = "rosclaw.lerobot.dataset_export.v1"
DEFAULT_EXPORT_SUBDIR = "lerobot/dataset_exports"

# Feature groups implemented in Gate B. Groups outside this set are planned for Gate C.
IMPLEMENTED_FEATURE_GROUPS = {"safety", "failure", "intervention", "action", "outcome", "physical_telemetry"}


def _profile_scope(requested: list[str], written: list[str]) -> dict[str, list[str]]:
    requested_set = set(requested)
    written_set = set(written)
    return {
        "requested": sorted(requested_set),
        "validated": sorted(written_set),
        "missing": sorted(requested_set - written_set),
        "planned": sorted(requested_set - IMPLEMENTED_FEATURE_GROUPS),
    }


@dataclass
class DatasetExportReport:
    """A single ROSClaw Practice -> LeRobotDataset export report."""

    schema_version: str = DATASET_EXPORT_SCHEMA_VERSION
    created_at: str = ""
    status: str = "error"
    source: dict[str, Any] = field(default_factory=dict)
    target: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    visual: dict[str, Any] = field(default_factory=dict)
    lerobot_dataset_api: dict[str, Any] = field(default_factory=dict)
    quality_gates: dict[str, Any] = field(default_factory=dict)
    extension_schema: str = ""
    feature_groups: list[str] = field(default_factory=list)
    profile: dict[str, Any] = field(default_factory=dict)
    extension_gate: str = "P2.1 Gate B"
    error: dict[str, Any] | None = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "status": self.status,
            "source": self.source,
            "target": self.target,
            "dataset": self.dataset,
            "validation": self.validation,
            "safety": self.safety,
            "limitations": self.limitations,
            "timing": self.timing,
        }
        if self.runtime:
            out["runtime"] = self.runtime
        if self.visual:
            out["visual"] = self.visual
        if self.lerobot_dataset_api:
            out["lerobot_dataset_api"] = self.lerobot_dataset_api
        if self.quality_gates:
            out["quality_gates"] = self.quality_gates
        if self.extension_schema:
            out["extension_schema"] = self.extension_schema
        if self.feature_groups:
            out["feature_groups"] = self.feature_groups
        if self.profile:
            out["profile"] = self.profile
        if self.extension_gate:
            out["extension_gate"] = self.extension_gate
        if self.error is not None:
            out["error"] = self.error
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetExportReport:
        return cls(
            schema_version=data.get("schema_version", LEGACY_SCHEMA_VERSION),
            created_at=data.get("created_at", ""),
            status=data.get("status", "error"),
            source=data.get("source", {}),
            target=data.get("target", {}),
            dataset=data.get("dataset", {}),
            validation=data.get("validation", {}),
            safety=data.get("safety", {}),
            limitations=list(data.get("limitations", [])),
            timing=data.get("timing", {}),
            runtime=data.get("runtime", {}),
            visual=data.get("visual", {}),
            lerobot_dataset_api=data.get("lerobot_dataset_api", {}),
            quality_gates=data.get("quality_gates", {}),
            extension_schema=data.get("extension_schema", ""),
            feature_groups=list(data.get("feature_groups", [])),
            profile=data.get("profile", {}),
            extension_gate=data.get("extension_gate", "P2.1 Gate A"),
            error=data.get("error"),
        )


def get_dataset_export_dir() -> Path:
    """Return the directory where dataset export reports are stored."""
    return get_rosclaw_home() / DEFAULT_EXPORT_SUBDIR


def write_dataset_export_report(
    report: DatasetExportReport,
    output_dir: Path | str,
    *,
    suffix: str = "",
) -> Path:
    """Write a report to the output directory and update the latest report link."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "rosclaw_export_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    export_dir = get_dataset_export_dir()
    export_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", report.target.get("repo_id", "unknown"))
    filename = f"{timestamp}_{safe_name}.json"
    if suffix:
        filename = f"{timestamp}_{safe_name}_{suffix}.json"

    persistent_path = export_dir / filename
    persistent_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    latest_link = export_dir / "latest.json"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(persistent_path.name)
    except OSError:
        latest_link.write_text(json.dumps({"latest_report": persistent_path.name}, indent=2), encoding="utf-8")

    return report_path


def read_latest_dataset_export_report() -> DatasetExportReport | None:
    """Return the most recent dataset export report, or None if none exists."""
    export_dir = get_dataset_export_dir()
    latest_link = export_dir / "latest.json"
    if latest_link.is_symlink():
        target = latest_link.resolve()
        if target.exists():
            return _read_report_file(target)
        return None

    if latest_link.exists():
        try:
            redirect = json.loads(latest_link.read_text(encoding="utf-8"))
            target = export_dir / redirect.get("latest_report", "")
            if target.exists():
                return _read_report_file(target)
        except Exception:  # noqa: BLE001
            pass

    reports = sorted(export_dir.glob("*.json"), reverse=True)
    for candidate in reports:
        if candidate.name != "latest.json":
            return _read_report_file(candidate)
    return None


def _read_report_file(path: Path) -> DatasetExportReport | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return DatasetExportReport.from_dict(data)
    except Exception:  # noqa: BLE001
        return None


def _report_age_days(report: DatasetExportReport) -> float:
    try:
        created = datetime.fromisoformat(report.created_at.replace("Z", "+00:00"))
        delta = datetime.now(UTC) - created
        return delta.total_seconds() / 86400.0
    except Exception:  # noqa: BLE001
        return 0.0


def get_dataset_export_validation_status(
    *,
    current_lerobot_version: str | None = None,
    current_python_executable: str | None = None,
    current_api_signature: str | None = None,
) -> dict[str, Any]:
    """Return the validation state of the latest dataset export report."""
    report = read_latest_dataset_export_report()
    if report is None:
        return {
            "state": "not_configured",
            "message": "No dataset export report found.",
        }

    if report.status != "ok":
        message = "Latest dataset export failed."
        if report.error and report.error.get("message"):
            message = report.error.get("message")
        return {
            "state": "failed",
            "message": message,
            "last_output_dir": report.target.get("output_dir"),
        }

    stale_reasons: list[str] = []
    last_output_dir = report.target.get("output_dir")
    if last_output_dir:
        p = Path(last_output_dir)
        if not p.exists() or not (p / "meta" / "info.json").exists():
            stale_reasons.append("Dataset output directory or meta/info.json is missing.")

    report_runtime = report.runtime or {}
    if (
        current_lerobot_version
        and report_runtime.get("lerobot_version")
        and report_runtime["lerobot_version"] != current_lerobot_version
    ):
        stale_reasons.append(
            f"LeRobot version changed: report={report_runtime['lerobot_version']} "
            f"current={current_lerobot_version}"
        )
    if (
        current_python_executable
        and report_runtime.get("python")
        and Path(report_runtime["python"]).resolve() != Path(current_python_executable).resolve()
    ):
        stale_reasons.append("Python executable changed since last export.")

    if (
        current_api_signature
        and report.lerobot_dataset_api.get("create_signature")
        and report.lerobot_dataset_api["create_signature"] != current_api_signature
    ):
        stale_reasons.append("LeRobotDataset API signature changed since last export.")

    if report.schema_version == LEGACY_SCHEMA_VERSION:
        stale_reasons.append(f"Report schema is older than current ({DATASET_EXPORT_SCHEMA_VERSION}).")

    if _report_age_days(report) > 30:
        stale_reasons.append("Report is older than 30 days and has not been revalidated.")

    if stale_reasons:
        return {
            "state": "stale",
            "last_output_dir": last_output_dir,
            "repo_id": report.target.get("repo_id"),
            "num_frames": report.dataset.get("num_frames"),
            "num_episodes": report.dataset.get("num_episodes"),
            "features": list(report.dataset.get("features", {}).keys()),
            "visual": report.visual,
            "quality_gates": report.quality_gates,
            "profile": report.profile,
            "extension_gate": report.extension_gate,
            "stale_reasons": stale_reasons,
        }

    return {
        "state": "validated",
        "last_output_dir": last_output_dir,
        "repo_id": report.target.get("repo_id"),
        "num_frames": report.dataset.get("num_frames"),
        "num_episodes": report.dataset.get("num_episodes"),
        "features": list(report.dataset.get("features", {}).keys()),
        "visual": report.visual,
        "quality_gates": report.quality_gates,
        "profile": report.profile,
        "extension_gate": report.extension_gate,
        "load_ok": report.validation.get("load_ok", False),
        "index_ok": report.validation.get("index_ok", False),
    }


def build_safety_block() -> dict[str, Any]:
    """Build the P2 safety block for export reports."""
    return {
        "contains_executed_actions": True,
        "actions_are_historical": True,
        "not_a_policy_execution": True,
        "execution_context": "playback",
        "safety_reviewed": False,
        "source_action_verified": False,
    }


def _build_limitations_from_features(feature_names: list[str]) -> list[str]:
    """Derive limitations from the actual exported feature set."""
    limitations: list[str] = []
    if not any(k.startswith("observation.depth.") for k in feature_names):
        limitations.append("No depth export in P2/P2.1")
    if "observation.motor_current" not in feature_names:
        limitations.append("No motor current export in P2/P2.1")
    if "observation.force_torque" not in feature_names:
        limitations.append("No force/torque export in P2/P2.1")
    if not any(k.startswith("rosclaw.sandbox.") for k in feature_names):
        limitations.append("No sandbox metadata in this export")
    if not any(k.startswith("rosclaw.intervention.") for k in feature_names):
        limitations.append("No intervention metadata in this export")
    return limitations or ["Minimal export: only state, action, task, and visual data"]


def build_limitations_block(feature_names: list[str] | None = None) -> list[str]:
    """Build the export limitations list.

    When ``feature_names`` is provided, the list is derived from the actual
    exported features so it does not claim missing capabilities that were used.
    """
    if feature_names is not None:
        return _build_limitations_from_features(feature_names)
    return [
        "No depth export in P2/P2.1",
        "No motor current export in P2/P2.1",
        "No force/torque export in P2/P2.1",
        "Single-camera practice episodes only in P2",
    ]


def report_from_worker_response(
    episode_path: str,
    episode_id: str,
    output_dir: str,
    repo_id: str,
    response_dict: dict[str, Any],
) -> DatasetExportReport:
    """Build a report from a dataset worker response dict."""
    status = "ok" if response_dict.get("status") == "ok" else "error"
    validation = response_dict.get("validation", {})
    dataset = response_dict.get("dataset", {})
    timing = response_dict.get("timing", {})
    error = response_dict.get("error") if status == "error" else None
    feature_names = list(dataset.get("features", {}).keys())
    visual = response_dict.get("visual", dataset.get("visual", {}))
    api_info = response_dict.get("api_info", {})

    return DatasetExportReport(
        status=status,
        source={
            "type": "rosclaw_practice_episode",
            "episode_path": episode_path,
            "episode_id": episode_id,
        },
        target={
            "format": "lerobot_v3",
            "repo_id": repo_id,
            "output_dir": output_dir,
        },
        dataset=dataset,
        validation={
            "load_ok": validation.get("load_ok", False),
            "index_ok": validation.get("index_ok", False),
            "dataloader_ok": validation.get("dataloader_ok"),
            "num_frames": validation.get("num_frames"),
            "num_episodes": validation.get("num_episodes"),
            "sample_keys": validation.get("sample_keys", []),
        },
        safety=build_safety_block(),
        limitations=build_limitations_block(feature_names),
        timing=timing,
        error=error,
        runtime=response_dict.get("runtime", {}),
        visual=visual,
        lerobot_dataset_api={
            "create_signature": api_info.get("create_signature", ""),
            "has_add_frame": api_info.get("has_add_frame", False),
            "has_save_episode": api_info.get("has_save_episode", False),
            "has_finalize": api_info.get("has_finalize", False),
            "has_consolidate": api_info.get("has_consolidate", False),
            "lerobot_version": api_info.get("lerobot_version"),
        },
        quality_gates={
            "load_ok": validation.get("load_ok", False),
            "index_ok": validation.get("index_ok", False),
            "dataloader_ok": validation.get("dataloader_ok"),
        },
        extension_schema=response_dict.get("extension_schema", ""),
        feature_groups=list(response_dict.get("feature_groups_written", [])),
        profile={
            "name": response_dict.get("profile", "minimal"),
            "requested": list(response_dict.get("requested_feature_groups", [])),
            "validated": list(response_dict.get("written_feature_groups", [])),
            "missing": list(response_dict.get("missing_feature_groups", [])),
            "planned": sorted(
                set(response_dict.get("requested_feature_groups", [])) - IMPLEMENTED_FEATURE_GROUPS
            ),
            "satisfied": bool(response_dict.get("profile_satisfied", True)),
            "scope": _profile_scope(
                list(response_dict.get("requested_feature_groups", [])),
                list(response_dict.get("written_feature_groups", [])),
            ),
        },
    )


__all__ = [
    "DATASET_EXPORT_SCHEMA_VERSION",
    "LEGACY_SCHEMA_VERSION",
    "DatasetExportReport",
    "build_limitations_block",
    "build_safety_block",
    "read_latest_dataset_export_report",
    "get_dataset_export_validation_status",
    "report_from_worker_response",
    "write_dataset_export_report",
]
