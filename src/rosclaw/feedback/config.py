"""Telemetry and feedback configuration dataclasses with YAML persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TelemetryConfig:
    """Product telemetry configuration matching rosclaw_feedback_v1.md."""

    version: str = "1"
    mode: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)
    product_telemetry: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    rich_feedback: dict[str, Any] = field(default_factory=dict)
    upload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._apply_defaults()

    def _apply_defaults(self) -> None:
        self.mode = self._merge(self.mode, {
            "enabled": True,
            "product_telemetry": True,
            "diagnostics_upload": False,
            "rich_feedback_upload": False,
        })
        self.identity = self._merge(self.identity, {
            "use_anonymous_installation_id": True,
            "include_hostname": False,
            "include_username": False,
            "include_ip_field": False,
            "include_robot_serial": False,
            "rotate_installation_id": False,
        })
        self.product_telemetry = self._merge(self.product_telemetry, {
            "enabled": True,
            "opt_out": True,
            "heartbeat": True,
            "heartbeat_interval_hours": 24,
            "send_on_install": True,
            "send_on_firstboot": True,
            "send_on_version_check": True,
            "send_command_usage": True,
            "send_module_usage": True,
            "send_error_summary": True,
            "send_duration_bucket": True,
        })
        self.diagnostics = self._merge(self.diagnostics, {
            "enabled": False,
            "require_consent": True,
            "redact": True,
            "collect_crash_summary": True,
            "collect_failure_stats": True,
            "collect_sandbox_blocks": True,
            "collect_provider_performance": True,
            "include_stacktrace": False,
            "include_logs": False,
        })
        self.rich_feedback = self._merge(self.rich_feedback, {
            "enabled": False,
            "require_manual_upload": True,
            "require_redact": True,
            "raw_prompt_upload": False,
            "raw_media_upload": False,
            "raw_mcap_upload": False,
            "raw_trace_upload": False,
        })
        self.upload = self._merge(self.upload, {
            "endpoint": "https://www.rosclaw.io/api/telemetry/event",
            "heartbeat_endpoint": "https://www.rosclaw.io/api/telemetry/heartbeat",
            "timeout_seconds": 3,
            "max_retries": 1,
            "fail_silently": True,
            "max_events_per_batch": 50,
        })

    @staticmethod
    def _merge(existing: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
        result = dict(existing)
        for key, value in defaults.items():
            if key not in result:
                result[key] = value
            elif isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = TelemetryConfig._merge(result[key], value)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "mode": self.mode,
            "identity": self.identity,
            "product_telemetry": self.product_telemetry,
            "diagnostics": self.diagnostics,
            "rich_feedback": self.rich_feedback,
            "upload": self.upload,
        }

    def save(self, home: Path) -> Path:
        path = home / "config" / "telemetry.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), default_flow_style=False, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, home: Path) -> TelemetryConfig:
        path = home / "config" / "telemetry.yaml"
        if not path.exists():
            return cls()
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, OSError):
            return cls()
        return cls(
            version=data.get("version", "1"),
            mode=data.get("mode", {}),
            identity=data.get("identity", {}),
            product_telemetry=data.get("product_telemetry", {}),
            diagnostics=data.get("diagnostics", {}),
            rich_feedback=data.get("rich_feedback", {}),
            upload=data.get("upload", {}),
        )


@dataclass
class FeedbackConfig:
    """Local feedback and rich bundle configuration."""

    version: str = "1"
    mode: dict[str, Any] = field(default_factory=dict)
    retention: dict[str, Any] = field(default_factory=dict)
    collect: dict[str, Any] = field(default_factory=dict)
    redaction: dict[str, Any] = field(default_factory=dict)
    upload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._apply_defaults()

    def _apply_defaults(self) -> None:
        self.mode = self._merge(self.mode, {
            "enabled": True,
            "local_store": True,
            "upload": False,
            "redact": True,
        })
        self.retention = self._merge(self.retention, {
            "local_days": 30,
            "max_local_size_mb": 512,
            "uploaded_metadata_days": 180,
            "uploaded_attachments_days": 30,
        })
        self.collect = self._merge(self.collect, {
            "failure_stats": {"enabled": True, "local_only": True},
            "skill_performance": {"enabled": True, "local_only": True},
            "crash_reports": {"enabled": True, "local_only": True, "include_stacktrace": False},
            "human_feedback": {"enabled": True, "local_only": True},
            "sandbox_blocks": {"enabled": True, "local_only": True},
            "provider_performance": {"enabled": True, "local_only": True},
            "prompts": {"enabled": False, "local_only": True},
            "media": {"enabled": False, "local_only": True},
            "mcap": {"enabled": False, "local_only": True},
        })
        self.redaction = self._merge(self.redaction, {
            "text": {
                "enabled": True,
                "replace_emails": True,
                "replace_phone_numbers": True,
                "replace_ips": True,
                "replace_urls": True,
                "replace_file_paths": True,
                "replace_tokens": True,
                "replace_usernames": True,
            },
            "prompts": {
                "enabled": True,
                "mode": "summary_and_hash",
                "keep_full_prompt": False,
            },
            "media": {
                "enabled": True,
                "face_blur": True,
                "person_blur": True,
                "qr_blur": True,
                "screen_blur": True,
                "downsample_fps": 1,
                "max_seconds": 15,
            },
            "mcap": {
                "enabled": True,
                "allow_topics": [
                    "/joint_states",
                    "/imu",
                    "/odom",
                    "/sandbox/decision",
                    "/rosclaw/runtime/event",
                    "/rosclaw/provider/perf",
                    "/rosclaw/skill/outcome",
                ],
                "deny_topics": [
                    "/camera",
                    "/rgb",
                    "/depth",
                    "/audio",
                    "/microphone",
                    "/pointcloud",
                ],
            },
        })
        self.upload = self._merge(self.upload, {
            "endpoint": "https://www.rosclaw.io/api/feedback/upload",
            "timeout_seconds": 10,
            "max_retries": 1,
            "fail_silently": True,
            "require_redact": True,
            "max_bundle_mb": 25,
        })

    @staticmethod
    def _merge(existing: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
        result = dict(existing)
        for key, value in defaults.items():
            if key not in result:
                result[key] = value
            elif isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = FeedbackConfig._merge(result[key], value)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "mode": self.mode,
            "retention": self.retention,
            "collect": self.collect,
            "redaction": self.redaction,
            "upload": self.upload,
        }

    def save(self, home: Path) -> Path:
        path = home / "config" / "feedback.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), default_flow_style=False, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, home: Path) -> FeedbackConfig:
        path = home / "config" / "feedback.yaml"
        if not path.exists():
            return cls()
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, OSError):
            return cls()
        return cls(
            version=data.get("version", "1"),
            mode=data.get("mode", {}),
            retention=data.get("retention", {}),
            collect=data.get("collect", {}),
            redaction=data.get("redaction", {}),
            upload=data.get("upload", {}),
        )


__all__ = ["TelemetryConfig", "FeedbackConfig"]
