"""Tests for body snapshot sanitization and provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rosclaw.integrations.lerobot.body_snapshot import (
    SensitiveBodyDataError,
    include_body_snapshot,
)

SAMPLE_BODY = """---
body_id: mock_body_001
serial: SECRET_SERIAL_12345
ip_address: 192.168.1.100
token: super_secret_token_xyz
location:
  lat: 37.7749
  lon: -122.4194
model: mock_robot_v1
"""


def test_sanitized_snapshot_strips_secrets(tmp_path: Path) -> None:
    body_yaml = tmp_path / "body.yaml"
    body_yaml.write_text(SAMPLE_BODY, encoding="utf-8")
    out_dir = tmp_path / "dataset"
    manifest = include_body_snapshot(out_dir, body_yaml, mode="sanitized")
    snapshot = out_dir / "meta" / "rosclaw" / "body_snapshots" / "body.yaml"
    assert snapshot.exists()
    text = snapshot.read_text(encoding="utf-8")
    assert "SECRET_SERIAL_12345" not in text
    assert "192.168.1.100" not in text
    assert "super_secret_token_xyz" not in text
    assert "37.7749" not in text

    file_entry = manifest["files"]["body.yaml"]
    assert file_entry["source_sha256"] == hashlib.sha256(SAMPLE_BODY.encode("utf-8")).hexdigest()
    sanitized_text = snapshot.read_text(encoding="utf-8")
    assert file_entry["sanitized_sha256"] == hashlib.sha256(sanitized_text.encode("utf-8")).hexdigest()
    assert file_entry["source_sha256"] != file_entry["sanitized_sha256"]
    assert file_entry["redaction_count"] > 0
    assert "serial" in file_entry["redacted_fields"]
    assert "ip_address" in file_entry["redacted_fields"]


def test_full_snapshot_requires_acknowledgement(tmp_path: Path) -> None:
    body_yaml = tmp_path / "body.yaml"
    body_yaml.write_text(SAMPLE_BODY, encoding="utf-8")
    out_dir = tmp_path / "dataset"
    with pytest.raises(SensitiveBodyDataError):
        include_body_snapshot(out_dir, body_yaml, mode="full")


def test_full_snapshot_preserves_values_when_acknowledged(tmp_path: Path) -> None:
    body_yaml = tmp_path / "body.yaml"
    body_yaml.write_text(SAMPLE_BODY, encoding="utf-8")
    out_dir = tmp_path / "dataset"
    manifest = include_body_snapshot(
        out_dir, body_yaml, mode="full", acknowledge_sensitive=True
    )
    snapshot = out_dir / "meta" / "rosclaw" / "body_snapshots" / "body.yaml"
    text = snapshot.read_text(encoding="utf-8")
    assert "SECRET_SERIAL_12345" in text
    assert "192.168.1.100" in text

    file_entry = manifest["files"]["body.yaml"]
    assert file_entry["source_sha256"] == file_entry["sanitized_sha256"]
    assert file_entry["redaction_count"] == 0
    assert file_entry["redacted_fields"] == []
    assert manifest["acknowledged_sensitive"] is True


def test_none_mode_skips_snapshot(tmp_path: Path) -> None:
    body_yaml = tmp_path / "body.yaml"
    body_yaml.write_text(SAMPLE_BODY, encoding="utf-8")
    out_dir = tmp_path / "dataset"
    manifest = include_body_snapshot(out_dir, body_yaml, mode="none")
    assert not (out_dir / "meta" / "rosclaw" / "body_snapshots").exists()
    assert manifest["files"] == {}


def test_missing_body_yaml_returns_empty_manifest(tmp_path: Path) -> None:
    out_dir = tmp_path / "dataset"
    manifest = include_body_snapshot(out_dir, tmp_path / "missing.yaml", mode="sanitized")
    assert manifest["files"] == {}
    assert manifest["body_yaml_path"] is None
