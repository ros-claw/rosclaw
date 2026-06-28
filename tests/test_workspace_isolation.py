"""Verify that ROSCLAW_HOME is respected across modules.

These tests make sure no module hard-codes ``Path.home() / ".rosclaw"`` when
no explicit workspace is provided.  They use a temporary directory pointed to
by the ``ROSCLAW_HOME`` environment variable and assert that state lands there.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Set ROSCLAW_HOME to a temp directory and return it."""
    home = tmp_path / "rosclaw_home"
    home.mkdir()
    monkeypatch.setenv("ROSCLAW_HOME", str(home))
    # Make sure imported modules re-evaluate the env var when instantiated.
    return home


def test_get_rosclaw_home_respects_env(isolated_home):
    from rosclaw.firstboot.workspace import get_rosclaw_home

    assert get_rosclaw_home() == isolated_home


def test_resolve_home_with_explicit_path():
    from rosclaw.firstboot.workspace import resolve_home

    explicit = Path("/tmp/custom_rosclaw")
    assert resolve_home(str(explicit)) == explicit


def test_body_resolver_uses_rosclaw_home(isolated_home):
    from rosclaw.body.resolver import BodyResolver

    resolver = BodyResolver()
    assert resolver.workspace == isolated_home


def test_body_instance_service_uses_rosclaw_home(isolated_home):
    from rosclaw.body.service import BodyInstanceService

    service = BodyInstanceService()
    assert service.workspace == isolated_home


def test_practice_config_uses_rosclaw_home(isolated_home):
    from rosclaw.practice.config import PracticeConfig

    cfg = PracticeConfig()
    assert cfg.config_root == isolated_home / "practice"


def test_dashboard_server_uses_rosclaw_home(isolated_home):
    from rosclaw.dashboard.server import DashboardServer
    from rosclaw.dashboard.metrics import DashboardMetrics

    server = DashboardServer(DashboardMetrics())
    summary = server.get_body_summary()
    assert summary["workspace"] == str(isolated_home)


def test_hub_lockfile_uses_rosclaw_home(isolated_home):
    from rosclaw.hub.lockfile import DEFAULT_LOCKFILE_PATH

    assert DEFAULT_LOCKFILE_PATH == isolated_home / "assets.lock"


def test_mcp_tools_audit_log_uses_rosclaw_home(isolated_home, monkeypatch):
    from rosclaw.mcp.tools import _audit

    # _audit writes to logs/mcp/audit.jsonl under ROSCLAW_HOME.
    _audit("trace-1", "test_tool", {"a": 1}, {"ok": True}, 1.23)

    audit_file = isolated_home / "logs" / "mcp" / "audit.jsonl"
    assert audit_file.exists()
    assert "trace-1" in audit_file.read_text()


def test_firstboot_doctor_uses_rosclaw_home(isolated_home):
    from rosclaw.firstboot.doctor import FirstbootDoctor

    doctor = FirstbootDoctor()
    assert doctor.home == isolated_home
