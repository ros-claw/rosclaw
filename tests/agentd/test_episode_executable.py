"""Actual dispatch regression for intact bytes with a missing execute bit."""

import hashlib
import subprocess
from pathlib import Path

import pytest

from rosclaw.agentd.episode_executable import (
    EpisodeExecutableError,
    validate_episode_executable,
)


def fixture_asset(tmp_path: Path) -> tuple[Path, str]:
    asset = tmp_path / "saved-scene-fixture"
    asset.write_bytes(b"#!/bin/sh\nexit 23\n")
    return asset, hashlib.sha256(asset.read_bytes()).hexdigest()


def test_same_digest_without_execute_mode_rejected_before_clock(tmp_path: Path):
    asset, digest = fixture_asset(tmp_path)
    asset.chmod(0o664)
    clock = tmp_path / "clock.json"
    with pytest.raises(PermissionError):
        subprocess.run([str(asset)], check=False)
    with pytest.raises(EpisodeExecutableError, match="EXECUTABLE_PERMISSION_DENIED"):
        validate_episode_executable(asset, digest)
        clock.write_text("started")
    assert not clock.exists()
    assert asset.stat().st_mode & 0o777 == 0o664
    assert hashlib.sha256(asset.read_bytes()).hexdigest() == digest


def test_executable_passes_and_actual_harmless_startup(tmp_path: Path):
    asset, digest = fixture_asset(tmp_path)
    asset.chmod(0o755)
    result = validate_episode_executable(asset, digest)
    assert result == {"path": str(asset), "sha256": digest, "mode": 0o755}
    assert subprocess.run([str(asset)], check=False).returncode == 23


def test_changed_executable_bytes_rejected(tmp_path: Path):
    asset, digest = fixture_asset(tmp_path)
    asset.chmod(0o755)
    asset.write_bytes(b"#!/bin/sh\nexit 0\n")
    with pytest.raises(EpisodeExecutableError, match="EXECUTABLE_HASH_MISMATCH"):
        validate_episode_executable(asset, digest)


def test_symlink_and_invalid_pin_rejected(tmp_path: Path):
    asset, digest = fixture_asset(tmp_path)
    asset.chmod(0o755)
    alias = tmp_path / "alias"
    alias.symlink_to(asset)
    with pytest.raises(EpisodeExecutableError, match="EXECUTABLE_NOT_REGULAR_FILE"):
        validate_episode_executable(alias, digest)
    with pytest.raises(EpisodeExecutableError, match="INVALID_EXECUTABLE_SHA256"):
        validate_episode_executable(asset, "untrusted")
