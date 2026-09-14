"""db doctor pyseekdb compat matrix tests + startup version warning."""

from __future__ import annotations

import logging
from unittest.mock import patch

from rosclaw.storage.cli import _pyseekdb_compat_checks


def _run(installed: str | None):
    checks: list[tuple[str, str, bool]] = []
    issues: list[str] = []
    result: dict = {"seekdb": {}}
    with patch("importlib.metadata.version") as mv:
        if installed is None:
            from importlib.metadata import PackageNotFoundError

            mv.side_effect = PackageNotFoundError("pyseekdb")
        else:
            mv.return_value = installed
        _pyseekdb_compat_checks(checks, issues, result)
    return checks, issues, result["seekdb"]["pyseekdb_compat"]


def test_validated_version_passes():
    checks, issues, compat = _run("1.3.0")
    assert checks == [("pyseekdb compat", "1.3.0 (validated)", True)]
    assert issues == []
    assert compat["status"] == "validated"


def test_known_incompatible_fails_with_guidance():
    checks, issues, compat = _run("1.4.0")
    assert checks[0][2] is False
    assert "1.3.0" in issues[0]
    assert compat["status"] == "incompatible"


def test_untested_version_warns_but_passes():
    checks, issues, compat = _run("1.3.1")
    assert checks[0][2] is True
    assert "untested" in checks[0][1]
    assert compat["status"] == "untested"
    assert issues


def test_missing_sdk_fails():
    checks, issues, compat = _run(None)
    assert checks[0][2] is False
    assert compat["status"] == "missing"


def test_startup_warning_fires_once_per_process(caplog):
    import rosclaw.storage.seekdb_native as native

    native._warned_pyseekdb_version = False
    with patch("importlib.metadata.version", return_value="1.3.1"), caplog.at_level(logging.WARNING):
        native._warn_on_unvalidated_pyseekdb(object())
        native._warn_on_unvalidated_pyseekdb(object())
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "outside the validated version matrix" in warnings[0].message
    native._warned_pyseekdb_version = True  # restore


def test_known_bad_version_fails_fast():
    """PR-SDB-140-1 (outline §五): pyseekdb 1.4.0 must raise, not log."""
    import rosclaw.storage.seekdb_native as native

    native._warned_pyseekdb_version = False
    with patch("importlib.metadata.version", return_value="1.4.0"):
        import pytest

        with pytest.raises(RuntimeError, match="KNOWN-INCOMPATIBLE"):
            native._warn_on_unvalidated_pyseekdb(object())
    native._warned_pyseekdb_version = True  # restore


def test_known_bad_override_env_allows_with_error_log(caplog, monkeypatch):
    """The lab escape hatch: ROSCLAW_ALLOW_KNOWN_BAD_SEEKDB=1."""
    import rosclaw.storage.seekdb_native as native

    monkeypatch.setenv("ROSCLAW_ALLOW_KNOWN_BAD_SEEKDB", "1")
    native._warned_pyseekdb_version = False
    with patch("importlib.metadata.version", return_value="1.4.0"), caplog.at_level(logging.ERROR):
        native._warn_on_unvalidated_pyseekdb(object())
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "KNOWN-INCOMPATIBLE" in errors[0].message
    native._warned_pyseekdb_version = True  # restore


def test_startup_no_warning_for_validated(caplog):
    import rosclaw.storage.seekdb_native as native

    native._warned_pyseekdb_version = False
    with patch("importlib.metadata.version", return_value="1.3.0"), caplog.at_level(logging.WARNING):
        native._warn_on_unvalidated_pyseekdb(object())
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == []
    native._warned_pyseekdb_version = True
