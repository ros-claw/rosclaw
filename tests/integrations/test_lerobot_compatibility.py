"""Tests for the LeRobot policy compatibility matrix.

These tests do not require a real LeRobot runtime.
"""

from __future__ import annotations

from rosclaw.integrations.lerobot.compatibility import (
    POLICY_COMPATIBILITY_MATRIX,
    build_compatibility_report,
    classify_compatibility_level,
    format_compatibility_text,
    get_policy_compatibility,
)
from rosclaw.integrations.lerobot.smoke_report import SmokeReport


def test_get_policy_compatibility_known_types():
    """Known policy families return expected capability flags."""
    act = get_policy_compatibility("act")
    assert act.inspect is True
    assert act.load_test is True
    assert act.infer is True
    assert act.body_mapping_required is True
    assert act.body_compatible is False

    vqbet = get_policy_compatibility("vqbet")
    assert vqbet.inspect is True
    assert vqbet.load_test is False
    assert vqbet.infer is False


def test_get_policy_compatibility_unknown_type():
    """Unknown policy families default to inspect-only."""
    unknown = get_policy_compatibility("some_future_policy")
    assert unknown.inspect is True
    assert unknown.load_test is False
    assert unknown.infer is False


def test_get_policy_compatibility_none():
    """``None`` policy type is unsupported."""
    none_compat = get_policy_compatibility(None)
    assert none_compat.inspect is False
    assert none_compat.load_test is False
    assert none_compat.infer is False


def test_classify_compatibility_level_without_report():
    """Without a smoke report, only the ACT path is marked infer-capable."""
    assert classify_compatibility_level("act") == "infer_ok"
    assert classify_compatibility_level("diffusion") == "inspect_ok"
    assert classify_compatibility_level("vqbet") == "inspect_ok"
    assert classify_compatibility_level("tdmpc") == "inspect_ok"
    assert classify_compatibility_level("unknown_future") == "inspect_ok"
    assert classify_compatibility_level(None) == "unsupported"


def test_classify_compatibility_level_validated_by_report():
    """A successful smoke report upgrades the level to validated."""
    report = SmokeReport(
        status="ok",
        policy={"policy_type": "act"},
    )
    assert classify_compatibility_level("act", report=report) == "validated"


def test_classify_compatibility_level_failed_report_does_not_validate():
    """A failed smoke report does not upgrade the level."""
    report = SmokeReport(
        status="error",
        policy={"policy_type": "act"},
    )
    assert classify_compatibility_level("act", report=report) == "infer_ok"


def test_build_compatibility_report_full_matrix():
    """The full-matrix report contains every known policy family."""
    report = build_compatibility_report()
    matrix = report["matrix"]
    types = {row["policy_type"] for row in matrix}
    assert types == set(POLICY_COMPATIBILITY_MATRIX.keys())
    assert "levels" in report


def test_build_compatibility_report_single_type():
    """Focusing on a single policy type returns one row."""
    report = build_compatibility_report(policy_type="act")
    assert report["policy_type"] == "act"
    assert report["level"] == "infer_ok"
    assert report["inspect"] is True
    assert report["infer"] is True
    assert report["body_mapping_required"] is True
    assert report["body_compatible"] is False


def test_format_compatibility_text_includes_matrix_and_latest():
    """Text rendering includes the matrix and latest smoke report."""
    report = build_compatibility_report()
    text = format_compatibility_text(report)
    assert "act" in text.lower()
    assert "infer_ok" in text
    assert "LeRobot Policy Compatibility Matrix" in text


def test_format_compatibility_text_single_type():
    """Focusing on a single type still renders the header."""
    report = build_compatibility_report(policy_type="act")
    text = format_compatibility_text(report)
    assert "LeRobot Policy Compatibility Matrix" in text
    assert "act" in text.lower()
