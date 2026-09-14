"""接触类 audit 红绿 fixture 测试（PR-MH4，规格 §17/§40/§41）。"""

from __future__ import annotations

import pytest


def test_a05_initial_penetration_red(fixture_backend) -> None:
    backend, ref = fixture_backend("broken_models", "03_initial_penetration")
    result = backend.audit(ref.model_ref, checks=["A05_initial_penetration"])
    check = result.checks["A05_initial_penetration"]
    assert check["status"] == "FAIL"
    assert check["min_distance_m"] < -1e-4


def test_a05_initial_penetration_green(fixture_backend) -> None:
    backend, ref = fixture_backend("fixed_models", "03_initial_penetration")
    result = backend.audit(ref.model_ref, checks=["A05_initial_penetration"])
    assert result.checks["A05_initial_penetration"]["status"] == "PASS"


def test_a06_sequence_penetration_red(fixture_backend) -> None:
    """start/end 清楚、中途穿墙——只查端点抓不到，必须逐步检查。"""
    backend, ref = fixture_backend("broken_models", "07_sequence_penetration")
    trace = backend.rollout(ref.model_ref, controller={"ctrl_series": [[1.0]] * 400})
    result = backend.audit(
        ref.model_ref, checks=["A06_sequence_penetration"], trace_ref=trace.trace_ref
    )
    check = result.checks["A06_sequence_penetration"]
    assert check["status"] == "FAIL"
    assert check["min_distance_m"] < -1e-3


def test_a06_sequence_penetration_green(fixture_backend) -> None:
    backend, ref = fixture_backend("fixed_models", "07_sequence_penetration")
    trace = backend.rollout(ref.model_ref, controller={"ctrl_series": [[1.0]] * 400})
    result = backend.audit(
        ref.model_ref, checks=["A06_sequence_penetration"], trace_ref=trace.trace_ref
    )
    assert result.checks["A06_sequence_penetration"]["status"] == "PASS"


def test_audit_cross_model_trace_rejected(fixture_backend) -> None:
    backend, ref = fixture_backend("fixed_models", "07_sequence_penetration")
    trace = backend.rollout(ref.model_ref, controller={"hold": True}, steps=10)
    _, other = fixture_backend("fixed_models", "01_no_collision")
    with pytest.raises(ValueError, match="CROSS_MODEL_REF"):
        backend.audit(other.model_ref, trace_ref=trace.trace_ref)
