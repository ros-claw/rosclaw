"""几何类 audit 红绿 fixture 测试（PR-MH4，规格 §17/§40/§41）。

每个 audit 至少 1 red fixture（必须被抓）+ 1 green fixture（必须通过）——
不为测试通过调松阈值，先修模型。
"""

from __future__ import annotations


def test_a01_collision_coverage_red(fixture_backend) -> None:
    backend, ref = fixture_backend("broken_models", "01_no_collision")
    result = backend.audit(ref.model_ref, checks=["A01_collision_coverage"])
    assert result.status == "FAIL"
    check = result.checks["A01_collision_coverage"]
    assert check["status"] == "FAIL"
    assert check["violations"][0]["reason"] == "collision_filtered_away"


def test_a01_collision_coverage_green(fixture_backend) -> None:
    backend, ref = fixture_backend("fixed_models", "01_no_collision")
    result = backend.audit(ref.model_ref, checks=["A01_collision_coverage"])
    assert result.checks["A01_collision_coverage"]["status"] == "PASS"


def test_a02_explicit_mass_red(fixture_backend) -> None:
    backend, ref = fixture_backend("broken_models", "02_default_mass")
    result = backend.audit(ref.model_ref, checks=["A02_explicit_mass"])
    check = result.checks["A02_explicit_mass"]
    assert check["status"] == "FAIL"
    assert check["violations"][0]["reason"] == "implicit_mass_or_density"


def test_a02_explicit_mass_green(fixture_backend) -> None:
    backend, ref = fixture_backend("fixed_models", "02_default_mass")
    result = backend.audit(ref.model_ref, checks=["A02_explicit_mass"])
    assert result.checks["A02_explicit_mass"]["status"] == "PASS"


def test_a07_hidden_self_overlap_red(fixture_backend) -> None:
    backend, ref = fixture_backend("broken_models", "06_hidden_self_overlap")
    result = backend.audit(ref.model_ref, checks=["A07_undeclared_self_overlap"])
    check = result.checks["A07_undeclared_self_overlap"]
    assert check["status"] == "FAIL"
    assert check["violations"][0]["distance_m"] < -1e-3


def test_a07_declared_exclusion_green(fixture_backend) -> None:
    backend, ref = fixture_backend("fixed_models", "06_hidden_self_overlap")
    result = backend.audit(ref.model_ref, checks=["A07_undeclared_self_overlap"])
    assert result.checks["A07_undeclared_self_overlap"]["status"] == "PASS"


def test_a08_marker_floating_red(fixture_backend) -> None:
    backend, ref = fixture_backend("broken_models", "09_marker_floating")
    result = backend.audit(ref.model_ref, checks=["A08_marker_grounding"])
    check = result.checks["A08_marker_grounding"]
    assert check["status"] == "FAIL"
    assert check["violations"][0]["reason"] == "floating"


def test_a08_marker_grounded_green(fixture_backend) -> None:
    backend, ref = fixture_backend("fixed_models", "09_marker_floating")
    result = backend.audit(ref.model_ref, checks=["A08_marker_grounding"])
    assert result.checks["A08_marker_grounding"]["status"] == "PASS"
