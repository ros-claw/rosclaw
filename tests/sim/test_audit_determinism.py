"""确定性类 audit 与引擎行为测试（PR-MH4，规格 §18/§19/§56）。"""

from __future__ import annotations

import pytest

from rosclaw.sim.audit import engine


def test_a18_reset_determinism_green(fixture_backend) -> None:
    backend, ref = fixture_backend("fixed_models", "08_unstable_controller")
    result = backend.audit(ref.model_ref, checks=["A18_reset_determinism"])
    assert result.checks["A18_reset_determinism"]["status"] == "PASS"


def test_a19_replay_determinism_green(fixture_backend) -> None:
    backend, ref = fixture_backend("fixed_models", "08_unstable_controller")
    result = backend.audit(ref.model_ref, checks=["A19_replay_determinism"])
    assert result.checks["A19_replay_determinism"]["status"] == "PASS"


def test_a20_state_model_mismatch_red(fixture_backend) -> None:
    backend, ref = fixture_backend("fixed_models", "01_no_collision")
    state_ref = backend.initial_state(ref.model_ref)
    patched = backend.patch_model(
        ref.model_ref,
        [
            {
                "op": "set",
                "target": {"type": "joint", "name": "j1"},
                "field": "damping",
                "value": 2.0,
            }
        ],
    )
    result = backend.audit(
        patched.new_model_ref, checks=["A20_state_model_mismatch"], state_ref=state_ref
    )
    check = result.checks["A20_state_model_mismatch"]
    assert check["status"] == "FAIL"
    assert "CROSS_MODEL_REF" in check["violations"][0]["error"]


def test_a20_matching_state_green(fixture_backend) -> None:
    backend, ref = fixture_backend("fixed_models", "01_no_collision")
    state_ref = backend.initial_state(ref.model_ref)
    result = backend.audit(ref.model_ref, checks=["A20_state_model_mismatch"], state_ref=state_ref)
    assert result.checks["A20_state_model_mismatch"]["status"] == "PASS"


# --- 引擎行为 ---------------------------------------------------------------


def test_audit_result_stored_and_typed(fixture_backend) -> None:
    backend, ref = fixture_backend("fixed_models", "01_no_collision")
    result = backend.audit(ref.model_ref, checks=["A18_reset_determinism"])
    assert result.status == "PASS"
    assert result.audit_ref.startswith("simadt_")
    record = backend.store.get(result.audit_ref)
    assert record["kind"] == "audit_result"
    assert record["model_ref"] == ref.model_ref
    assert record["model_digest"] == ref.model_digest


def test_audit_unknown_check_rejected(fixture_backend) -> None:
    backend, ref = fixture_backend("fixed_models", "01_no_collision")
    with pytest.raises(ValueError, match="AUDIT_CHECK_UNKNOWN"):
        backend.audit(ref.model_ref, checks=["A99_no_such_check"])


def test_audit_check_crash_is_error_not_pass(
    fixture_backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """审计器自身故障不能伪装成 PASS（fail closed）。"""

    def boom(ctx):  # noqa: ANN001, ANN202
        raise RuntimeError("simulated check crash")

    monkeypatch.setitem(engine.CHECKS, "A18_reset_determinism", boom)
    backend, ref = fixture_backend("fixed_models", "01_no_collision")
    result = backend.audit(ref.model_ref, checks=["A18_reset_determinism"])
    assert result.status == "FAIL"
    assert result.checks["A18_reset_determinism"]["status"] == "ERROR"


def test_audit_full_strict_profile_on_fixed_model(fixture_backend) -> None:
    """fixed 模型全量 strict profile：核心八项 + 确定性项全过。"""
    backend, ref = fixture_backend("fixed_models", "08_unstable_controller")
    result = backend.audit(ref.model_ref)
    assert result.status == "PASS", result.violations
    for name in (
        "A01_collision_coverage",
        "A02_explicit_mass",
        "A03_servo_hold",
        "A05_initial_penetration",
        "A15_nan_inf",
        "A18_reset_determinism",
        "A19_replay_determinism",
    ):
        assert result.checks[name]["status"] == "PASS"
