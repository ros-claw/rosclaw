"""控制/动力学类 audit 红绿 fixture 测试（PR-MH4，规格 §17/§40/§41）。"""

from __future__ import annotations


def test_a03_servo_hold_red(fixture_backend) -> None:
    backend, ref = fixture_backend("broken_models", "04_servo_droop")
    result = backend.audit(ref.model_ref, checks=["A03_servo_hold"])
    check = result.checks["A03_servo_hold"]
    assert check["status"] == "FAIL"
    assert check["max_angular_drift_rad"] > 0.017  # 超 1 度


def test_a03_servo_hold_green(fixture_backend) -> None:
    backend, ref = fixture_backend("fixed_models", "04_servo_droop")
    result = backend.audit(ref.model_ref, checks=["A03_servo_hold"])
    assert result.checks["A03_servo_hold"]["status"] == "PASS"


def test_a04_link_continuity_red(fixture_backend) -> None:
    backend, ref = fixture_backend("broken_models", "05_broken_prismatic")
    result = backend.audit(ref.model_ref, checks=["A04_link_continuity"])
    check = result.checks["A04_link_continuity"]
    assert check["status"] == "FAIL"
    assert check["violations"][0]["gap_m"] > 5e-3


def test_a04_link_continuity_green(fixture_backend) -> None:
    backend, ref = fixture_backend("fixed_models", "05_broken_prismatic")
    result = backend.audit(ref.model_ref, checks=["A04_link_continuity"])
    assert result.checks["A04_link_continuity"]["status"] == "PASS"


def test_a15_a16_a17_unstable_red(fixture_backend) -> None:
    backend, ref = fixture_backend("broken_models", "08_unstable_controller")
    result = backend.audit(
        ref.model_ref,
        checks=["A15_nan_inf", "A16_physics_divergence", "A17_energy_explosion"],
    )
    assert result.status == "FAIL"
    assert result.checks["A16_physics_divergence"]["status"] == "FAIL"
    assert result.checks["A17_energy_explosion"]["status"] == "FAIL"


def test_a15_nan_trace_red(fixture_backend) -> None:
    """A15 红：trace 中含 NaN 的状态必须被抓（float64 下物理自然产生
    真 NaN 极难，数据完整性是 A15 的核心检测面）。"""
    import hashlib

    from rosclaw.contracts.common import canonical_json
    from rosclaw.sim.refs import make_ref

    backend, ref = fixture_backend("fixed_models", "01_no_collision")
    poisoned = {
        "kind": "simulation_trace",
        "model_ref": ref.model_ref,
        "model_digest": ref.model_digest,
        "seed": 0,
        "controller": {"hold": True},
        "steps": 1,
        "timestep_s": 0.002,
        "duration_s": 0.002,
        "states_digest": "sha256:poisoned",
        "states": [{"t": 0.0, "qpos": [float("nan")], "qvel": [0.0], "ctrl": [0.0]}],
    }
    forged_ref = make_ref(
        "simtrc", hashlib.sha256(canonical_json(poisoned).encode("utf-8")).hexdigest()
    )
    backend.store.put("traces", poisoned, ref=forged_ref)

    result = backend.audit(ref.model_ref, checks=["A15_nan_inf"], trace_ref=forged_ref)
    check = result.checks["A15_nan_inf"]
    assert check["status"] == "FAIL"
    assert check["violations"][0]["reason"] == "non_finite_trace_state"


def test_a15_a16_a17_stable_green(fixture_backend) -> None:
    backend, ref = fixture_backend("fixed_models", "08_unstable_controller")
    result = backend.audit(
        ref.model_ref,
        checks=["A15_nan_inf", "A16_physics_divergence", "A17_energy_explosion"],
    )
    assert result.status == "PASS"
