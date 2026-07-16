"""RH56 sandbox preflight unit tests."""

from __future__ import annotations

from pathlib import Path

from rosclaw.body.action_mapping.schema import BodyActionSpace, MappedAction
from rosclaw.body.rh56.calibration import load_rh56_calibration
from rosclaw.body.rh56.sandbox import run_rh56_sandbox_preflight
from rosclaw.body.rh56.transport_profile import load_transport_profile

CONFIGS = Path(__file__).resolve().parents[3] / "configs"
NAMES = ["little", "ring", "middle", "index", "thumb", "thumb_rot"]


def _stack():
    profile = load_transport_profile(CONFIGS / "rh56_right_rs485_v1.yaml")
    calibration = load_rh56_calibration(CONFIGS / "rh56_right_01_calibration.yaml")
    space = BodyActionSpace(
        body_id="rh56_mock",
        representation="joint_position",
        joint_names=NAMES,
        units=["raw_device_unit"] * 6,
    )
    return profile, calibration, space


def _mapped(values: list[float]) -> MappedAction:
    from rosclaw.body.action_mapping.schema import MappingCompatibility

    return MappedAction(
        body_action_values=list(values),
        body_joint_names=NAMES,
        compatibility=MappingCompatibility.EXACT,
        blocked=False,
        block_reasons=[],
        warnings=[],
    )


def test_sandbox_allows_in_range_action() -> None:
    profile, calibration, space = _stack()
    result = run_rh56_sandbox_preflight(
        _mapped([1000.0, 900.0, 800.0, 700.0, 600.0, 500.0]),
        space,
        profile=profile,
        calibration=calibration,
    )
    assert result["is_safe"] is True
    assert result["decision"] == "ALLOW"


def test_sandbox_blocks_out_of_profile_range() -> None:
    profile, calibration, space = _stack()
    result = run_rh56_sandbox_preflight(
        _mapped([1200.0] * 6),
        space,
        profile=profile,
    )
    assert result["is_safe"] is False
    assert any("profile_range" in v for v in result["violations"])


def test_sandbox_blocks_out_of_calibration_safe_range() -> None:
    profile, calibration, space = _stack()
    # safe_min_raw = 100 for all actuators.
    result = run_rh56_sandbox_preflight(
        _mapped([50.0] * 6),
        space,
        profile=profile,
        calibration=calibration,
    )
    assert result["is_safe"] is False
    assert any("calibration_safe_range" in v for v in result["violations"])


def test_sandbox_blocks_nan() -> None:
    profile, calibration, space = _stack()
    result = run_rh56_sandbox_preflight(
        _mapped([float("nan")] * 6),
        space,
        profile=profile,
    )
    assert result["is_safe"] is False
    assert any("nan_inf" in v for v in result["violations"])


def test_sandbox_step_delta_limit() -> None:
    profile, calibration, space = _stack()
    current = [1000.0] * 6
    ok = run_rh56_sandbox_preflight(
        _mapped([980.0] * 6),
        space,
        profile=profile,
        current_positions=current,
        max_step_delta_raw=30.0,
    )
    assert ok["is_safe"] is True
    blocked = run_rh56_sandbox_preflight(
        _mapped([900.0] * 6),
        space,
        profile=profile,
        current_positions=current,
        max_step_delta_raw=30.0,
    )
    assert blocked["is_safe"] is False
    assert any("step_delta" in v for v in blocked["violations"])
