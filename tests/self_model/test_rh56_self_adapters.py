"""PR-PE-4 tests: self protocols, RH56/dual/D435i adapters, snapshot,
residuals, uncertainty — plus the real-telemetry replay acceptance."""

from __future__ import annotations

import pytest

from rosclaw.self_model.adapters.dual_rh56 import D435iSensorSelfAdapter, DualHandSelfAdapter
from rosclaw.self_model.adapters.rh56 import (
    BodyHashMismatchError,
    RH56ForwardPrior,
    RH56HandSelfAdapter,
    bind_model_to_body,
)
from rosclaw.self_model.protocols import SelfObservation
from rosclaw.self_model.residuals import (
    ChannelResidualEncoder,
    shift_flags,
)
from rosclaw.self_model.snapshot import SelfSnapshot, calibrate_uncertainty

JOINTS = ("little", "ring", "middle", "index", "thumb", "thumb_rot")


def _adapter(body_id: str = "rh56_right_01") -> RH56HandSelfAdapter:
    return RH56HandSelfAdapter(body_id)


def _prior(adapter: RH56HandSelfAdapter) -> RH56ForwardPrior:
    return RH56ForwardPrior(adapter.body_id(), adapter.body_hash())


def test_first_order_position_response() -> None:
    adapter = _adapter()
    prior = _prior(adapter)
    # Motion evidence toward the target → first-order approach.
    state = {"pos_index": 400.0, "vel_index": 120.0, "temp_max": 44.0}
    action = {"target_index": 600.0, "dt_s": 0.16, "gestures": 1.0}
    prediction = prior.predict(state, action)
    # tau_index=0.28 (session-calibrated), dt=0.16 → exp(-0.571) ≈ 0.565
    # of the 200-unit gap remains.
    import math

    remaining = math.exp(-0.16 / 0.28)
    expected = 600.0 + (400.0 - 600.0) * remaining
    assert abs(prediction.channels["next_pos_index"] - expected) < 0.5
    assert prediction.channels["tracking_error_index"] == pytest.approx(
        abs(400.0 - 600.0) * remaining, abs=0.5
    )
    assert prediction.channels["time_to_reach_index"] == pytest.approx(0.28 * 2.9957, abs=0.01)
    assert prediction.analytical_only
    # Thermal: heat in minus cooling out.
    assert "temp_delta" in prediction.channels


def test_stale_register_without_motion_predicts_hold() -> None:
    """Real-telemetry driven: angle_set holds stale values between
    gestures while the hand holds position — the prior must predict hold
    (measured: toward-register was 5× worse than hold)."""
    adapter = _adapter()
    prior = _prior(adapter)
    prediction = prior.predict(
        {"pos_index": 1000.0, "vel_index": 0.0}, {"target_index": 70.0, "dt_s": 0.5}
    )
    assert prediction.channels["next_pos_index"] == pytest.approx(1000.0)
    assert prediction.channels["time_to_reach_index"] == 0.0
    # No velocity channel at all → also hold (unknown motion ≠ motion).
    prediction2 = prior.predict({"pos_index": 1000.0}, {"target_index": 70.0, "dt_s": 0.5})
    assert prediction2.channels["next_pos_index"] == pytest.approx(1000.0)


def test_no_target_means_hold_position() -> None:
    adapter = _adapter()
    prior = _prior(adapter)
    prediction = prior.predict({"pos_thumb": 250.0}, {"dt_s": 0.5})
    assert prediction.channels["next_pos_thumb"] == pytest.approx(250.0)
    assert prediction.channels["time_to_reach_thumb"] == 0.0


def test_body_hash_binding_rejects_mutation() -> None:
    adapter = _adapter()
    prior = _prior(adapter)
    assert bind_model_to_body(prior, adapter) is prior
    other = RH56HandSelfAdapter("rh56_right_01", firmware="different")
    with pytest.raises(BodyHashMismatchError):
        bind_model_to_body(prior, other)


def test_dual_hand_skew_channels_and_hash() -> None:
    dual = DualHandSelfAdapter(_adapter("rh56_left_01"), _adapter(), body_id="rh56_dual_01")
    skew = dual.skew_channels(
        {"angle_actual": {"index": 400}, "temperature_c": {"index": 44}},
        {"angle_actual": {"index": 430}, "temperature_c": {"index": 46}},
    )
    assert skew["mirror_skew_index"] == 30.0
    assert skew["temperature_asymmetry"] == 2.0
    assert dual.body_hash() != dual.left.body_hash()
    swapped = DualHandSelfAdapter(
        RH56HandSelfAdapter("rh56_left_01", firmware="new"), _adapter(), body_id="rh56_dual_01"
    )
    assert swapped.body_hash() != dual.body_hash()


def test_d435i_freshness_state_machine() -> None:
    cam = D435iSensorSelfAdapter("d435i_231122070092", camera_pose_hash="campose_x")
    fresh = cam.freshness_state(frame_age_ms=30.0, rgb_depth_skew_ms=5.0, consecutive_missing=0)
    assert fresh["state"] == "FRESH" and fresh["reliability"] == 1.0
    stale = cam.freshness_state(frame_age_ms=600.0, rgb_depth_skew_ms=5.0, consecutive_missing=0)
    assert stale["state"] == "STALE" and stale["reliability"] == 0.0
    missing = cam.freshness_state(frame_age_ms=30.0, rgb_depth_skew_ms=5.0, consecutive_missing=3)
    assert missing["state"] == "STALE"
    # Unknown skew: honest reliability haircut, never full confidence.
    unknown = cam.freshness_state(frame_age_ms=30.0, rgb_depth_skew_ms=None, consecutive_missing=0)
    assert unknown["reliability"] == 0.8


def test_residual_encoding_and_shift_flags() -> None:
    adapter = _adapter()
    prior = _prior(adapter)
    prediction = prior.predict(
        {"pos_index": 400.0, "temp_max": 44.0}, {"target_index": 600.0, "dt_s": 0.16}
    )
    observation = SelfObservation(
        channels={
            "next_pos_index": prediction.channels["next_pos_index"] + 100.0,  # big miss
            "temp_delta": prediction.channels["temp_delta"] + 3.0,
        },
        timestamp_ns=1,
        source="telemetry",
    )
    encoder = ChannelResidualEncoder()
    residuals = encoder.encode_families(prediction, observation)
    assert residuals.channels["joint_state"] == pytest.approx(100.0)
    flags = shift_flags(residuals)
    assert flags["joint_state"] is True  # 100 > 60 threshold
    assert flags.get("thermal", False) is True  # 3.0 > 2.0


def test_snapshot_hash_and_mutation_report() -> None:
    snap = SelfSnapshot(
        body_id="rh56_right_01",
        body_hash="body_a",
        health={"temperature_max": 44},
        perception_confidence=0.9,
        regime_label="COLD_HEALTHY",
        capabilities={"rps": "compatible"},
        forward_model_hash="fwm_a",
        prediction_uncertainty={"next_pos_index": 15.0},
        active_policy_hash="pol_a",
        agency_summary={"SELF_CAUSED": 3},
        sequence=1,
    )
    mutated = SelfSnapshot(
        body_id="rh56_right_01",
        body_hash="body_b",
        health={"temperature_max": 44},
        perception_confidence=0.9,
        regime_label="COLD_HEALTHY",
        capabilities={"rps": "compatible"},
        forward_model_hash="fwm_a",
        prediction_uncertainty={"next_pos_index": 15.0},
        active_policy_hash="pol_a",
        agency_summary={"SELF_CAUSED": 3},
        sequence=2,
    )
    assert snap.snapshot_hash != mutated.snapshot_hash
    assert "body_hash" in snap.mutation_report(mutated)


def test_uncertainty_calibration_honest_until_enough_samples() -> None:
    prior_std = {"next_pos_index": 15.0}
    few = [{"next_pos_index": 3.0}] * 10
    result = calibrate_uncertainty(few, prior_std, min_samples=30)
    assert not result.calibrated
    assert result.coverage == 1.0
    many = [{"next_pos_index": 3.0}] * 40
    result2 = calibrate_uncertainty(many, prior_std, min_samples=30)
    assert result2.calibrated


# ------------------------------------------------ real telemetry replay (read-only)


def test_rh56_prior_replay_on_real_session() -> None:
    """Independent Hardware (read-only): replay the first-order prior over
    a REAL canary session's telemetry and compare its next-position
    residuals against the naive 'no motion' baseline.  The prior must be
    better during gestures — otherwise it is not a model."""
    import json
    from pathlib import Path

    session = Path(
        "/home/nvidia/.rosclaw/acceptance/evo_rps/evo_rps_2026_01/practice/sessions/prac_20260730T024004Z_3260ea"
    )
    if not (session / "raw" / "events.jsonl").is_file():
        pytest.skip("real session not present on this machine")
    telemetry = []
    with (session / "raw" / "events.jsonl").open() as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event_type") == "rps.telemetry":
                telemetry.append(event["payload"])
    if len(telemetry) < 50:
        pytest.skip("not enough telemetry")

    adapter = _adapter()
    prior = _prior(adapter)
    joints = ("index", "thumb")
    prior_sq = 0.0
    naive_sq = 0.0
    motion_prior_sq = 0.0
    motion_naive_sq = 0.0
    motion_n = 0
    n = 0
    prev_prev = None
    for prev, cur in zip(telemetry, telemetry[1:], strict=False):
        dt = float(cur["timestamp"]) - float(prev["timestamp"])
        if not 0.05 < dt < 2.0:
            prev_prev = prev
            continue
        prev_right = prev.get("right") or {}
        cur_right = cur.get("right") or {}
        prev_pos = prev_right.get("angle_actual") or {}
        target = prev_right.get("angle_set") or {}
        cur_pos = cur_right.get("angle_actual") or {}
        # Measured velocity from the previous step (motion evidence).
        old_pos = (prev_prev.get("right") or {}).get("angle_actual") if prev_prev else None
        dt_prev = float(prev["timestamp"]) - float(prev_prev["timestamp"]) if prev_prev else None
        state = {f"pos_{j}": float(prev_pos[j]) for j in joints if j in prev_pos}
        if old_pos and dt_prev and dt_prev > 0.01:
            for j in joints:
                if j in old_pos and j in prev_pos:
                    state[f"vel_{j}"] = (float(prev_pos[j]) - float(old_pos[j])) / dt_prev
        action = {f"target_{j}": float(target.get(j, prev_pos[j])) for j in joints if j in prev_pos}
        action["dt_s"] = dt
        prediction = prior.predict(state, action)
        for j in joints:
            if j not in cur_pos or f"next_pos_{j}" not in prediction.channels:
                continue
            prior_sq += (float(cur_pos[j]) - prediction.channels[f"next_pos_{j}"]) ** 2
            naive_sq += (float(cur_pos[j]) - float(prev_pos[j])) ** 2
            n += 1
            vel = state.get(f"vel_{j}", 0.0)
            if abs(vel) > 30.0:
                motion_prior_sq += (float(cur_pos[j]) - prediction.channels[f"next_pos_{j}"]) ** 2
                motion_naive_sq += (float(cur_pos[j]) - float(prev_pos[j])) ** 2
                motion_n += 1
        prev_prev = prev
    assert n > 30
    prior_rmse = (prior_sq / n) ** 0.5
    naive_rmse = (naive_sq / n) ** 0.5
    # Overall the two-mode prior must not be worse than hold.
    assert prior_rmse <= naive_rmse * 1.01, f"prior {prior_rmse:.1f} >> naive {naive_rmse:.1f}"
    # During measured motion it must be strictly better.
    assert motion_n > 5
    motion_prior = (motion_prior_sq / motion_n) ** 0.5
    motion_naive = (motion_naive_sq / motion_n) ** 0.5
    assert motion_prior < motion_naive, (
        f"motion prior {motion_prior:.1f} >= naive {motion_naive:.1f}"
    )
