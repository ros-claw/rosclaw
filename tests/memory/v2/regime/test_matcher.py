"""Applicability envelope + RegimeMatcher tests (PR-MEM-6, v4 §13)."""

from __future__ import annotations

from rosclaw.memory.seekdb_client import InMemoryKnowledgeStore
from rosclaw.memory.v2.regime import (
    ApplicabilityEnvelope,
    ApplicabilityStore,
    EnvelopeType,
    MatcherConfig,
    OperatingRegime,
    RegimeLabel,
    RegimeMatcher,
    empty_regime,
    envelope_from_regime,
    interval_distance,
)

T0 = 1_700_000_000.0


def _regime(**overrides) -> OperatingRegime:
    regime = empty_regime(robot_id="r1", body_id="rh56_right_01", task_id="rh56_rps", now=T0)
    regime.regime_label = RegimeLabel.THERMAL_TRACKING_DEGRADATION.value
    regime.temperature_c = 57.0
    regime.temperature_slope_c_per_min = 0.25
    regime.session_elapsed_sec = 5400.0
    regime.cumulative_action_count = 900
    regime.position_error_p95 = 22.0
    regime.recent_failure_rate = 0.12
    regime.recent_invalid_rate = 0.10
    regime.calibration_hash = "cal_abc"
    regime.control_profile_hash = "ctl_xyz"
    regime.joint_name = "middle"
    for key, value in overrides.items():
        setattr(regime, key, value)
    return regime


def _thermal_envelope(**overrides) -> ApplicabilityEnvelope:
    envelope = ApplicabilityEnvelope(
        memory_id="mem_slow_down",
        body_ids=["rh56_right_01"],
        task_ids=["rh56_rps"],
        joints=["middle"],
        temperature_min=55.0,
        temperature_max=60.0,
        temperature_slope_min=0.1,
        temperature_slope_max=0.5,
        elapsed_sec_min=3600.0,
        elapsed_sec_max=7200.0,
        regime_labels=[RegimeLabel.THERMAL_TRACKING_DEGRADATION.value],
        envelope_type=EnvelopeType.VALIDATED.value,
        evidence_count=4,
        success_count=4,
        confidence=0.85,
        required_features=["temperature_c"],
    )
    for key, value in overrides.items():
        setattr(envelope, key, value)
    return envelope


# ---------------------------------------------------------------------------
# interval_distance
# ---------------------------------------------------------------------------


def test_interval_distance_semantics() -> None:
    assert interval_distance(None, 1.0, 2.0, scale=1.0) is None  # unknown ≠ match
    assert interval_distance(5.0, None, None, scale=1.0) == 0.0  # unbounded
    assert interval_distance(1.5, 1.0, 2.0, scale=1.0) == 0.0  # inside
    assert interval_distance(0.5, 1.0, 2.0, scale=0.5) == 1.0  # below
    assert interval_distance(3.0, 1.0, 2.0, scale=2.0) == 0.5  # above


# ---------------------------------------------------------------------------
# RegimeMatcher
# ---------------------------------------------------------------------------


def test_matched_regime_is_applicable() -> None:
    matcher = RegimeMatcher()
    result = matcher.match("mem_slow_down", [_thermal_envelope()], _regime())
    assert result.applicable is True
    assert result.score >= matcher.config.abstain_below
    assert result.envelope_type == EnvelopeType.VALIDATED.value
    assert result.matched_envelope_id is not None


def test_healthy_regime_thermal_memory_not_applicable() -> None:
    """v4 核心负测试: 48–50°C 健康工况下热退化记忆不得适用."""
    matcher = RegimeMatcher()
    healthy = _regime(
        regime_label=RegimeLabel.COLD_HEALTHY.value,
        temperature_c=49.0,
        temperature_slope_c_per_min=0.01,
        session_elapsed_sec=600.0,
        cumulative_action_count=80,
        position_error_p95=5.0,
        recent_failure_rate=0.0,
        recent_invalid_rate=0.02,
    )
    result = matcher.match("mem_slow_down", [_thermal_envelope()], healthy)
    assert result.applicable is False
    assert result.score < matcher.config.abstain_below
    # The distance features must show WHY (temperature/interval distance).
    assert result.feature_scores.get("temperature_c", 0.0) > 0.0
    assert result.feature_scores.get("session_elapsed_sec", 0.0) > 0.0


def test_contraindicated_envelope_rejects() -> None:
    """v4 §13: run1 death-spiral envelope vetoes even a perfect regime match."""
    contraindicated = ApplicabilityEnvelope(
        memory_id="mem_slow_down",
        body_ids=["rh56_right_01"],
        task_ids=["rh56_rps"],
        regime_labels=[RegimeLabel.THERMAL_TRACKING_DEGRADATION.value],
        envelope_type=EnvelopeType.CONTRAINDICATED.value,
        reason="breaks_reveal_timing",
        evidence_refs=["run1_patch_proofs"],
    )
    matcher = RegimeMatcher()
    result = matcher.match("mem_slow_down", [_thermal_envelope(), contraindicated], _regime())
    assert result.applicable is False
    assert result.score == 0.0
    assert "contraindicated_envelope_hit" in result.hard_rejections


def test_required_feature_missing_rejects() -> None:
    """v4 §13: a required feature the regime lacks cannot match."""
    matcher = RegimeMatcher()
    regime = _regime(temperature_c=None)  # temperature unknown
    result = matcher.match("mem_slow_down", [_thermal_envelope()], regime)
    assert result.applicable is False
    assert "missing_required_features" in result.hard_rejections
    assert "temperature_c" in result.missing_required_features


def test_wrong_body_rejects() -> None:
    matcher = RegimeMatcher()
    result = matcher.match(
        "mem_slow_down",
        [_thermal_envelope(body_ids=["rh56_left_01"])],
        _regime(),
    )
    assert result.applicable is False
    assert "body_mismatch" in result.hard_rejections


def test_wrong_joint_rejects() -> None:
    matcher = RegimeMatcher()
    result = matcher.match(
        "mem_slow_down",
        [_thermal_envelope(joints=["thumb_rot"])],
        _regime(),
    )
    assert result.applicable is False
    assert "joint_mismatch" in result.hard_rejections


def test_control_profile_mismatch_rejects() -> None:
    matcher = RegimeMatcher()
    result = matcher.match(
        "mem_slow_down",
        [_thermal_envelope(control_profile_hashes=["ctl_other"])],
        _regime(),
    )
    assert result.applicable is False
    assert "control_profile_mismatch" in result.hard_rejections


def test_no_envelope_not_applicable() -> None:
    matcher = RegimeMatcher()
    result = matcher.match("mem_unknown", [], _regime())
    assert result.applicable is False
    assert "no_applicability_envelope" in result.hard_rejections


def test_regime_score_explanation() -> None:
    """v4 §13: the score decomposes into inspectable parts."""
    matcher = RegimeMatcher()
    result = matcher.match("mem_slow_down", [_thermal_envelope()], _regime())
    explanation = result.explanation
    assert "regime_similarity" in explanation
    assert "evidence_factor" in explanation
    assert "envelope_type_factor" in explanation
    assert 0.0 <= explanation["regime_similarity"] <= 1.0
    assert explanation["regime_label"] == RegimeLabel.THERMAL_TRACKING_DEGRADATION.value


def test_validated_beats_observed() -> None:
    matcher = RegimeMatcher()
    observed = _thermal_envelope(envelope_type=EnvelopeType.OBSERVED.value, envelope_id="env_obs")
    validated = _thermal_envelope(envelope_id="env_val")
    result = matcher.match("mem_slow_down", [observed, validated], _regime())
    assert result.matched_envelope_id == "env_val"


def test_envelope_from_regime_roundtrip() -> None:
    regime = _regime()
    envelope = envelope_from_regime("mem_x", regime, envelope_type=EnvelopeType.OBSERVED.value)
    assert envelope.body_ids == ["rh56_right_01"]
    assert envelope.temperature_min == 57.0
    assert envelope.regime_labels == [RegimeLabel.THERMAL_TRACKING_DEGRADATION.value]
    matcher = RegimeMatcher()
    result = matcher.match("mem_x", [envelope], _regime())
    # OBSERVED envelope at the exact regime it was seen in: no distance.
    assert result.explanation["regime_similarity"] == 1.0


def test_matcher_config_from_yaml() -> None:
    import yaml

    with open("configs/regimes/rh56_rps_v1.yaml", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = MatcherConfig.from_dict(raw.get("regime_matcher", {}))
    assert config.abstain_below == 0.70
    assert config.suggest_below == 0.85
    assert config.feature_weights["temperature_c"] == 1.0


# ---------------------------------------------------------------------------
# ApplicabilityStore
# ---------------------------------------------------------------------------


def test_applicability_store_roundtrip_and_outcomes() -> None:
    client = InMemoryKnowledgeStore()
    client.connect()
    store = ApplicabilityStore(client)
    envelope = _thermal_envelope()
    store.upsert(envelope)

    loaded = store.for_memory("mem_slow_down")
    assert len(loaded) == 1
    assert loaded[0].envelope_type == EnvelopeType.VALIDATED.value
    assert loaded[0].temperature_min == 55.0

    updated = store.record_outcome(envelope.envelope_id, success=True, evidence_ref="evt_9")
    assert updated is not None
    assert updated.evidence_count == 5
    assert updated.success_count == 5
    assert "evt_9" in updated.evidence_refs
    assert updated.confidence > 0.5

    updated = store.record_outcome(envelope.envelope_id, success=False)
    assert updated is not None
    assert updated.failure_count == 1
    reloaded = store.for_memory("mem_slow_down")[0]
    assert reloaded.evidence_count == 6

    assert store.delete(envelope.envelope_id) is True
    assert store.for_memory("mem_slow_down") == []


def test_envelope_record_json_roundtrip() -> None:
    envelope = _thermal_envelope()
    record = envelope.to_record()
    loaded = ApplicabilityEnvelope.from_record(record)
    assert loaded.memory_id == envelope.memory_id
    assert loaded.joints == envelope.joints
    assert loaded.regime_labels == envelope.regime_labels
    assert loaded.required_features == envelope.required_features
    assert loaded.temperature_max == envelope.temperature_max


# ---------------------------------------------------------------------------
# Review fixes (PR-MEM-6 follow-up): identity semantics
# ---------------------------------------------------------------------------


def test_unverified_identity_not_explicit_mismatch_but_disclosed() -> None:
    """Regime with NO calibration context + calibration-constrained envelope:
    not a hard rejection (no explicit mismatch), but unverified_identity
    must name the dimension so the APPLY rung can refuse it downstream."""
    matcher = RegimeMatcher()
    envelope = _thermal_envelope(calibration_hashes=["cal_old"])
    regime = _regime(calibration_hash=None)
    result = matcher.match("mem_slow_down", [envelope], regime)
    assert "calibration_profile_mismatch" not in result.hard_rejections
    assert "calibration_hash" in result.unverified_identity
    assert result.explanation.get("unverified_identity") is None  # field, not dict
    assert result.to_dict()["unverified_identity"] == ["calibration_hash"]


def test_wrong_calibration_still_hard_rejects() -> None:
    matcher = RegimeMatcher()
    result = matcher.match(
        "mem_slow_down",
        [_thermal_envelope(calibration_hashes=["cal_old"])],
        _regime(calibration_hash="cal_NEW"),
    )
    assert "calibration_profile_mismatch" in result.hard_rejections
    assert result.applicable is False


def test_required_features_work_for_string_identities() -> None:
    """required_features must handle string identity fields (feature_value
    is float-only; the escape hatch previously could never pass)."""
    matcher = RegimeMatcher()
    envelope = _thermal_envelope(required_features=["calibration_hash"])
    ok = matcher.match("mem_slow_down", [envelope], _regime(calibration_hash="cal_abc"))
    assert "missing_required_features" not in ok.hard_rejections
    missing = matcher.match("mem_slow_down", [envelope], _regime(calibration_hash=None))
    assert "missing_required_features" in missing.hard_rejections
    assert "calibration_hash" in missing.missing_required_features


def test_contraindicated_veto_scoped_to_envelope_ranges() -> None:
    """A hot-zone contraindication vetoes at 57 °C but NOT at 45 °C."""
    contra = ApplicabilityEnvelope(
        memory_id="mem_slow_down",
        body_ids=["rh56_right_01"],
        task_ids=["rh56_rps"],
        temperature_min=56.0,
        temperature_max=60.0,
        envelope_type=EnvelopeType.CONTRAINDICATED.value,
        reason="breaks_reveal_timing",
    )
    matcher = RegimeMatcher()
    hot = matcher.match("mem_slow_down", [_thermal_envelope(), contra], _regime())
    assert "contraindicated_envelope_hit" in hot.hard_rejections
    cold = matcher.match(
        "mem_slow_down",
        [_thermal_envelope(), contra],
        _regime(temperature_c=45.0, regime_label=RegimeLabel.COLD_HEALTHY.value),
    )
    assert "contraindicated_envelope_hit" not in cold.hard_rejections
    # ...but the cold regime is also outside the VALIDATED envelope, so the
    # memory is merely not-applicable (score-based), not vetoed.
    assert cold.applicable is False


def test_contraindicated_veto_with_unset_ranges_applies_globally() -> None:
    """A range-free contraindication (identity-scoped only) vetoes anywhere —
    that IS its documented meaning."""
    contra = ApplicabilityEnvelope(
        memory_id="mem_slow_down",
        body_ids=["rh56_right_01"],
        envelope_type=EnvelopeType.CONTRAINDICATED.value,
        reason="globally_unsafe_on_this_body",
    )
    result = RegimeMatcher().match(
        "mem_slow_down",
        [_thermal_envelope(), contra],
        _regime(temperature_c=45.0),
    )
    assert "contraindicated_envelope_hit" in result.hard_rejections
