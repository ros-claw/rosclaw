"""Source-only regressions using genuine quality API objects; no execution at import."""


def _stats(key, episode=17, valid=4):
    from rosclaw.integrations.lerobot.sync_stats import SyncFeatureStats

    # Four exact target samples, optionally one invalid: coverage is valid/4.
    return SyncFeatureStats(
        episode_index=episode,
        feature_key=key,
        source_clock_domain="episode_time",
        target_clock_domain="episode_time",
        clock_mapping_model="identity",
        sync_method="nearest",
        source_samples=4,
        target_frames=4,
        valid_frames=valid,
        missing_frames=4 - valid,
        dropped_frames=0,
        exact_frames=valid,
        nearest_frames=0,
        interpolated_frames=0,
        held_frames=0,
        aggregated_frames=0,
        unknown_frames=4 - valid,
        coverage_ratio=valid / 4,
        source_rate_hz=4.0,
        effective_output_rate_hz=float(valid),
        mean_skew_ms=0.0,
        p50_skew_ms=0.0,
        p95_skew_ms=0.0,
        max_skew_ms=0.0,
        mean_hold_age_ms=None,
        p95_hold_age_ms=None,
        max_hold_age_ms=None,
        mean_samples_per_window=None,
        min_samples_per_window=None,
        max_samples_per_window=None,
    )


def _mapping(unmapped=False):
    from rosclaw.integrations.lerobot.clock_mapping import ClockMappingResult

    return ClockMappingResult("episode_time", {}, {"device_time"} if unmapped else set(), True)


def test_default_profiles_reject_each_whole_required_omission():
    from rosclaw.integrations.lerobot.sync_quality import default_quality_config, evaluate_quality

    for profile in ("strict", "balanced", "permissive"):
        for missing, present in (("action", "observation.state"), ("observation.state", "action")):
            result = evaluate_quality(
                [_stats(present)], _mapping(), default_quality_config(profile)
            )
            assert not result.passed
            assert result.action == "fail"
            assert any(missing in error for error in result.errors)
            failed = [r for r in result.feature_results if not r.passed]
            assert [r.feature_key for r in failed] == [missing]
            assert failed[0].errors
            serialized = result.to_dict()
            assert serialized["passed"] is False
            assert serialized["errors"] == result.errors


def test_empty_table_rejects_without_inventing_episode():
    from rosclaw.integrations.lerobot.sync_quality import default_quality_config, evaluate_quality

    for profile in ("strict", "balanced", "permissive"):
        result = evaluate_quality([], _mapping(), default_quality_config(profile))
        assert not result.passed and result.action == "fail"
        assert {r.feature_key for r in result.feature_results if not r.passed} == {
            "observation.state",
            "action",
        }
        assert len(result.errors) == 2
        assert all("empty" in error and "episode unknown" in error for error in result.errors)
        assert all("episode_index=" not in error for error in result.errors)
        assert result.clock_result["passed"] is True


def test_complete_single_episode_recovers_with_optional_features_absent():
    from rosclaw.integrations.lerobot.sync_quality import default_quality_config, evaluate_quality

    for profile in ("strict", "balanced", "permissive"):
        result = evaluate_quality(
            [_stats("observation.state"), _stats("action")],
            _mapping(),
            default_quality_config(profile),
        )
        assert result.passed and result.action == "ignore"
        assert not result.errors and not result.warnings
        assert len(result.feature_results) == 2


def test_crossed_episode_omissions_cannot_cover_each_other():
    from rosclaw.integrations.lerobot.sync_quality import default_quality_config, evaluate_quality

    rows = [_stats("observation.state", 17), _stats("action", 29)]
    result = evaluate_quality(rows, _mapping(), default_quality_config())
    assert not result.passed
    assert any("action" in error and "episode_index=17" in error for error in result.errors)
    assert any(
        "observation.state" in error and "episode_index=29" in error for error in result.errors
    )
    assert len(result.errors) == 2
    rows += [_stats("action", 17), _stats("observation.state", 29)]
    recovered = evaluate_quality(rows, _mapping(), default_quality_config())
    assert recovered.passed and not recovered.errors


def test_optional_only_episode_still_requires_core_evidence():
    from rosclaw.integrations.lerobot.sync_quality import default_quality_config, evaluate_quality

    rows = [
        _stats("observation.state", 17),
        _stats("action", 17),
        _stats("observation.images.front", 29),
    ]
    result = evaluate_quality(rows, _mapping(), default_quality_config())
    assert not result.passed
    assert len(result.errors) == 2
    assert all("episode_index=29" in error for error in result.errors)


def test_custom_and_empty_rules_are_authoritative_not_profile_labels():
    from rosclaw.integrations.lerobot.sync_quality import QualityConfig, evaluate_quality

    # A profile label alone does not expand an explicit rule map.
    for config in (
        QualityConfig(),
        QualityConfig.from_dict({"profile": "strict", "rules": {}}),
        QualityConfig.from_dict(
            {"rules": {"observation.images.front": {"minimum_coverage": 1.0, "action": "fail"}}}
        ),
    ):
        result = evaluate_quality([], _mapping(), config)
        assert result.passed and result.action == "ignore"
        assert not result.feature_results and not result.errors


def test_custom_required_rules_honor_fail_warn_ignore_even_without_thresholds():
    from rosclaw.integrations.lerobot.sync_quality import QualityConfig, evaluate_quality

    for key in ("observation.state", "action"):
        for action in ("fail", "warn", "ignore"):
            config = QualityConfig.from_dict(
                {"profile": "custom", "rules": {key: {"action": action}}}
            )
            result = evaluate_quality([], _mapping(), config)
            assert result.passed is (action != "fail")
            assert result.action == action
            assert bool(result.errors) is (action == "fail")
            assert bool(result.warnings) is (action == "warn")
            assert all(key in message for message in result.errors + result.warnings)


def test_present_coverage_and_optional_thresholds_remain_effective():
    from rosclaw.integrations.lerobot.sync_quality import default_quality_config, evaluate_quality

    for key in ("observation.state", "action", "observation.images.front"):
        rows = [_stats("observation.state"), _stats("action")]
        rows = [r for r in rows if r.feature_key != key] + [_stats(key, valid=3)]
        result = evaluate_quality(rows, _mapping(), default_quality_config("balanced"))
        assert not result.passed and result.action == "fail"
        assert any(key in error and "coverage" in error for error in result.errors)
    rows = [_stats("observation.state"), _stats("action"), _stats("observation.contact", valid=3)]
    result = evaluate_quality(rows, _mapping(), default_quality_config("balanced"))
    assert result.passed and result.action == "warn"
    assert any("observation.contact" in warning for warning in result.warnings)


def test_clocks_remain_independent_and_fail_overrides_warn():
    from rosclaw.integrations.lerobot.sync_quality import QualityConfig, evaluate_quality

    for action in ("fail", "warn", "ignore"):
        config = QualityConfig.from_dict({"clocks": {"unmapped_mixed_domains_action": action}})
        result = evaluate_quality([], _mapping(True), config)
        assert result.passed is (action != "fail")
        assert result.action == action
        assert result.clock_result == {
            "passed": action != "fail",
            "action": action,
            "unmapped_clocks": ["device_time"],
        }
    config = QualityConfig.from_dict({"rules": {"action": {"action": "warn"}}})
    result = evaluate_quality([], _mapping(True), config)
    assert not result.passed and result.action == "fail"
    assert result.errors and result.warnings


def test_evaluation_does_not_mutate_inputs():
    from copy import deepcopy

    from rosclaw.integrations.lerobot.sync_quality import default_quality_config, evaluate_quality

    rows = [_stats("observation.state", 43)]
    config = default_quality_config()
    mapping = _mapping(True)
    before = deepcopy((rows, config, mapping))
    result = evaluate_quality(rows, mapping, config)
    assert not result.passed
    assert (rows, config, mapping) == before
