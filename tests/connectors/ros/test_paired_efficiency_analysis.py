"""Every preregistered paired seed contributes to statistical acceptance."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "paired_analysis",
    Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance/paired_analysis.py",
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def pair(seed, baseline, candidate):
    p = {
        "status": "PASS",
        "profile": "waffle",
        "phase": "evaluation",
        "seed": seed,
        "source_commit": "source",
        "image_id": "image",
        "candidate": "perimeter",
        "mission_timeout_sec": 900,
        "protocol_sha256": "protocol",
    }
    p["runs"] = [
        {k: p[k] for k in ["profile", "seed", "source_commit", "image_id"]}
        | {
            "arm": arm,
            "preset": "baseline" if arm == "baseline" else p["candidate"],
            "status": "PASS",
            "audit_complete": True,
            "complete_independent_observations": True,
            "mission_timeout_sec": p["mission_timeout_sec"],
            "run_id": f"{seed}-{arm}",
            "coverage_ratio": 0.98,
            "collision_count": 0,
            "trace_gaps": 0,
            "post_cleanup_displacement_m": 0.001,
            "post_cleanup_yaw_change_rad": 0.01,
            "measured_distance_m": value,
            "sim_duration_sec": value * 10,
        }
        for arm, value in [("baseline", baseline), ("candidate", candidate)]
    ]
    return p


@pytest.mark.parametrize("status", ["FAIL", "NO_FINAL_RESULT", None])
def test_whole_pair_failure_cannot_be_promoted_by_two_passing_arms(status):
    value = pair(1, 10, 1)
    value["status"] = status
    result = analysis.analyze_pairs(
        [value], expected_seeds=[1], profile="waffle", phase="evaluation"
    )
    assert result["pair_success_rate"] == 0
    assert not result["complete_frozen_series"]
    assert result["metrics"] == {}
    assert "whole paired run did not pass" in result["failed_pairs"][0]["reasons"]


def test_cli_requires_exact_supplied_preregistration_bytes(tmp_path, monkeypatch):
    original = b'{"pilot_seeds": [1]}'
    different = b'{"pilot_seeds": [1]}\n'
    value = pair(1, 10, 1)
    value["phase"] = "pilot"
    value["protocol_sha256"] = hashlib.sha256(original).hexdigest()
    source = tmp_path / "pair.json"
    protocol = tmp_path / "protocol.json"
    output = tmp_path / "analysis.json"
    source.write_text(json.dumps(value))
    protocol.write_bytes(different)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analysis",
            "--pairs",
            str(source),
            "--protocol",
            str(protocol),
            "--profile",
            "waffle",
            "--phase",
            "pilot",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="exact supplied preregistration bytes"):
        analysis.main()
    assert not output.exists()
    protocol.write_bytes(original)
    analysis.main()
    assert json.loads(output.read_text())["complete_frozen_series"]


def test_paired_bootstrap_is_reproducible_and_medians_are_not_conflated():
    pairs = [pair(1, 10, 1), pair(2, 20, 12), pair(3, 100, 30)]
    before = copy.deepcopy(pairs)
    kwargs = {
        "expected_seeds": [1, 2, 3],
        "profile": "waffle",
        "phase": "evaluation",
        "bootstrap_samples": 200,
    }
    result = analysis.analyze_pairs(pairs, **kwargs)
    assert result == analysis.analyze_pairs(pairs, **kwargs)
    assert pairs == before
    metric = result["metrics"]["measured_distance_m"]
    assert metric["reduction_of_medians"] == pytest.approx(0.4)
    assert metric["median_paired_reduction"] == pytest.approx(0.7)
    assert result["complete_frozen_series"] is True
    assert result["median_30_percent_target"] is True
    assert result["v1_done"] is False


def test_failed_or_missing_pairs_never_become_successful_subset_improvement():
    pairs = [pair(1, 10, 1), pair(2, 20, 2)]
    kwargs = {"expected_seeds": [1, 2, 3], "profile": "waffle", "phase": "evaluation"}
    result = analysis.analyze_pairs(pairs, **kwargs)
    assert result["missing_seeds"] == [3]
    assert result["metrics"] == {}
    pairs[1]["runs"][1]["collision_count"] = 1
    result = analysis.analyze_pairs(pairs, **kwargs)
    assert result["pair_success_rate"] == 1 / 3
    assert result["failed_pairs"][0]["seed"] == 2
    assert result["median_30_percent_target"] is False
    with pytest.raises(ValueError, match="duplicate"):
        analysis.analyze_pairs([pairs[0], pairs[0]], **kwargs)


@pytest.mark.parametrize("mutation", ["source", "image", "run_id", "stop", "nan"])
def test_changed_freeze_or_invalid_measurements_fail_series(mutation):
    pairs = [pair(1, 10, 1), pair(2, 20, 2)]
    if mutation in ("source", "image"):
        key = "source_commit" if mutation == "source" else "image_id"
        pairs[1][key] = "changed"
        for r in pairs[1]["runs"]:
            r[key] = "changed"
    elif mutation == "run_id":
        pairs[1]["runs"][0]["run_id"] = pairs[0]["runs"][0]["run_id"]
    elif mutation == "stop":
        pairs[1]["runs"][0]["post_cleanup_displacement_m"] = 0.011
    else:
        pairs[1]["runs"][0]["measured_distance_m"] = float("nan")
    result = analysis.analyze_pairs(
        pairs, expected_seeds=[1, 2], profile="waffle", phase="evaluation"
    )
    assert result["failed_pairs"]
    assert result["metrics"] == {}
    assert result["complete_frozen_series"] is False


@pytest.mark.parametrize(
    "fault",
    ["arm_preset", "baseline_repair", "candidate_repair", "mixed_repair", "incomplete_observer"],
)
def test_repair_ablation_identity_and_complete_observations_are_required(fault):
    pairs = [pair(1, 10, 1), pair(2, 20, 2)]
    row = pairs[1]["runs"][1]
    if fault == "arm_preset":
        row["preset"] = "baseline"
    elif fault == "baseline_repair":
        pairs[1]["runs"][0]["repair_strategy"] = "pose_aware"
    elif fault == "candidate_repair":
        row["repair_strategy"] = "pose_aware"
    elif fault == "mixed_repair":
        pairs[1]["candidate_repair_strategy"] = row["repair_strategy"] = "pose_aware"
    else:
        row["complete_independent_observations"] = False
    result = analysis.analyze_pairs(
        pairs, expected_seeds=[1, 2], profile="waffle", phase="evaluation"
    )
    assert result["failed_pairs"] and not result["complete_frozen_series"]
    assert result["metrics"] == {} and not result["median_30_percent_target"]


@pytest.mark.parametrize(
    "fault", ["arm_deadline", "series_deadline", "protocol", "missing_deadline"]
)
def test_extended_deadlines_or_changed_protocol_cannot_form_a_frozen_comparison(fault):
    pairs = [pair(1, 10, 1), pair(2, 20, 2)]
    second = pairs[1]
    if fault == "arm_deadline":
        second["runs"][1]["mission_timeout_sec"] = 1800
    elif fault == "series_deadline":
        second["mission_timeout_sec"] = 1800
        for row in second["runs"]:
            row["mission_timeout_sec"] = 1800
    elif fault == "protocol":
        second["protocol_sha256"] = "changed"
    else:
        del second["mission_timeout_sec"]
    result = analysis.analyze_pairs(
        pairs, expected_seeds=[1, 2], profile="waffle", phase="evaluation"
    )
    assert result["failed_pairs"] and not result["complete_frozen_series"]
    assert result["metrics"] == {} and not result["median_30_percent_target"]
