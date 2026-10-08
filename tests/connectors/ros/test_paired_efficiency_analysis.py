"""Every preregistered paired seed contributes to statistical acceptance."""

import copy
import importlib.util
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
        "profile": "waffle",
        "phase": "evaluation",
        "seed": seed,
        "source_commit": "source",
        "image_id": "image",
        "candidate": "perimeter",
    }
    p["runs"] = [
        {k: p[k] for k in ["profile", "seed", "source_commit", "image_id"]}
        | {
            "arm": arm,
            "status": "PASS",
            "audit_complete": True,
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
