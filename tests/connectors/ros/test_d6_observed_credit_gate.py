"""Nonvacuous D6 source calculation; synthetic samples are not physical evidence."""

import importlib
from pathlib import Path

import pytest

from tests.connectors.ros.test_qualified_backend_episode import protocol
from tests.connectors.ros.test_temporal_mission_runtime import evidence, sample


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("qualified_backend_episode"), importlib.import_module(
        "dynamic_native_episode"
    )


def d6_spec():
    return {
        **protocol(),
        "schema_version": "rosclaw.dynamic_native_episode.v5",
        "case": "D6",
        "scenario_source": {
            "exposure_policy": "PENDING_BLOCKED_ENABLED_BRUSH_AND_SAME_CELL_FREE_REVISIT"
        },
    }


def test_d6_keeps_original_body_budget_model_geometry_and_qualified_wire(modules):
    qualified, episode = modules
    spec = d6_spec()
    base, wire = qualified.validate_qualified_spec(spec, episode.validate_episode_spec)
    assert base["case"] == "D6" and base["scenario_source"] == spec["scenario_source"]
    for key in ("profile", "mission_timeout_sec", "model", "plugin_sha256", "p0_merge_commit"):
        assert base[key] == spec[key]
    assert wire == spec["backend_source"]
    with pytest.raises(ValueError):
        episode.validate_episode_spec(spec)


@pytest.mark.parametrize("fault", ["missing_policy", "downgrade", "skip_zero", "wrong_case"])
def test_no_d6_policy_downgrade_or_self_asserted_pass(modules, fault):
    qualified, episode = modules
    spec = d6_spec()
    if fault == "missing_policy":
        spec.pop("scenario_source")
    elif fault == "downgrade":
        spec["scenario_source"]["exposure_policy"] = "ANY_OBSTACLE_PRESENT"
    elif fault == "skip_zero":
        spec["scenario_source"]["passed"] = True
    else:
        spec["case"] = "D2"
    with pytest.raises(ValueError, match="closed qualified D6"):
        qualified.validate_qualified_spec(spec, episode.validate_episode_spec)


@pytest.mark.parametrize(
    "kind", ["no_overlap", "brush_off", "already_clean", "no_revisit", "other_cell_revisit"]
)
def test_vacuous_exposure_or_unrelated_revisit_refuses_d6(modules, kind):
    qualified, _ = modules
    if kind == "no_overlap":
        rows = [sample(0, 0, 0.005, []), sample(0.1, 1, 0.015, [])]
    elif kind == "already_clean":
        rows = [sample(0, 0, 0.005, []), sample(0.1, 1, 0.005, [0]), sample(0.2, 2, 0.005, [])]
    else:
        rows = [sample(0, 0, 0.005, [0]), sample(0.1, 1, 0.015, [])]
        if kind == "brush_off":
            rows[0]["cleaning_enabled"] = False
            rows.append(sample(0.2, 2, 0.005, []))
        if kind == "other_cell_revisit":
            rows.append(sample(0.2, 2, 0.015, []))
    with pytest.raises(ValueError, match="D6 requires actual pending"):
        qualified.require_d6_observed_credit(evidence(rows))


def test_actual_enabled_same_cell_free_revisit_is_required_without_promoting_physics(modules):
    qualified, _ = modules
    result = qualified.require_d6_observed_credit(
        evidence([sample(0, 0, 0.005, [0]), sample(0.1, 1, 0.015, []), sample(0.2, 2, 0.005, [])])
    )
    assert result["previously_unclean_blocked_enabled_brush_exposure_cells"] == [0]
    assert result["previously_unclean_exposed_cells_actually_revisited_free"] == [0]
    assert result["false_new_credit_while_occupied_cells"] == []
    assert result["physical_acceptance"] == "NOT_VERIFIED"


def test_mutated_occupancy_hash_cannot_enter_d6_calculation(modules):
    qualified, _ = modules
    source = evidence([sample(0, 0, 0.005, [0]), sample(0.1, 1, 0.005, [])])
    source["occupancy_samples"][0]["occupancy_hash"] = "forged"
    with pytest.raises(ValueError):
        qualified.require_d6_observed_credit(source)
