"""Native cannot accept a foreign, failed or uncompleted fixture perturbation."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance"


def load(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location("native_required_scenario", ROOT / "native.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(root, case="D2", kind="SCENARIO_STARTED"):
    scenario = json.dumps(
        {
            "schema_version": "rosclaw.dynamic_fixture_scenario.v1",
            "case": case,
            "run_id": "fresh-run",
            "mission_id": "fresh-mission",
        }
    ).encode()
    config = {
        "occupancy_binding": {"run_id": "fresh-run"},
        "dynamic_fixture_admission": {"mission_id": "fresh-mission"},
    }
    (root / "execution_config.json").write_text(json.dumps(config))
    event = {
        "kind": kind,
        "run_id": "fresh-run",
        "mission_id": "fresh-mission",
        "scenario_sha256": hashlib.sha256(scenario).hexdigest(),
        "physical_acceptance": "NOT_VERIFIED",
    }
    (root / "dynamic-scenario-events.jsonl").write_text(json.dumps(event) + "\n")
    return scenario, config, event


@pytest.mark.parametrize(
    "case,kind,complete",
    [
        ("D2", "SCENARIO_STARTED", False),
        ("D4", "ACTUAL_POSTUPDATE_POSITION_CONFIRMED", False),
        ("D2", "PERTURBATION_COMPLETE_REQUIRES_NATIVE_AND_CREDIT_VALIDATION", True),
    ],
)
def test_fixture_progress_never_promotes_physical_acceptance(
    tmp_path, monkeypatch, case, kind, complete
):
    scenario, _, _ = fixture(tmp_path, case, kind)
    assert load(monkeypatch).required_scenario_progress(tmp_path, scenario) == {
        "case": case,
        "perturbation_complete": complete,
        "physical_acceptance": "NOT_VERIFIED",
    }


@pytest.mark.parametrize("kind", ["SCENARIO_FAILED", "TASK_RUNNER_STOP_REQUESTED"])
def test_failed_or_stopped_required_controller_refuses_native_completion(
    tmp_path, monkeypatch, kind
):
    scenario, _, _ = fixture(tmp_path, kind=kind)
    with pytest.raises(RuntimeError, match="failed or stopped"):
        load(monkeypatch).required_scenario_progress(tmp_path, scenario)


@pytest.mark.parametrize(
    "field,value",
    [
        ("scenario_sha256", "foreign"),
        ("run_id", "old-run"),
        ("mission_id", "old-mission"),
        ("physical_acceptance", "PASS"),
    ],
)
def test_event_cannot_substitute_identity_or_assert_acceptance(tmp_path, monkeypatch, field, value):
    scenario, _, event = fixture(tmp_path)
    event[field] = value
    (tmp_path / "dynamic-scenario-events.jsonl").write_text(json.dumps(event) + "\n")
    with pytest.raises(ValueError, match="identity"):
        load(monkeypatch).required_scenario_progress(tmp_path, scenario)


@pytest.mark.parametrize(
    "config",
    [
        None,
        [],
        {"occupancy_binding": None},
        {"occupancy_binding": {}, "dynamic_fixture_admission": []},
        {
            "occupancy_binding": {"run_id": "foreign"},
            "dynamic_fixture_admission": {"mission_id": "fresh-mission"},
        },
    ],
)
def test_malformed_or_foreign_source_admission_is_refused(tmp_path, monkeypatch, config):
    scenario, _, _ = fixture(tmp_path)
    (tmp_path / "execution_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load(monkeypatch).required_scenario_progress(tmp_path, scenario)


@pytest.mark.parametrize("scenario", [b"[]", b"null", b"", b"x" * 65537])
def test_malformed_or_unbounded_scenario_is_refused(tmp_path, monkeypatch, scenario):
    fixture(tmp_path)
    with pytest.raises(ValueError):
        load(monkeypatch).required_scenario_progress(tmp_path, scenario)


def test_partial_next_append_cannot_hide_completed_failure(tmp_path, monkeypatch):
    scenario, _, _ = fixture(tmp_path, kind="SCENARIO_FAILED")
    with (tmp_path / "dynamic-scenario-events.jsonl").open("ab") as stream:
        stream.write(b'{"kind":"PERTURBATION_COMPLETE')
    with pytest.raises(RuntimeError, match="failed or stopped"):
        load(monkeypatch).required_scenario_progress(tmp_path, scenario)
