"""Unactivated source launch plans, not a new Body or World acceptance."""

import importlib
import json
from pathlib import Path

import pytest

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from tests.connectors.ros.test_generic_stack_source import synthetic_stack_inputs


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("generic_navigation_launch")


def test_plan_retains_all_declared_names_remaps_and_never_autoactivates(module, tmp_path):
    inputs = synthetic_stack_inputs()
    directory = tmp_path / "prepared"
    prepare = importlib.import_module("generic_stack_source").prepare_generic_stack_source
    prepare(directory, **inputs)
    plan = module.navigation_launch_plan(directory)
    assert set(plan["lifecycle_node_names"]) == set(
        inputs["navigation_declaration"]["nodes"].values()
    )
    assert len(plan["nodes"]) == 10
    assert (
        plan["lifecycle_namespace"]
        == next(iter(inputs["navigation_declaration"]["nodes"].values())).rsplit("/", 1)[0]
    )
    assert plan["lifecycle_autostart"] is False
    assert plan["live_admission"] is False and plan["authorization"] is False
    assert plan["physical_acceptance"] == "NOT_RUN"


@pytest.mark.parametrize(
    "field,value", [("package", "unreviewed"), ("name", "other"), ("role", "other")]
)
def test_resealed_inventory_cannot_change_navigation_report_identity(
    module, tmp_path, field, value
):
    directory = tmp_path / "prepared"
    importlib.import_module("generic_stack_source").prepare_generic_stack_source(
        directory, **synthetic_stack_inputs()
    )
    source = directory / "navigation-launch-source.json"
    rows = json.loads(source.read_bytes())
    rows[0][field] = value
    source.write_text(json.dumps(rows))
    # Rehash the untrusted manifest, retaining the independently bound report.
    import hashlib

    path = directory / "generic-stack-source-manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["output_hashes"][source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest["artifact_hash"] = digest({k: v for k, v in manifest.items() if k != "artifact_hash"})
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="navigation source report"):
        module.navigation_launch_plan(directory)


@pytest.mark.parametrize("namespace", ["/probe_one", "/probe_two/nested"])
def test_namespace_changes_derive_all_lifecycle_targets_from_source(module, tmp_path, namespace):
    inputs = synthetic_stack_inputs()
    declaration = inputs["navigation_declaration"]
    declaration["nodes"] = {
        role: namespace + "/" + path.rsplit("/", 1)[1]
        for role, path in declaration["nodes"].items()
    }
    directory = tmp_path / "prepared"
    importlib.import_module("generic_stack_source").prepare_generic_stack_source(
        directory, **inputs
    )
    plan = module.navigation_launch_plan(directory)
    assert plan["lifecycle_namespace"] == namespace
    assert set(plan["lifecycle_node_names"]) == set(declaration["nodes"].values())
    assert plan["lifecycle_autostart"] is False
