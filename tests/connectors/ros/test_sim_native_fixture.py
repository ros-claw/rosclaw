"""Generic Native declaration/reopen contracts; no held-out asset or action."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from rosclaw.connectors.ros.context.sim_execution_config import prepare_sim_execution_config
from rosclaw.connectors.ros.context.sim_native_fixture import prepare_sim_native_fixture
from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from tests.connectors.ros.test_sim_execution_config import sources  # noqa: F401

ROOT = Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance"


def load(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location("generic_" + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixture(request, tmp_path):
    workspace, model, data, kwargs = request.getfixturevalue("sources")
    root = tmp_path / "fixture"
    root.mkdir()
    workspace.rename(root / "home")
    (root / "robot.urdf").write_bytes(data)
    admission = prepare_sim_execution_config(root / "home", model, data, **kwargs)
    return root, admission


def test_generic_native_declarations_use_actual_compiled_frame_body_and_whole_region(fixture):
    root, admission = fixture
    before = deepcopy(admission)
    result = prepare_sim_native_fixture(root, admission)
    assert admission == before
    assert result["physical_acceptance"] == "NOT_RUN" and not result["actions_dispatched"]
    assert result["body"]["map_frame"] == "world"
    assert result["body"]["coverage_polygon"] == admission["configuration"]["mission_polygon"]
    assert len(result["body"]["coverage_polygon"]) == 6
    assert (
        result["execution_config"]["endpoints"]["navigate_complete_coverage"]
        == "/different/robot/cover"
    )
    assert result["execution_config"]["generic_execution_proposal"]["mission_id"] == "mission"
    assert not (root / "body.json").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "root_urdf",
        "compiled_urdf",
        "source",
        "radius",
        "endpoint",
        "frame",
        "brush",
        "region",
        "real",
        "executed",
        "mission",
    ],
)
def test_generic_reopen_refuses_source_or_declared_execution_substitution(fixture, fault):
    root, admission = fixture
    if fault == "hash":
        admission["artifact_hash"] = "different"
    elif fault == "root_urdf":
        (root / "robot.urdf").write_bytes(b"foreign")
    elif fault == "compiled_urdf":
        from rosclaw.body.resolver import BodyResolver

        BodyResolver(workspace=root / "home").eurdf_profile_path.with_name(
            "robot.urdf"
        ).write_bytes(b"foreign")
    else:
        if fault == "source":
            admission["source_snapshot_hash"] = "old"
        elif fault == "radius":
            admission["configuration"]["physical_radius_m"] = 0.01
        elif fault == "endpoint":
            admission["configuration"]["endpoints"]["set_cleaning"] = "/foreign"
        elif fault == "frame":
            admission["configuration"]["grid"]["frame_id"] = "foreign"
        elif fault == "brush":
            admission["configuration"]["grid"]["cleaning_polygon"] = [[0, 0], [1, 0], [0, 1]]
        elif fault == "region":
            admission["region_preflight"]["grid"] = {}
        elif fault == "real":
            admission["usable_for_real_execution"] = True
        elif fault == "executed":
            admission["actions_dispatched"] = True
        elif fault == "mission":
            admission["mission_id"] = ""
        admission.pop("artifact_hash")
        admission["artifact_hash"] = digest(admission)
    with pytest.raises(ValueError):
        prepare_sim_native_fixture(root, admission)
    assert not (root / "body.json").exists()


def test_generic_cli_configuration_never_looks_up_known_profiles_or_overwrites_outputs(
    fixture, monkeypatch
):
    root, admission = fixture
    path = root / "proposal.json"
    path.write_text(json.dumps(admission))
    module = load("configure_native", monkeypatch)
    monkeypatch.setattr(
        module, "profile_for_urdf", lambda *a: pytest.fail("known robot profile fallback")
    )
    result = module.configure(
        root, "ws://127.0.0.1:21990", generic_proposal=path, mission_timeout=700
    )
    assert result["supported_modes"] == ["SIMULATION"]
    config = yaml.safe_load((root / "home/config.yaml").read_bytes())
    assert config["mcp_servers"][0]["timeout_ms"] == 700000
    assert config["agent"]["body_id"] == admission["configuration"]["body_id"]
    before = (root / "execution_config.json").read_bytes()
    with pytest.raises(FileExistsError):
        module.configure(root, "ws://127.0.0.1:21990", generic_proposal=path, overwrite=True)
    assert (root / "execution_config.json").read_bytes() == before


@pytest.mark.parametrize("fault", ["missing", "typed", "mission", "run"])
def test_daemon_refuses_unadmitted_generic_fixture_before_runtime_creation(
    fixture, monkeypatch, fault
):
    root, admission = fixture
    config = prepare_sim_native_fixture(root, admission)["execution_config"]
    if fault != "missing":
        config["dynamic_fixture_admission"] = {
            "mission_id": "mission",
            "initial_packet_sha256": "synthetic",
        }
        config["occupancy_binding"] = {"run_id": "fresh_run", "geometry_hash": "synthetic"}
        if fault == "typed":
            config["dynamic_fixture_admission"] = []
        elif fault == "mission":
            config["dynamic_fixture_admission"]["mission_id"] = "old"
        elif fault == "run":
            config["occupancy_binding"]["run_id"] = "old"
    (root / "execution_config.json").write_text(json.dumps(config))
    module = load("daemon", monkeypatch)
    monkeypatch.setattr(module, "Runtime", lambda *a: pytest.fail("unadmitted Runtime creation"))
    monkeypatch.setattr("sys.argv", ["daemon.py", "--directory", str(root)])
    with pytest.raises(ValueError):
        module.main()


@pytest.mark.parametrize(
    "config",
    [
        {"generic_execution_proposal": {"mission_id": "fresh"}},
        {"dynamic_fixture_admission": {"mission_id": "fresh"}},
        {
            "generic_execution_proposal": {"mission_id": "fresh"},
            "dynamic_fixture_admission": {"mission_id": "fresh"},
        },
    ],
)
def test_catalog_mission_default_follows_generic_or_admitted_identity(monkeypatch, config):
    assert load("native_tools", monkeypatch).fixture_mission_id(config) == "fresh"


def test_catalog_refuses_generic_dynamic_mission_mismatch(monkeypatch):
    with pytest.raises(ValueError):
        load("native_tools", monkeypatch).fixture_mission_id(
            {
                "generic_execution_proposal": {"mission_id": "fresh"},
                "dynamic_fixture_admission": {"mission_id": "old"},
            }
        )
