"""Fresh offline Body compilation is not live map or Gazebo acceptance."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance"


@pytest.fixture
def bootstrap(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "dynamic_bootstrap", ROOT / "dynamic_bootstrap.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    urdf = tmp_path / "robot.urdf"
    # Synthetic identity, deliberately offline; never loaded in Gazebo.
    urdf.write_text(
        '<robot name="turtlebot3_waffle"><link name="base_footprint"/><link name="base_link"/><joint name="base" type="fixed"><parent link="base_footprint"/><child link="base_link"/></joint></robot>'
    )
    library = tmp_path / "mock.so"
    library.write_bytes(b"\x7fELFoffline-not-loadable")
    return module, tmp_path, urdf, library


def test_fresh_compilation_source_binding_and_no_live_evidence(bootstrap):
    module, root, urdf, library = bootstrap
    output = root / "bootstrap"
    fixture = module.prepare_bindings(
        output, urdf_path=urdf, library_path=library, mission_id="mission", profile_name="waffle"
    )
    brush = json.loads((output / "brush.json").read_text())
    metadata = json.loads((output / "bootstrap.json").read_text())
    binding = fixture["binding"]
    assert (
        binding["body_snapshot_hash"]
        == brush["body_snapshot_hash"]
        == metadata["compiled_body_hash"]
    )
    assert binding["grid"]["resolution"] == 0.05000000074505806
    assert len(binding["grid"]["accessible_cells"]) == 3576
    assert len(binding["scene_model_names"]) == 7
    assert metadata["physical_acceptance"] == "NOT_RUN"
    assert metadata["requires_actual_map_and_source_readmission"] is True
    assert not (output / "measured_map.json").exists()
    assert not (output / "physics_ready.json").exists()
    from fixture_body import configure_fixture_body
    from profiles import PROFILES

    profile = PROFILES["waffle"]
    assert (
        configure_fixture_body(
            root / "fresh-recompilation",
            {
                "body_id": profile.body_id,
                "physical_radius_m": profile.physical_radius_m,
                "cleaning_polygon": profile.cleaning_polygon,
            },
            urdf,
        )
        == binding["body_snapshot_hash"]
    )
    with pytest.raises(FileExistsError):
        module.prepare_bindings(
            output,
            urdf_path=urdf,
            library_path=library,
            mission_id="mission",
            profile_name="waffle",
        )
    second = module.prepare_bindings(
        root / "second",
        urdf_path=urdf,
        library_path=library,
        mission_id="mission",
        profile_name="waffle",
    )
    assert second["binding"]["body_snapshot_hash"] == binding["body_snapshot_hash"]
    assert second["binding"]["run_id"] != binding["run_id"]


@pytest.mark.parametrize("fault", ["mission", "plugin", "profile"])
def test_preparation_rejects_invalid_prerequisites_before_output(bootstrap, fault):
    module, root, urdf, library = bootstrap
    mission, profile = "mission", "waffle"
    if fault == "mission":
        mission = ""
    elif fault == "plugin":
        library.write_bytes(b"not-ELF")
    else:
        profile = "burger"
    output = root / "rejected"
    with pytest.raises(ValueError):
        module.prepare_bindings(
            output, urdf_path=urdf, library_path=library, mission_id=mission, profile_name=profile
        )
    assert not output.exists()
