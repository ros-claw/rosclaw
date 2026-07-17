from pathlib import Path

from rosclaw.body.rh56.resources import rh56_config_path, rh56_reference_policy_path


def test_bundled_rh56_resources_resolve() -> None:
    profile = rh56_config_path("rh56_right_rs485_v1.yaml")
    calibration = rh56_config_path("rh56_right_01_calibration.yaml")
    policy = rh56_reference_policy_path()

    assert profile.is_file()
    assert calibration.is_file()
    assert (policy / "config.json").is_file()
    assert (policy / "model.safetensors").is_file()


def test_bundled_rh56_resource_paths_are_absolute() -> None:
    assert Path(rh56_config_path("rh56_can_v1.yaml")).is_absolute()
    assert rh56_reference_policy_path().is_absolute()
