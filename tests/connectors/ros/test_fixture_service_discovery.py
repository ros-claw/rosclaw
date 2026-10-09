"""The fixture may change service reply discovery, never arbitrary DDS QoS."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "integrations/ros_probe/acceptance"
SPEC = importlib.util.spec_from_file_location(
    "fixture_middleware", RUNNER / "fixture_middleware.py"
)
middleware = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(middleware)


def test_only_child_profile_environment_changes_and_evidence_matches(tmp_path):
    environment = {"ROS_DOMAIN_ID": "86", "ROS_LOCALHOST_ONLY": "1"}
    before = dict(environment)
    output = tmp_path / "evidence"
    child = middleware.configure_service_reply_discovery(RUNNER, output, environment)
    assert environment == before
    assert child == {
        **before,
        "FASTRTPS_DEFAULT_PROFILES_FILE": str(RUNNER / middleware.PROFILE_NAME),
    }
    assert (output / middleware.PROFILE_NAME).read_bytes() == (
        RUNNER / middleware.PROFILE_NAME
    ).read_bytes()
    evidence = json.loads((output / "middleware_configuration.json").read_text())
    assert evidence["scope"] == "SERVICE_REPLY_WRITER_ONLY"
    assert evidence["declared_max_blocking_ms"] == 1000
    assert not evidence["physical_acceptance"] and not evidence["startup_deadline_changed"]
    # A second invocation cannot overwrite the original declaration.
    with pytest.raises(FileExistsError):
        middleware.configure_service_reply_discovery(RUNNER, output, environment)


@pytest.mark.parametrize(
    "environment",
    [
        {"RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp"},
        {"RMW_FASTRTPS_USE_QOS_FROM_XML": "1"},
        {"FASTRTPS_DEFAULT_PROFILES_FILE": "/external.xml"},
        {"FASTDDS_DEFAULT_PROFILES_FILE": "/external.xml"},
    ],
)
def test_conflicting_middleware_cannot_silently_override_scope(tmp_path, environment):
    with pytest.raises(ValueError):
        middleware.configure_service_reply_discovery(RUNNER, tmp_path / "evidence", environment)
    assert not (tmp_path / "evidence").exists()


@pytest.mark.parametrize(
    "original,replacement",
    [
        ('profile_name="service"', 'profile_name="service" is_default_profile="true"'),
        ('profile_name="service"', 'profile_name="/nav_cmd_vel"'),
        ("<sec>1</sec>", "<sec>10</sec>"),
        ("<kind>RELIABLE</kind>", "<kind>BEST_EFFORT</kind>"),
        ("</qos>", "<publishMode><kind>ASYNCHRONOUS</kind></publishMode></qos>"),
        ("</profiles>", '<subscriber profile_name="client"/></profiles>'),
    ],
)
def test_widened_or_relaxed_profile_fails_before_environment_or_evidence(
    tmp_path, original, replacement
):
    fixture = tmp_path / "input"
    fixture.mkdir()
    text = (RUNNER / middleware.PROFILE_NAME).read_text()
    assert original in text
    (fixture / middleware.PROFILE_NAME).write_text(text.replace(original, replacement))
    environment = {}
    with pytest.raises(ValueError):
        middleware.configure_service_reply_discovery(fixture, tmp_path / "evidence", environment)
    assert environment == {} and not (tmp_path / "evidence").exists()
