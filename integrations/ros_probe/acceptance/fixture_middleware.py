"""Service discovery mitigation for the owned SDK fixture, never robot policy.

The frozen Fast DDS SDK waits 100 ms for a service reply reader to match.
A 1 s service-only bound tolerates slower matching without changing ordinary
topic QoS, the fixture startup deadline, or mission/control-plane timeouts.
This mitigates a known failure class; it does not identify every startup fault.
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

PROFILE_NAME = "service_reply_discovery.xml"
NS = "{http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles}"


def configure_service_reply_discovery(root, output, environment):
    """Return child environment and retain the exact, narrowly scoped profile."""
    profile = Path(root) / PROFILE_NAME
    raw = profile.read_bytes()
    document = ET.fromstring(raw)
    expected = ET.fromstring(
        f'<dds xmlns="{NS[1:-1]}"><profiles><publisher profile_name="service">'
        "<qos><reliability><kind>RELIABLE</kind><max_blocking_time><sec>1</sec>"
        "<nanosec>0</nanosec></max_blocking_time></reliability></qos>"
        "</publisher></profiles></dds>"
    )

    def shape(element):
        return (
            element.tag,
            element.attrib,
            (element.text or "").strip(),
            tuple(shape(child) for child in element),
        )

    if shape(document) != shape(expected):
        raise ValueError("fixture middleware profile must change only service reply blocking time")
    if environment.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp") != "rmw_fastrtps_cpp":
        raise ValueError("fixture discovery profile requires the verified Fast DDS implementation")
    if environment.get("RMW_FASTRTPS_USE_QOS_FROM_XML", "0") != "0":
        raise ValueError("fixture does not override XML publication or memory policy")
    for key in ("FASTRTPS_DEFAULT_PROFILES_FILE", "FASTDDS_DEFAULT_PROFILES_FILE"):
        if environment.get(key) and Path(environment[key]).resolve() != profile.resolve():
            raise ValueError("conflicting external Fast DDS profile: " + key)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / PROFILE_NAME).open("xb") as stream:
        stream.write(raw)
    evidence = {
        "schema_version": "rosclaw.fixture_service_discovery_profile.v1",
        "source_path": str(profile.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "scope": "SERVICE_REPLY_WRITER_ONLY",
        "declared_max_blocking_ms": 1000,
        "actual_RMW_QoS": "SEPARATE_SDK_COMPONENT_OBSERVATION_REQUIRED",
        "startup_deadline_changed": False,
        "mission_or_lease_timeout_changed": False,
        "startup_retries": 0,
        "physical_acceptance": False,
    }
    with (output / "middleware_configuration.json").open("x") as stream:
        stream.write(json.dumps(evidence, indent=2) + "\n")
    child_environment = dict(environment)
    child_environment["FASTRTPS_DEFAULT_PROFILES_FILE"] = str(profile.resolve())
    return child_environment
