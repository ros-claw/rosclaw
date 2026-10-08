"""Actual isolated SDK contact mapping plus corruption, no live physics claims."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from tests.connectors.ros.test_physics_component_packets import parse


def sdk_rows():
    path = Path(__file__).parent / "fixtures/passive-ecm-contact-v4-contract-packets.jsonl"
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if json.loads(line)["case"].startswith("v4_")
    ]


def packet():
    return deepcopy(sdk_rows()[0]["packet"])


def policy():
    return (
        {
            "sensor_name": "actual_contact_sensor",
            "link_name": "actual_base",
            "collision_name": "actual_collision",
            "gz_topic": "/synthetic/body_contact",
        },
    )


@pytest.mark.parametrize("row", sdk_rows(), ids=lambda row: row["case"])
def test_sdk_actual_contact_sensor_mapping_and_fault_packets(row):
    if not row["packet"]["complete"]:
        assert "body" not in row["packet"]
        with pytest.raises(ValueError, match="complete"):
            parse(row["packet"], required_body_contact_mapping=policy())
        return
    result = parse(
        row["packet"],
        required_body_reference_link="actual_base",
        required_body_contact_mapping=policy(),
        maximum_body_planar_radius_m=0.25,
    )
    sources = result["body_contact_sources"]
    assert len(sources) == 1
    assert sources[0]["collision_entity_id"] == result["body_collision_geometry"][0]["entity_id"]
    assert sources[0]["link_entity_id"] == result["body_reference_link"]["entity_id"]
    assert len(result["body_contact_mapping_hash"]) == 64
    assert result["geometry"].model_radii[0][0] == "anonymous_blocker"
    assert result["geometry"].model_radii[0][1] == pytest.approx(0.3)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "extra",
        "duplicate",
        "sensor_entity",
        "link_entity",
        "collision_entity",
        "boolean",
        "reference_name",
        "sensor_name",
        "collision_name",
        "topic",
        "relative",
        "oversized",
        "downgrade",
    ],
)
def test_substituted_or_incomplete_actual_contact_mapping_refuses_before_admission(fault):
    p = packet()
    rows = p["body"]["contact_sources"]
    row = rows[0]
    if fault == "missing":
        rows.clear()
    elif fault == "extra":
        row["operator_claim"] = True
    elif fault == "duplicate":
        rows.append(deepcopy(row))
    elif fault == "sensor_entity":
        row["sensor_entity_id"] = p["body"]["entity_id"]
    elif fault == "link_entity":
        row["link_entity_id"] = row["sensor_entity_id"]
    elif fault == "collision_entity":
        row["collision_entity_id"] = p["obstacles"][0]["collision_geometry"][0]["entity_id"]
    elif fault == "boolean":
        row["sensor_entity_id"] = True
    elif fault == "reference_name":
        row["link_name"] = "foreign"
    elif fault == "sensor_name":
        row["sensor_name"] = "foreign"
    elif fault == "collision_name":
        row["collision_name"] = "foreign"
    elif fault == "topic":
        row["gz_topic"] = "/foreign"
    elif fault == "relative":
        row["gz_topic"] = "relative"
    elif fault == "oversized":
        row["sensor_name"] = "x" * 257
    else:
        p["schema_version"] = "rosclaw.gazebo_postupdate_observation.v3"
        p["body"].pop("contact_sources")
    with pytest.raises(ValueError):
        parse(p, required_body_contact_mapping=policy())


@pytest.mark.parametrize("value", [[], (), ({},), (policy()[0], policy()[0])])
def test_untyped_or_incomplete_contact_policy_cannot_replace_component_binding(value):
    with pytest.raises(ValueError):
        parse(packet(), required_body_contact_mapping=value)
