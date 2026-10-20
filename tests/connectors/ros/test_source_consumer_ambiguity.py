"""Adversarial static XML: ambiguity never establishes a consumer contract."""

import hashlib

import pytest

from rosclaw.connectors.ros.engineering.source_consumer import inspect_source_consumer
from tests.connectors.ros.test_source_consumer_diagnosis import (
    ENTITY,
    known_sdk_originals,
    sealed_graph,
)


@pytest.mark.parametrize("prefix", ["gz", "ns0"])
@pytest.mark.parametrize("side", ["urdf", "sdf"])
@pytest.mark.parametrize("field", ["mu", "mu2", "fdir1"])
def test_duplicate_friction_fields_abstain_without_a_candidate(prefix, side, field):
    values = known_sdk_originals(prefix)
    index = 1 if side == "urdf" else 2
    closing = f"</{field}>".encode()
    # Duplicate identical values are still ambiguous; a consumer may select a
    # first or last occurrence, reject the document or apply a schema default.
    original = values[index]
    start = original.index(f"<{field}".encode())
    end = original.index(closing, start) + len(closing)
    values[index] = original[:end] + original[start:end] + original[end:]
    if index == 1:
        data = values[0].model_dump(mode="json")
        data["source_urdf_sha256"] = hashlib.sha256(values[1]).hexdigest()
        values[0] = sealed_graph(data)
    review, _ = inspect_source_consumer(*values, robot_entity=ENTITY, include_compatibility=True)
    assert review["status"] == "UNKNOWN"
    assert "files" not in review["compatibility_proposal"]


@pytest.mark.parametrize("prefix", ["gz", "ns0"])
@pytest.mark.parametrize("side", ["urdf", "sdf"])
@pytest.mark.parametrize("name", ["base", "wheel"])
def test_duplicate_link_identity_abstains(prefix, side, name):
    values = known_sdk_originals(prefix)
    index = 1 if side == "urdf" else 2
    marker = b"</robot>" if index == 1 else b"</model>"
    values[index] = values[index].replace(marker, f'<link name="{name}"/>'.encode() + marker)
    if index == 1:
        data = values[0].model_dump(mode="json")
        data["source_urdf_sha256"] = hashlib.sha256(values[1]).hexdigest()
        values[0] = sealed_graph(data)
    review, _ = inspect_source_consumer(*values, robot_entity=ENTITY, include_compatibility=True)
    assert review["status"] == "UNKNOWN"
    assert "files" not in review["compatibility_proposal"]


@pytest.mark.parametrize("prefix", ["gz", "ns0"])
def test_model_and_run_witness_cannot_be_taken_from_different_worlds(prefix):
    values = known_sdk_originals(prefix)
    values[2] = values[2].replace(b"</sdf>", b'<world name="unrelated"/></sdf>')
    review, _ = inspect_source_consumer(*values, robot_entity=ENTITY, include_compatibility=True)
    assert review["status"] == "UNKNOWN"
    assert "files" not in review["compatibility_proposal"]


@pytest.mark.parametrize("tag", ["surface", "friction", "ode"])
def test_duplicate_friction_containers_cannot_choose_the_first(tag):
    values = known_sdk_originals("gz")
    raw = values[2]
    start, end = raw.index(f"<{tag}>".encode()), raw.index(f"</{tag}>".encode()) + len(tag) + 3
    values[2] = raw[:end] + raw[start:end] + raw[end:]
    review, _ = inspect_source_consumer(*values, robot_entity=ENTITY, include_compatibility=True)
    assert review["status"] == "UNKNOWN"
    assert "files" not in review["compatibility_proposal"]


@pytest.mark.parametrize("side", ["urdf", "sdf"])
@pytest.mark.parametrize("replacement", [b"<link/>", b'<link name=""/>'])
def test_missing_link_identity_abstains(side, replacement):
    values = known_sdk_originals()
    index = 1 if side == "urdf" else 2
    values[index] = values[index].replace(b'<link name="base"/>', replacement)
    if index == 1:
        data = values[0].model_dump(mode="json")
        data["source_urdf_sha256"] = hashlib.sha256(values[1]).hexdigest()
        values[0] = sealed_graph(data)
    review, _ = inspect_source_consumer(*values, robot_entity=ENTITY, include_compatibility=True)
    assert review["status"] == "UNKNOWN"
    assert "files" not in review["compatibility_proposal"]


def test_duplicate_sdf_collision_identity_abstains():
    values = known_sdk_originals("gz")
    raw = values[2]
    start, end = raw.index(b'<collision name="wheel">'), raw.index(b"</collision>") + 12
    values[2] = raw[:end] + raw[start:end] + raw[end:]
    # Duplicate the source facts too: comparing multisets alone must not accept
    # two ambiguous SDF collision identities as faithful source correspondence.
    raw = values[1]
    start, end = raw.index(b"<collision>"), raw.index(b"</collision>") + 12
    values[1] = raw[:end] + raw[start:end] + raw[end:]
    data = values[0].model_dump(mode="json")
    data["source_urdf_sha256"] = hashlib.sha256(values[1]).hexdigest()
    values[0] = sealed_graph(data)
    review, _ = inspect_source_consumer(*values, robot_entity=ENTITY, include_compatibility=True)
    assert review["status"] == "UNKNOWN"
    assert "files" not in review["compatibility_proposal"]


@pytest.mark.parametrize("prefix,expected", [("gz", "SATISFIED"), ("ns0", "VIOLATED")])
def test_other_named_models_and_distinct_collisions_preserve_diagnosis(prefix, expected):
    values = known_sdk_originals(prefix)
    raw = values[1]
    start, end = raw.index(b"<collision>"), raw.index(b"</collision>") + 12
    values[1] = raw[:end] + raw[start:end] + raw[end:]
    raw = values[2]
    start, end = raw.index(b'<collision name="wheel">'), raw.index(b"</collision>") + 12
    second = raw[start:end].replace(b'name="wheel"', b'name="another_collision"')
    values[2] = (raw[:end] + second + raw[end:]).replace(
        b"</world>", b'<model name="other"><link name="base"/></model></world>'
    )
    data = values[0].model_dump(mode="json")
    data["source_urdf_sha256"] = hashlib.sha256(values[1]).hexdigest()
    values[0] = sealed_graph(data)
    review, _ = inspect_source_consumer(*values, robot_entity=ENTITY, include_compatibility=True)
    assert review["status"] == expected
    assert len(review["original_friction_facts"]) == 2
