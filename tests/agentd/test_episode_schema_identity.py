"""Late SDK session creation can leave a hash-valid, undeliverable schema."""

import copy
import hashlib
import json

import pytest

from rosclaw.agentd.episode_schema_identity import (
    EpisodeSchemaIdentityError,
    validate_episode_schema_identity,
)


def packet():
    spec = {"new_UUID": "session-new", "nonce": "episode-once"}
    schema = {
        "required": ["UUID", "nonce"],
        "properties": {
            "UUID": {"const": spec["new_UUID"]},
            "nonce": {"const": spec["nonce"]},
        },
    }
    expected = {"UUID": spec["new_UUID"], "nonce": spec["nonce"]}
    return schema, expected


def test_matching_identity_is_read_only():
    schema, expected = packet()
    before = copy.deepcopy((schema, expected))
    validate_episode_schema_identity(schema, expected)
    assert (schema, expected) == before


def test_fresh_file_hash_does_not_admit_stale_sdk_uuid(tmp_path):
    schema, expected = packet()
    schema["properties"]["UUID"]["const"] = "session-before-sdk-create"
    path = tmp_path / "delivery.schema.json"
    path.write_text(json.dumps(schema))
    bound_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == bound_sha
    with pytest.raises(EpisodeSchemaIdentityError, match="MISMATCH:UUID"):
        validate_episode_schema_identity(json.loads(path.read_text()), expected)
    assert not (tmp_path / "clock.json").exists()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == bound_sha


@pytest.mark.parametrize("field", ["UUID", "nonce"])
def test_optional_or_nonconstant_identity_is_rejected(field):
    schema, expected = packet()
    schema["required"].remove(field)
    with pytest.raises(EpisodeSchemaIdentityError, match="NOT_REQUIRED"):
        validate_episode_schema_identity(schema, expected)
    schema["required"].append(field)
    schema["properties"][field] = {"enum": [expected[field], "foreign"]}
    with pytest.raises(EpisodeSchemaIdentityError, match="CONST_REQUIRED"):
        validate_episode_schema_identity(schema, expected)


@pytest.mark.parametrize("expected", [{}, {"UUID": ""}, {"UUID": True}])
def test_independent_identity_must_be_present_and_typed(expected):
    schema, _ = packet()
    with pytest.raises(EpisodeSchemaIdentityError, match="INDEPENDENT"):
        validate_episode_schema_identity(schema, expected)
