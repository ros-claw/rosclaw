"""Read-only identity admission before an episode acquires a clock or worker."""

from collections.abc import Mapping


class EpisodeSchemaIdentityError(ValueError):
    """A delivery schema disagrees with independently admitted identity."""


def validate_episode_schema_identity(
    schema: Mapping[str, object], expected_identity: Mapping[str, str]
) -> None:
    """Require exact required constants supplied independently of the schema.

    Callers map their admitted spec's aliases explicitly, e.g. ``new_UUID`` to
    the delivery field ``UUID``. Hash agreement alone cannot detect stale
    constants after SDK session creation. This grants no execution authority
    and does not replace full JSON Schema validation.
    """
    if not expected_identity or any(
        not isinstance(key, str) or not key or not isinstance(value, str) or not value
        for key, value in expected_identity.items()
    ):
        raise EpisodeSchemaIdentityError("INDEPENDENT_SCHEMA_IDENTITY_REQUIRED")
    required = schema.get("required")
    properties = schema.get("properties")
    if not isinstance(required, list) or not isinstance(properties, Mapping):
        raise EpisodeSchemaIdentityError("SCHEMA_IDENTITY_PROPERTIES_REQUIRED")
    for field, expected in expected_identity.items():
        if field not in required:
            raise EpisodeSchemaIdentityError(f"SCHEMA_IDENTITY_NOT_REQUIRED:{field}")
        definition = properties.get(field)
        if not isinstance(definition, Mapping) or "const" not in definition:
            raise EpisodeSchemaIdentityError(f"SCHEMA_IDENTITY_CONST_REQUIRED:{field}")
        actual = definition["const"]
        if not isinstance(actual, str) or actual != expected:
            raise EpisodeSchemaIdentityError(f"SCHEMA_IDENTITY_MISMATCH:{field}")
