"""Match public output declarations to operator policy before starting work."""

from collections.abc import Sequence
from pathlib import PurePosixPath


class EpisodeWriteScopeError(ValueError):
    """Public output paths and independent operator policy cannot be reconciled."""


def _paths(values: Sequence[str], label: str) -> frozenset[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise EpisodeWriteScopeError(f"OUTPUT_PATH_LIST_REQUIRED:{label}")
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value or "\x00" in value:
            raise EpisodeWriteScopeError(f"OUTPUT_PATH_REQUIRED:{label}")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or not path.parts
            or ".." in path.parts
            or "\\" in value
            or path.as_posix() != value
        ):
            raise EpisodeWriteScopeError(f"CANONICAL_RELATIVE_PATH_REQUIRED:{label}")
        if value in result:
            raise EpisodeWriteScopeError(f"DUPLICATE_OUTPUT_PATH:{label}")
        result.add(value)
    return frozenset(result)


def validate_episode_write_scope(
    public_outputs: Sequence[str],
    operator_outputs: Sequence[str],
    *,
    protected_inputs: Sequence[str] = (),
) -> frozenset[str]:
    """Require identical explicit relative paths, including logs and reports.

    Callers must supply independently admitted operator policy and immutable
    public declarations before creating a clock or model worker. Prose such
    as "reports/logs allowed" does not define a path list. This read-only
    check grants no authority and provides no filesystem or symlink isolation.
    Existing workers and frozen episodes are never changed by this helper.
    """
    public = _paths(public_outputs, "public")
    operator = _paths(operator_outputs, "operator")
    protected = _paths(protected_inputs, "protected")
    if public != operator:
        raise EpisodeWriteScopeError("PUBLIC_OPERATOR_OUTPUT_SCOPE_MISMATCH")
    if public & protected:
        raise EpisodeWriteScopeError("OUTPUT_OVERLAPS_PROTECTED_INPUT")
    return public
