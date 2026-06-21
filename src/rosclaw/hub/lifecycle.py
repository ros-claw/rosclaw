"""Asset lifecycle state machine for the ROSClaw Hub.

The lifecycle tracks an installed asset through discovery, verification,
installation, configuration, and health monitoring.  Transitions are
restricted so that an asset cannot jump from ``discovered`` to ``healthy``
without passing through the intermediate states that record what was actually
done.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from rosclaw.hub.errors import HubError, HubErrorCode


class AssetLifecycleState(StrEnum):
    """Canonical lifecycle states for an installed asset."""

    DISCOVERED = "discovered"
    DOWNLOADED = "downloaded"
    VERIFIED = "verified"
    INSTALLED = "installed"
    CONFIGURED = "configured"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    REMOVING = "removing"
    REMOVED = "removed"
    DEPRECATED = "deprecated"
    YANKED = "yanked"


# Directed graph of allowed state transitions.  Missing edges are illegal.
_TRANSITIONS: dict[str, set[str]] = {
    AssetLifecycleState.DISCOVERED.value: {
        AssetLifecycleState.DOWNLOADED.value,
        AssetLifecycleState.VERIFIED.value,
        AssetLifecycleState.INSTALLED.value,
        AssetLifecycleState.REMOVING.value,
    },
    AssetLifecycleState.DOWNLOADED.value: {
        AssetLifecycleState.VERIFIED.value,
        AssetLifecycleState.INSTALLED.value,
        AssetLifecycleState.REMOVING.value,
    },
    AssetLifecycleState.VERIFIED.value: {
        AssetLifecycleState.INSTALLED.value,
        AssetLifecycleState.REMOVING.value,
    },
    AssetLifecycleState.INSTALLED.value: {
        AssetLifecycleState.CONFIGURED.value,
        AssetLifecycleState.HEALTHY.value,
        AssetLifecycleState.UNHEALTHY.value,
        AssetLifecycleState.REMOVING.value,
    },
    AssetLifecycleState.CONFIGURED.value: {
        AssetLifecycleState.HEALTHY.value,
        AssetLifecycleState.UNHEALTHY.value,
        AssetLifecycleState.REMOVING.value,
    },
    AssetLifecycleState.HEALTHY.value: {
        AssetLifecycleState.UNHEALTHY.value,
        AssetLifecycleState.DEPRECATED.value,
        AssetLifecycleState.YANKED.value,
        AssetLifecycleState.REMOVING.value,
    },
    AssetLifecycleState.UNHEALTHY.value: {
        AssetLifecycleState.HEALTHY.value,
        AssetLifecycleState.REMOVING.value,
        AssetLifecycleState.DEPRECATED.value,
    },
    AssetLifecycleState.DEPRECATED.value: {
        AssetLifecycleState.REMOVING.value,
    },
    AssetLifecycleState.YANKED.value: {
        AssetLifecycleState.REMOVING.value,
    },
    AssetLifecycleState.REMOVING.value: {
        AssetLifecycleState.REMOVED.value,
        AssetLifecycleState.UNHEALTHY.value,
    },
}


def normalize_lifecycle_state(state: str) -> AssetLifecycleState:
    """Return the canonical lifecycle state for *state*.

    Raises:
        HubError: If *state* is not a recognized lifecycle state.
    """
    try:
        return AssetLifecycleState(state.lower())
    except ValueError as exc:
        raise HubError(
            code=HubErrorCode.INCOMPATIBLE_RUNTIME,
            message=f"Unknown lifecycle state: {state!r}",
            suggested_fix=(f"Use one of: {', '.join(s.value for s in AssetLifecycleState)}"),
        ) from exc


def can_transition(from_state: str, to_state: str) -> bool:
    """Return whether a transition from *from_state* to *to_state* is legal."""
    from_normalized = normalize_lifecycle_state(from_state).value
    to_normalized = normalize_lifecycle_state(to_state).value
    if from_normalized == to_normalized:
        return True
    return to_normalized in _TRANSITIONS.get(from_normalized, set())


def transition_state(
    current: str,
    target: str,
    *,
    reason: str | None = None,
) -> str:
    """Validate and return the new lifecycle state.

    Args:
        current: The current lifecycle state.
        target: The desired lifecycle state.
        reason: Optional human-readable reason for the transition.

    Returns:
        The normalized target state string.

    Raises:
        HubError: If the transition is not allowed.
    """
    if can_transition(current, target):
        return normalize_lifecycle_state(target).value

    current_normalized = normalize_lifecycle_state(current).value
    target_normalized = normalize_lifecycle_state(target).value
    message = f"Illegal lifecycle transition from {current_normalized!r} to {target_normalized!r}"
    if reason:
        message = f"{message}: {reason}"
    raise HubError(
        code=HubErrorCode.INCOMPATIBLE_RUNTIME,
        message=message,
        suggested_fix="Follow the normal install/upgrade flow or uninstall first.",
    )


def terminal_state(state: str) -> bool:
    """Return whether *state* is a terminal lifecycle state."""
    normalized = normalize_lifecycle_state(state).value
    return normalized in {
        AssetLifecycleState.REMOVED.value,
        AssetLifecycleState.YANKED.value,
    }


def is_active_state(state: str) -> bool:
    """Return whether *state* represents an asset that is still installed."""
    normalized = normalize_lifecycle_state(state).value
    return normalized not in {
        AssetLifecycleState.REMOVED.value,
        AssetLifecycleState.REMOVING.value,
        AssetLifecycleState.YANKED.value,
    }


class Lifecycle:
    """Mutable lifecycle tracker for a single asset.

    The tracker keeps a short in-memory history of transitions.  It is the
    installer's responsibility to persist the current state to the lockfile.
    """

    def __init__(
        self,
        state: str = AssetLifecycleState.DISCOVERED.value,
        history: list[dict[str, Any]] | None = None,
    ) -> None:
        self._state = normalize_lifecycle_state(state).value
        self._history: list[dict[str, Any]] = list(history or [])

    @property
    def state(self) -> str:
        """Current lifecycle state."""
        return self._state

    @property
    def history(self) -> list[dict[str, Any]]:
        """Transition history records."""
        return list(self._history)

    def transition(
        self,
        target: str,
        *,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Move to *target* if allowed and record the transition.

        Returns:
            The new state string.

        Raises:
            HubError: If the transition is illegal.
        """
        new_state = transition_state(self._state, target, reason=reason)
        record: dict[str, Any] = {
            "from": self._state,
            "to": new_state,
        }
        if reason:
            record["reason"] = reason
        if metadata:
            record["metadata"] = metadata
        self._history.append(record)
        self._state = new_state
        return new_state

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "state": self._state,
            "history": self._history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Lifecycle:
        """Restore a :class:`Lifecycle` from a serialized dict."""
        return cls(
            state=data.get("state", AssetLifecycleState.DISCOVERED.value),
            history=data.get("history"),
        )
