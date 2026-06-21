"""Tests for ROSClaw Hub asset lifecycle state machine."""

from __future__ import annotations

import pytest

from rosclaw.hub.errors import HubError
from rosclaw.hub.lifecycle import (
    AssetLifecycleState,
    Lifecycle,
    can_transition,
    is_active_state,
    normalize_lifecycle_state,
    terminal_state,
    transition_state,
)

# ---------------------------------------------------------------------------
# normalize_lifecycle_state
# ---------------------------------------------------------------------------


def test_normalize_lifecycle_state_enum() -> None:
    """Passing a state enum returns its value."""
    assert normalize_lifecycle_state(AssetLifecycleState.HEALTHY).value == "healthy"


def test_normalize_lifecycle_state_string() -> None:
    """Passing a valid string returns the canonical value."""
    assert normalize_lifecycle_state("INSTALLED").value == "installed"


def test_normalize_lifecycle_state_invalid() -> None:
    """An invalid state raises HubError."""
    with pytest.raises(HubError):
        normalize_lifecycle_state("nonexistent")


# ---------------------------------------------------------------------------
# can_transition / transition_state
# ---------------------------------------------------------------------------


def test_can_transition_same_state() -> None:
    """Transitioning from a state to itself is allowed."""
    assert can_transition("discovered", "discovered") is True


def test_can_transition_forward() -> None:
    """Forward transitions along the normal path are allowed."""
    assert can_transition("discovered", "downloaded") is True
    assert can_transition("downloaded", "verified") is True
    assert can_transition("verified", "installed") is True
    assert can_transition("installed", "configured") is True
    assert can_transition("configured", "healthy") is True


def test_can_transition_to_removed() -> None:
    """Any active state can transition to the removed terminal state via removing."""
    assert can_transition("discovered", "removing") is True
    assert can_transition("removing", "removed") is True
    assert can_transition("healthy", "removing") is True


def test_can_transition_to_yanked() -> None:
    """Healthy assets can be yanked."""
    assert can_transition("healthy", "yanked") is True


def test_can_transition_repair_unhealthy() -> None:
    """Unhealthy assets can recover to healthy or be removed."""
    assert can_transition("unhealthy", "healthy") is True
    assert can_transition("unhealthy", "removing") is True


def test_can_transition_invalid() -> None:
    """Skipping verification before install is not allowed."""
    assert can_transition("downloaded", "healthy") is False
    assert can_transition("installed", "verified") is False
    assert can_transition("removed", "healthy") is False


def test_transition_state_success() -> None:
    """transition_state returns the target state on success."""
    assert transition_state("discovered", "downloaded") == "downloaded"


def test_transition_state_failure() -> None:
    """transition_state raises HubError on invalid transitions."""
    with pytest.raises(HubError) as exc_info:
        transition_state("discovered", "healthy")
    assert "Illegal lifecycle transition" in str(exc_info.value)


def test_transition_state_with_reason() -> None:
    """transition_state can record a reason in the error message."""
    with pytest.raises(HubError) as exc_info:
        transition_state("discovered", "healthy", reason="jump too far")
    assert "jump too far" in str(exc_info.value)


# ---------------------------------------------------------------------------
# terminal / active helpers
# ---------------------------------------------------------------------------


def test_terminal_state_true() -> None:
    """Removed and yanked are terminal states."""
    assert terminal_state("removed") is True
    assert terminal_state("yanked") is True


def test_terminal_state_false() -> None:
    """Healthy and error-like states are not terminal."""
    assert terminal_state("healthy") is False
    assert terminal_state("unhealthy") is False
    assert terminal_state("deprecated") is False


def test_is_active_state_true() -> None:
    """Installed, configured, and healthy are active states."""
    assert is_active_state("installed") is True
    assert is_active_state("configured") is True
    assert is_active_state("healthy") is True


def test_is_active_state_false() -> None:
    """Removed, removing, and yanked are not active."""
    assert is_active_state("discovered") is True
    assert is_active_state("removing") is False
    assert is_active_state("removed") is False
    assert is_active_state("yanked") is False


# ---------------------------------------------------------------------------
# Lifecycle tracker
# ---------------------------------------------------------------------------


def test_lifecycle_defaults() -> None:
    """A new Lifecycle starts in the discovered state."""
    lc = Lifecycle()
    assert lc.state == "discovered"
    assert lc.history == []


def test_lifecycle_transition_records_history() -> None:
    """Transitions append state and reason to history."""
    lc = Lifecycle()
    lc.transition("downloaded", reason="fetch ok")
    assert lc.state == "downloaded"
    assert len(lc.history) == 1
    assert lc.history[0]["from"] == "discovered"
    assert lc.history[0]["to"] == "downloaded"
    assert lc.history[0]["reason"] == "fetch ok"


def test_lifecycle_invalid_transition_raises() -> None:
    """The tracker rejects invalid transitions."""
    lc = Lifecycle()
    with pytest.raises(HubError):
        lc.transition("healthy")
    assert lc.state == "discovered"


def test_lifecycle_terminal_lock() -> None:
    """Once terminal, no further transitions are allowed."""
    lc = Lifecycle("installed")
    lc.transition("removing", reason="user removed")
    lc.transition("removed", reason="gone")
    assert terminal_state(lc.state) is True
    with pytest.raises(HubError):
        lc.transition("discovered")


def test_lifecycle_to_dict_roundtrip() -> None:
    """Lifecycle serializes and restores its state and history."""
    lc = Lifecycle("discovered")
    lc.transition("downloaded", reason="fetch ok")
    lc.transition("verified", reason="checksum ok")
    data = lc.to_dict()
    restored = Lifecycle.from_dict(data)
    assert restored.state == "verified"
    assert len(restored.history) == 2
    assert restored.history[0]["reason"] == "fetch ok"
