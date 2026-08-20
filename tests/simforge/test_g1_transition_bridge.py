from __future__ import annotations

import numpy as np
import pytest

from rosclaw.simforge.g1_transition_bridge import (
    G1TransitionBridgeConfig,
    G1VelocityMatchedTransitionBridge,
)


def test_velocity_matched_bridge_satisfies_position_velocity_boundaries() -> None:
    entry_position = np.linspace(-0.3, 0.4, 29)
    exit_position = np.linspace(0.5, -0.2, 29)
    entry_velocity = np.linspace(-1.0, 1.0, 29)
    exit_velocity = np.linspace(0.4, -0.4, 29)
    config = G1TransitionBridgeConfig(
        duration_sec=0.50,
        entry_velocity_scale=0.6,
        exit_velocity_scale=1.0,
    )
    bridge = G1VelocityMatchedTransitionBridge(
        entry_position=entry_position,
        entry_velocity=entry_velocity,
        exit_position=exit_position,
        exit_velocity=exit_velocity,
        config=config,
    )

    start = bridge.sample(0.0)
    end = bridge.sample(config.duration_sec)

    np.testing.assert_allclose(start.position, entry_position, atol=1e-12)
    np.testing.assert_allclose(start.velocity, entry_velocity * 0.6, atol=1e-12)
    np.testing.assert_allclose(start.acceleration, 0.0, atol=1e-12)
    np.testing.assert_allclose(end.position, exit_position, atol=1e-12)
    np.testing.assert_allclose(end.velocity, exit_velocity, atol=1e-12)
    np.testing.assert_allclose(end.acceleration, 0.0, atol=1e-11)
    assert bridge.audit_dict()["velocity_matched"] is True


def test_zero_velocity_bridge_is_legacy_minimum_jerk_position_blend() -> None:
    entry = np.zeros(29)
    exit = np.ones(29)
    bridge = G1VelocityMatchedTransitionBridge(
        entry_position=entry,
        entry_velocity=np.ones(29),
        exit_position=exit,
        exit_velocity=-np.ones(29),
        config=G1TransitionBridgeConfig(duration_sec=0.60),
    )

    midpoint = bridge.sample(0.30)

    np.testing.assert_allclose(midpoint.position, 0.5, atol=1e-12)
    np.testing.assert_allclose(bridge.sample(0.0).velocity, 0.0, atol=1e-12)
    np.testing.assert_allclose(bridge.sample(0.60).velocity, 0.0, atol=1e-12)


def test_transition_bridge_supports_a_slow_recovery_without_weakening_bounds() -> None:
    config = G1TransitionBridgeConfig(duration_sec=2.0, entry_velocity_scale=0.65)
    bridge = G1VelocityMatchedTransitionBridge(
        entry_position=np.zeros(29),
        entry_velocity=np.ones(29),
        exit_position=np.ones(29),
        exit_velocity=np.zeros(29),
        config=config,
    )

    np.testing.assert_allclose(bridge.sample(2.0).position, 1.0, atol=1e-12)
    np.testing.assert_allclose(bridge.sample(2.0).velocity, 0.0, atol=1e-12)
    with pytest.raises(ValueError, match="duration"):
        G1TransitionBridgeConfig(duration_sec=2.01)


def test_transition_bridge_projects_boundary_velocity_and_fails_closed() -> None:
    bridge = G1VelocityMatchedTransitionBridge(
        entry_position=np.zeros(29),
        entry_velocity=np.full(29, 10.0),
        exit_position=np.ones(29),
        exit_velocity=np.full(29, -10.0),
        config=G1TransitionBridgeConfig(
            entry_velocity_scale=1.0,
            exit_velocity_scale=1.0,
            maximum_boundary_velocity_rad_s=1.5,
        ),
    )

    assert bridge.boundary_velocity_projection_applied
    np.testing.assert_allclose(bridge.entry_velocity, 1.5)
    np.testing.assert_allclose(bridge.exit_velocity, -1.5)
    with pytest.raises(ValueError, match="29 joints"):
        G1VelocityMatchedTransitionBridge(
            entry_position=np.zeros(28),
            entry_velocity=np.zeros(29),
            exit_position=np.zeros(29),
            exit_velocity=np.zeros(29),
            config=G1TransitionBridgeConfig(),
        )
    with pytest.raises(ValueError, match="entry velocity scale"):
        G1TransitionBridgeConfig(entry_velocity_scale=1.01)
    with pytest.raises(ValueError, match="duration"):
        G1TransitionBridgeConfig(duration_sec=0.15)
    with pytest.raises(ValueError, match="sample time"):
        bridge.sample(float("nan"))
