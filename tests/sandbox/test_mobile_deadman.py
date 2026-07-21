"""MuJoCo proof that a continuous mobile command cannot run indefinitely."""

from __future__ import annotations

import pytest

from rosclaw.sandbox.mobile_deadman import run_mobile_deadman_scenario


def test_mobile_base_stops_after_client_lease_is_not_renewed() -> None:
    evidence = run_mobile_deadman_scenario()

    assert evidence.backend == "mujoco"
    assert evidence.has_physics is True
    assert evidence.deadman_tripped_at_sec == pytest.approx(0.26, abs=0.01)
    assert evidence.stopped is True
    assert abs(evidence.final_velocity_mps) <= 0.01
    assert evidence.distance_after_client_loss_m < 0.05


def test_mobile_deadman_rejects_invalid_lease_schedule() -> None:
    with pytest.raises(ValueError, match="renew_interval"):
        run_mobile_deadman_scenario(lease_ttl_ms=100, renew_interval_ms=100)
