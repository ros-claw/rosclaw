from __future__ import annotations

import numpy as np
import pytest

from rosclaw.simforge.g1_torque_authority import (
    project_g1_additive_torque_authority,
    project_g1_torque_authority,
)


def test_g1_torque_authority_projects_and_preserves_audit_metrics() -> None:
    command = np.zeros(29, dtype=np.float64)
    limits = np.full(29, 100.0, dtype=np.float64)
    command[8] = -143.0
    command[10] = 50.0

    result = project_g1_torque_authority(
        commanded_torque_nm=command,
        hard_limits_nm=limits,
        maximum_demand_ratio=0.98,
    )

    assert result.active
    assert result.projected_joint_count == 1
    assert result.preprojection_peak_demand_ratio == pytest.approx(1.43)
    assert result.projected_peak_demand_ratio == pytest.approx(0.98)
    assert result.projected_torque_nm[8] == pytest.approx(-98.0)
    assert result.correction_nm[8] == pytest.approx(45.0)
    assert result.projected_torque_nm[10] == pytest.approx(50.0)


@pytest.mark.parametrize("ratio", [0.0, 0.89, 0.991, float("nan")])
def test_g1_torque_authority_rejects_invalid_ratios(ratio: float) -> None:
    with pytest.raises(ValueError, match="ratio"):
        project_g1_torque_authority(
            commanded_torque_nm=np.zeros(29),
            hard_limits_nm=np.ones(29),
            maximum_demand_ratio=ratio,
        )


def test_g1_torque_authority_rejects_invalid_vectors() -> None:
    with pytest.raises(ValueError, match="29-DoF"):
        project_g1_torque_authority(
            commanded_torque_nm=np.zeros(28),
            hard_limits_nm=np.ones(29),
            maximum_demand_ratio=0.98,
        )


def test_g1_additive_authority_preserves_direction_with_common_scale() -> None:
    parent = np.zeros(29, dtype=np.float64)
    parent[8] = -40.0
    additive = np.zeros(29, dtype=np.float64)
    additive[8] = -80.0
    additive[10] = 40.0
    limits = np.full(29, 100.0, dtype=np.float64)

    result = project_g1_additive_torque_authority(
        parent_torque_nm=parent,
        additive_torque_nm=additive,
        hard_limits_nm=limits,
        maximum_demand_ratio=0.98,
    )

    assert result.active
    assert result.scale == pytest.approx(0.725)
    np.testing.assert_allclose(result.projected_additive_torque_nm[[8, 10]], (-58.0, 29.0))


def test_g1_additive_authority_returns_full_scale_with_headroom() -> None:
    result = project_g1_additive_torque_authority(
        parent_torque_nm=np.zeros(29),
        additive_torque_nm=np.full(29, 1.0),
        hard_limits_nm=np.full(29, 100.0),
        maximum_demand_ratio=0.98,
    )

    assert not result.active
    assert result.scale == 1.0
