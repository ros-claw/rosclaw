import copy

import numpy as np
import pytest

from rosclaw.growth.anchor_plane import AnchorProtectionPlane, fit_protected_readout


def test_arbitrary_plastic_readout_preserves_all_protected_latents_exactly():
    rng = np.random.default_rng(1)
    anchors = rng.normal(size=(9, 33))
    plane = AnchorProtectionPlane(anchors)
    assert plane.rank == 9
    assert plane.plastic_dimensions == 24
    update = rng.normal(size=(37, 33)) * 1e6
    for anchor in anchors:
        assert np.array_equal(update @ plane.project(anchor), np.zeros(37))
    assert np.linalg.norm(plane.project(rng.normal(size=33))) > 1


def test_learning_changes_an_unprotected_input_without_changing_old_outputs():
    plane = AnchorProtectionPlane([[1, 0, 0]])
    output = fit_protected_readout(
        plane,
        current_readout=np.zeros((2, 3)),
        latents=[[0, 1, 0]],
        residual_targets=[[3, -2]],
        sample_weights=[1],
    )
    assert np.allclose(output @ plane.project([0, 1, 0]), [3, -2])
    assert np.array_equal(output @ plane.project([1, 0, 0]), [0, 0])


def test_full_anchor_span_has_no_capacity_and_zero_anchors_have_full_capacity():
    full = AnchorProtectionPlane(np.eye(3))
    assert full.plastic_dimensions == 0
    assert np.array_equal(full.project([1e100, -1e100, 1e100]), np.zeros(3))
    zero = AnchorProtectionPlane(np.zeros((2, 3)))
    assert zero.plastic_dimensions == 3
    assert np.array_equal(zero.project([1, 2, 3]), [1, 2, 3])


def test_serialize_reload_and_input_mutation_do_not_change_plane():
    anchors = np.eye(3)[:1]
    plane = AnchorProtectionPlane(anchors)
    anchors[:] = 999
    loaded = AnchorProtectionPlane.from_dict(plane.to_dict())
    assert np.array_equal(loaded.project([1, 2, 3]), plane.project([1, 2, 3]))
    changed = copy.deepcopy(plane.to_dict())
    changed["anchors"][0][0] = 2
    with pytest.raises(ValueError):
        AnchorProtectionPlane.from_dict(changed)


@pytest.mark.parametrize("values", [[], [1, 2], [[float("nan"), 0]], np.zeros((1, 513))])
def test_invalid_anchor_bank_fails_closed(values):
    with pytest.raises(ValueError):
        AnchorProtectionPlane(values)


def test_projection_and_fit_reject_nonfinite_or_invalid_weights():
    plane = AnchorProtectionPlane([[1, 0, 0]])
    with pytest.raises(ValueError):
        plane.project([1, float("inf"), 0])
    with pytest.raises(ValueError):
        fit_protected_readout(
            plane,
            current_readout=np.zeros((2, 3)),
            latents=[[0, 1, 0]],
            residual_targets=[[3, -2]],
            sample_weights=[0],
        )
    assert plane.to_dict()["hardware_authorized"] is False
    assert plane.to_dict()["promotion_authorized"] is False
