import numpy as np
import pytest

from rosclaw.control.fixed_value_scale import FixedValueScale


def test_affine_conversion_preserves_original_units():
    x = np.array([[1.0, 2.0, 3.0], [-3.0, 5.0, 2.0]])
    w, b = np.array([[0.1, 0.2, -0.3]]), np.array([0.4])
    scale = FixedValueScale.from_targets(np.arange(1000.0, 2000.0))
    nw, nb = scale.normalize_affine(w, b)
    np.testing.assert_allclose(
        (x @ nw.T + nb) * scale.scale + scale.center, x @ w.T + b, atol=1e-12
    )
    assert not np.shares_memory(nw, w)


def test_constant_zero_targets_do_not_amplify_a_joint_critic_loss():
    assert FixedValueScale.from_targets(np.zeros(100)) == FixedValueScale(0.0, 1.0)


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf], [1e7], [[1.0, 2.0]]])
def test_invalid_targets_rejected(values):
    with pytest.raises(ValueError):
        FixedValueScale.from_targets(np.asarray(values))
