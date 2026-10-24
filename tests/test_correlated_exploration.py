import json

import numpy as np
import pytest

from rosclaw.control.correlated_exploration import (
    CorrelatedExplorationConfig,
    CorrelatedGaussianExploration,
)


def test_seed_reset_and_snapshot_resume_are_exact_without_global_rng(monkeypatch):
    monkeypatch.setattr(
        np.random, "standard_normal", lambda *a, **k: pytest.fail("global RNG used")
    )
    config = CorrelatedExplorationConfig(seed=926028)
    stream = CorrelatedGaussianExploration(3, config)
    means, log_std = np.zeros(3), np.full(3, -3.0)
    start = [stream.perturb(means, log_std) for _ in range(7)]
    snapshot = json.loads(json.dumps(stream.snapshot()))
    expected = stream.perturb(means, log_std)
    resumed = CorrelatedGaussianExploration(3, config)
    resumed.restore(snapshot)
    np.testing.assert_array_equal(expected, resumed.perturb(means, log_std))
    stream.reset()
    for value in start:
        np.testing.assert_array_equal(value, stream.perturb(means, log_std))


def test_zero_scale_is_exact_and_does_not_draw():
    stream = CorrelatedGaussianExploration(2, CorrelatedExplorationConfig(seed=1, scale=0))
    mean = np.array([0.1, 0.2], dtype=np.float32)
    result = stream.perturb(mean, np.zeros(2))
    np.testing.assert_array_equal(mean, result)
    assert result.dtype == mean.dtype
    assert not np.shares_memory(mean, result)
    assert stream.draw_steps == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"seed": True},
        {"seed": 2**32},
        {"seed": 0, "scale": 0.26},
        {"seed": 0, "scale": np.nan},
        {"seed": 0, "correlation": 1},
        {"seed": 0, "activation_ceiling": "REAL"},
    ],
)
def test_invalid_or_hardware_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CorrelatedExplorationConfig(**kwargs)


def test_restore_failure_preserves_existing_rng():
    config = CorrelatedExplorationConfig(seed=2)
    a, b = CorrelatedGaussianExploration(2, config), CorrelatedGaussianExploration(2, config)
    invalid = a.snapshot()
    invalid["state"] = [np.nan, 0]
    with pytest.raises(ValueError):
        a.restore(invalid)
    np.testing.assert_array_equal(
        a.perturb(np.zeros(2), np.zeros(2)), b.perturb(np.zeros(2), np.zeros(2))
    )
