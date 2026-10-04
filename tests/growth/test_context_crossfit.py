import numpy as np
import pytest

from rosclaw.growth.context_crossfit import context_crossfit_advantages


def batch():
    groups = np.repeat(np.arange(16), 60)
    contexts = np.repeat(np.arange(8), 2)
    phases = np.tile(np.repeat(np.arange(3), 20), 16)
    features = np.column_stack((np.ones(len(groups)), contexts[groups] / 8))
    reward = np.repeat(np.arange(16, dtype=float) % 5, 60)
    return features, phases, groups, reward, contexts


def fit(data):
    x, p, g, r, c = data
    return context_crossfit_advantages(x, p, g, r, trajectory_context_ids=c)


def test_repeated_contexts_never_cross_folds():
    data = batch()
    result = fit(data)
    for context in np.unique(data[-1]):
        assert len(np.unique(result["trajectory_fold_ids"][data[-1] == context])) == 1
    assert result["crossfit_unit"] == "whole_context"
    assert np.isfinite(result["advantages"]).all()
    assert abs(result["advantages"].mean()) < 1e-12
    assert abs(result["advantages"].std() - 1) < 1e-12


def test_held_out_returns_do_not_affect_held_out_raw_predictions():
    data = list(batch())
    first = fit(data)
    mask = first["trajectory_fold_ids"][data[2]] == 0
    data[3] = data[3].copy()
    data[3][mask] += 100
    second = fit(data)
    np.testing.assert_array_equal(
        first["crossfit_predictions"][mask], second["crossfit_predictions"][mask]
    )
    x, p, _, r, _ = data
    for phase in range(3):
        train = (p == phase) & ~mask
        test = (p == phase) & mask
        weights = np.linalg.solve(x[train].T @ x[train] + 0.01 * np.eye(2), x[train].T @ r[train])
        np.testing.assert_array_equal(second["crossfit_predictions"][test], x[test] @ weights)


@pytest.mark.parametrize("field", range(5))
def test_rejects_misaligned_arrays(field):
    data = list(batch())
    data[field] = data[field][:-1]
    with pytest.raises(ValueError):
        fit(data)


def test_rejects_frame_varying_terminal_reward():
    data = list(batch())
    data[3] = data[3].copy()
    data[3][0] += 1
    with pytest.raises(ValueError, match="terminal return"):
        fit(data)


def test_rejects_insufficient_contexts_and_nonfinite_features():
    data = list(batch())
    data[-1] = np.zeros(16, dtype=int)
    with pytest.raises(ValueError):
        fit(data)
    data = list(batch())
    data[0][0, 0] = np.nan
    with pytest.raises(ValueError):
        fit(data)


def test_context_label_renaming_preserves_predictions():
    data = list(batch())
    first = fit(data)
    data[-1] = data[-1] * 100 + 7
    second = fit(data)
    np.testing.assert_array_equal(first["advantages"], second["advantages"])


def test_rejects_missing_phase_support_in_a_fold():
    data = list(batch())
    data[1] = data[1].copy()
    data[1][data[-1][data[2]] % 4 == 0] = 0
    with pytest.raises(ValueError, match="every phase"):
        fit(data)


@pytest.mark.parametrize("field", [0, 3])
def test_finite_inputs_with_derived_overflow_rejected(field):
    data = list(batch())
    data[field] = np.full_like(data[field], 1e308)
    with pytest.raises(ValueError, match="finite"):
        fit(data)


def test_integer_features_do_not_silently_overflow_covariance():
    data = list(batch())
    data[0] = (data[0] * 10000000000).astype(np.int64)
    integer = fit(data)
    data[0] = data[0].astype(np.float64)
    floating = fit(data)
    np.testing.assert_array_equal(integer["crossfit_predictions"], floating["crossfit_predictions"])


def test_singular_numeric_solver_rejected(monkeypatch):
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("fixture failure")

    monkeypatch.setattr(np.linalg, "solve", fail)
    with pytest.raises(ValueError, match="solvable"):
        fit(batch())
