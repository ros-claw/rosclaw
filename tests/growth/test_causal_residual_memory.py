import copy

import numpy as np
import pytest

from rosclaw.growth.causal_residual_memory import CausalResidualMemory, initial_parameters


def test_zero_head_is_exact_and_independent_owned_state():
    parameters = initial_parameters(5, 3, hidden_dimension=4, seed=9)
    a = CausalResidualMemory(parameters)
    b = a.new_episode()
    parameters["weight_ih"][0][0] += 1
    for index, context in enumerate(np.random.default_rng(3).normal(size=(20, 5))):
        np.testing.assert_array_equal(a.step(context, index=index), np.zeros(3))
    assert a.next_index == 20 and b.next_index == 0
    np.testing.assert_array_equal(b.state, np.zeros(4))
    assert not np.array_equal(a.state, b.state)
    state = a.state
    state[:] = 0
    assert not np.array_equal(a.state, state)
    assert a.parameters() == b.parameters()
    assert a.parameters() != parameters
    with pytest.raises(ValueError):
        a._parameters["head_weight"][0, 0] = 1


def test_numpy_memory_matches_actual_torch_gru_cell():
    torch = pytest.importorskip("torch")
    parameters = initial_parameters(5, 3, hidden_dimension=4, seed=9)
    parameters["head_weight"] = np.random.default_rng(11).normal(size=(3, 4)).tolist()
    memory = CausalResidualMemory(parameters)
    cell = torch.nn.GRUCell(5, 4, dtype=torch.float64)
    with torch.no_grad():
        for key in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            getattr(cell, key).copy_(torch.tensor(parameters[key], dtype=torch.float64))
        hidden = torch.zeros((1, 4), dtype=torch.float64)
        for index, context in enumerate(np.random.default_rng(19).normal(size=(30, 5))):
            hidden = torch.clamp(
                cell(torch.tensor(context[None], dtype=torch.float64), hidden), -1, 1
            )
            output = torch.tanh(
                hidden @ torch.tensor(parameters["head_weight"], dtype=torch.float64).T
                + torch.tensor(parameters["head_bias"], dtype=torch.float64)
            )
            np.testing.assert_allclose(
                memory.step(context, index=index), output.numpy()[0], atol=1e-14, rtol=0
            )
            np.testing.assert_allclose(memory.state, hidden.numpy()[0], atol=1e-14, rtol=0)


@pytest.mark.parametrize(
    "context,index",
    [
        (np.ones(5), 1),
        (np.ones(5), True),
        (np.ones(4), 0),
        (np.full(5, np.nan), 0),
        (np.full(5, 1e7), 0),
        (np.ones(5, dtype=bool), 0),
    ],
)
def test_invalid_inputs_do_not_advance_state(context, index):
    memory = CausalResidualMemory(initial_parameters(5, 3, hidden_dimension=4))
    with pytest.raises(ValueError):
        memory.step(context, index=index)
    assert memory.next_index == 0
    np.testing.assert_array_equal(memory.state, np.zeros(4))


@pytest.mark.parametrize(
    "key,value",
    [
        ("weight_hh", [[1]]),
        ("head_bias", [True, False, True]),
        ("head_weight", [[float("inf")] * 4] * 3),
        ("bias_ih", [1e7] * 12),
    ],
)
def test_invalid_parameters_rejected(key, value):
    parameters = initial_parameters(5, 3, hidden_dimension=4)
    changed = copy.deepcopy(parameters)
    changed[key] = value
    with pytest.raises(ValueError):
        CausalResidualMemory(changed)


def test_saturated_inputs_remain_bounded_without_overflow():
    parameters = initial_parameters(5, 3, hidden_dimension=4)
    parameters["weight_ih"] = np.full((12, 5), 1e6).tolist()
    parameters["head_weight"] = np.full((3, 4), 1e6).tolist()
    memory = CausalResidualMemory(parameters)
    with np.errstate(over="raise", invalid="raise"):
        output = memory.step(np.full(5, 1e6), index=0)
    assert np.isfinite(output).all() and np.max(np.abs(output)) <= 1
    assert np.max(np.abs(memory.state)) <= 1
