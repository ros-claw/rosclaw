"""Task-neutral causal GRU state, not an actuator or runtime policy.

The reset/update/new ordering and reset-after equation match torch.nn.GRU.
Numeric parameters are owned and read-only. Each episode needs its own state.
Outputs are unit-bounded; external gates/caps and physical proof remain caller
obligations. This module has no simulator, model I/O or authorization surface.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

KEYS = ("weight_ih", "weight_hh", "bias_ih", "bias_hh", "head_weight", "head_bias")


def initial_parameters(
    input_dimension: int, output_dimension: int, *, hidden_dimension: int = 64, seed: int = 0
) -> dict[str, Any]:
    if (
        type(input_dimension) is not int
        or not 1 <= input_dimension <= 512
        or type(output_dimension) is not int
        or not 1 <= output_dimension <= 64
        or type(hidden_dimension) is not int
        or not 1 <= hidden_dimension <= 128
        or type(seed) is not int
        or not 0 <= seed < 2**32
    ):
        raise ValueError("bounded integer memory dimensions and seed required")
    rng = np.random.default_rng(seed)
    values = {
        "weight_ih": rng.normal(size=(3 * hidden_dimension, input_dimension))
        / np.sqrt(input_dimension),
        "weight_hh": rng.normal(size=(3 * hidden_dimension, hidden_dimension))
        / np.sqrt(hidden_dimension),
        "bias_ih": np.zeros(3 * hidden_dimension),
        "bias_hh": np.zeros(3 * hidden_dimension),
        "head_weight": np.zeros((output_dimension, hidden_dimension)),
        "head_bias": np.zeros(output_dimension),
    }
    return {k: v.tolist() for k, v in values.items()}


class CausalResidualMemory:
    """Sequential numerical state only; never authorizes a motor command."""

    def __init__(self, parameters: Any) -> None:
        if type(parameters) is not dict or set(parameters) != set(KEYS):
            raise ValueError("complete causal residual memory parameters required")
        arrays = {k: np.asarray(parameters[k]) for k in KEYS}
        if any(v.dtype.kind not in "fiu" for v in arrays.values()):
            raise ValueError("finite numeric memory parameters required")
        owned = {k: np.array(v, dtype=np.float64, copy=True) for k, v in arrays.items()}
        recurrent, inputs, output = owned["weight_hh"], owned["weight_ih"], owned["head_weight"]
        if (
            recurrent.ndim != 2
            or not 1 <= recurrent.shape[1] <= 128
            or recurrent.shape[0] != 3 * recurrent.shape[1]
            or inputs.ndim != 2
            or inputs.shape[0] != recurrent.shape[0]
            or not 1 <= inputs.shape[1] <= 512
            or output.ndim != 2
            or output.shape[1] != recurrent.shape[1]
            or not 1 <= output.shape[0] <= 64
            or owned["bias_ih"].shape != (recurrent.shape[0],)
            or owned["bias_hh"].shape != (recurrent.shape[0],)
            or owned["head_bias"].shape != (output.shape[0],)
            or any(not np.isfinite(v).all() or np.max(np.abs(v)) > 1e6 for v in owned.values())
        ):
            raise ValueError("finite aligned bounded memory parameters required")
        for value in owned.values():
            value.flags.writeable = False
        self._parameters = owned
        self.input_dimension = inputs.shape[1]
        self.output_dimension = output.shape[0]
        self.hidden_dimension = recurrent.shape[1]
        self._state = np.zeros(self.hidden_dimension)
        self._next_index = 0

    @property
    def state(self) -> np.ndarray[Any, Any]:
        return self._state.copy()

    @property
    def next_index(self) -> int:
        return self._next_index

    def parameters(self) -> dict[str, Any]:
        return {k: v.tolist() for k, v in self._parameters.items()}

    def new_episode(self) -> CausalResidualMemory:
        return CausalResidualMemory(copy.deepcopy(self.parameters()))

    def step(self, context: Any, *, index: int) -> np.ndarray[Any, Any]:
        values = np.asarray(context)
        if (
            type(index) is not int
            or index != self._next_index
            or not 0 <= index < 4096
            or values.dtype.kind not in "fiu"
            or values.shape != (self.input_dimension,)
        ):
            raise ValueError("sequential finite causal memory context required")
        values = np.array(values, dtype=np.float64, copy=True)
        if not np.isfinite(values).all() or np.max(np.abs(values)) > 1e6:
            raise ValueError("sequential finite causal memory context required")
        p = self._parameters
        xi = np.split(p["weight_ih"] @ values + p["bias_ih"], 3)
        hh = np.split(p["weight_hh"] @ self._state + p["bias_hh"], 3)

        def sigmoid(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            exponential = np.exp(-np.abs(x))
            return np.where(x >= 0, 1 / (1 + exponential), exponential / (1 + exponential))

        reset, update = sigmoid(xi[0] + hh[0]), sigmoid(xi[1] + hh[1])
        candidate = np.tanh(xi[2] + reset * hh[2])
        state = np.clip((1 - update) * candidate + update * self._state, -1, 1)
        output = np.tanh(p["head_weight"] @ state + p["head_bias"])
        if not np.isfinite(state).all() or not np.isfinite(output).all():
            raise ValueError("nonfinite causal residual memory computation")
        self._state, self._next_index = state, index + 1
        return np.asarray(output, dtype=np.float64)
