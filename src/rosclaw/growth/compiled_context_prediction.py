"""Private, source-pinned numerical inference for prediction-only models.

The unchanged reference validates model semantics once. Every query remains
bounded, copied, source-checked and finite. No policy, simulator, transport,
training, execution or authority interface is exposed.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.growth import context_prediction_mlp as reference


class CompiledContextPrediction:
    def __init__(self, model: dict[str, Any]) -> None:
        owned = copy.deepcopy(model)
        try:
            width = len(owned["input_mean"])
        except (KeyError, TypeError) as exc:
            raise ValueError("complete prediction-only model required") from exc
        if not 1 <= width <= 8192:
            raise ValueError("bounded prediction input dimensions required")
        # The ORIGINAL full reference owns source, authority, hash, shape and
        # normalization validation. Compiled storage never trusts object ids.
        reference.predict(owned, np.zeros((1, width)))
        self._model_hash = owned["model_hash"]
        self._input_mean = np.asarray(owned["input_mean"], dtype=np.float64)
        self._input_scale = np.asarray(owned["input_scale"], dtype=np.float64)
        self._target_mean = np.asarray(owned["target_mean"], dtype=np.float64)
        self._target_scale = np.asarray(owned["target_scale"], dtype=np.float64)
        self._layers = tuple(
            (np.asarray(v["weight"], dtype=np.float32), np.asarray(v["bias"], dtype=np.float32))
            for v in owned["layers"]
        )
        for value in (
            self._input_mean,
            self._input_scale,
            self._target_mean,
            self._target_scale,
            *(v for layer in self._layers for v in layer),
        ):
            value.flags.writeable = False
        self._pins = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(reference.__file__))
        }

    def predict(self, features: Any) -> np.ndarray[Any, Any]:
        if any(
            hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in self._pins.items()
        ):
            raise ValueError("compiled prediction dependency source changed")
        x = reference._matrix(features)
        if x.shape[1] != len(self._input_mean):
            raise ValueError("aligned compiled prediction features required")
        hidden = ((x - self._input_mean) / self._input_scale).astype(np.float32)
        if not np.isfinite(hidden).all() or np.max(np.abs(hidden)) > 1e6:
            raise ValueError("prediction features exceed normalized numeric envelope")
        for i, (weight, bias) in enumerate(self._layers):
            hidden = hidden @ weight.T + bias
            if not np.isfinite(hidden).all():
                raise ValueError("nonfinite intermediate neural prediction")
            if i < 2:
                hidden = np.tanh(hidden)
        result: np.ndarray[Any, Any] = (
            hidden.astype(np.float64) * self._target_scale + self._target_mean
        )
        if not np.isfinite(result).all():
            raise ValueError("nonfinite neural prediction")
        return result

    def contract(self) -> dict[str, Any]:
        return {
            "schema": "rosclaw.growth.compiled_context_prediction.v1",
            "model_hash": self._model_hash,
            "source_pins": dict(self._pins),
            "original_model_semantics_validated": True,
            "private_readonly_parameters": True,
            "bounded_features_checked_each_query": True,
            "dependency_sources_checked_each_query": True,
            "prediction_only": True,
            "activation_ceiling": "SIM_ONLY",
            "parameters_changed": False,
            "motor_policy": False,
            "promotion_authorized": False,
            "hardware_authorized": False,
        }
