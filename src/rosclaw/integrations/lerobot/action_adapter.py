"""Convert worker action output into a ROSClaw action proposal.

This module must not import torch or lerobot.
"""

from __future__ import annotations

from typing import Any

from rosclaw.integrations.lerobot.worker_schema import WorkerAction


def adapt_action_to_proposal(action: WorkerAction | dict[str, Any] | None) -> dict[str, Any]:
    """Turn a worker action into a ROSClaw action proposal dict.

    The returned proposal always has ``executable=false`` and
    ``requires_sandbox=true`` as part of the P1 safety contract.

    Action chunks (e.g. shape ``[100, 14]``) are kept in their original shape
    rather than flattened, so downstream sandbox/body mapping can decide how to
    interpret the chunk.
    """
    if action is None:
        return {
            "type": "none",
            "values": [],
            "shape": [],
            "dtype": "float32",
            "executable": False,
            "requires_sandbox": True,
            "not_executed": True,
            "body_mapping_required": True,
            "body_compatible": False,
            "body_name": None,
        }

    if isinstance(action, dict):
        shape = list(action.get("shape", []))
        values = action.get("values", [])
        action_type = action.get("type", _infer_action_type(shape))
        dtype = action.get("dtype", "float32")
    else:
        shape = list(action.shape)
        values = action.values
        action_type = action.type
        dtype = action.dtype

    proposal: dict[str, Any] = {
        "type": action_type,
        "values": _as_serializable_values(values, shape),
        "shape": shape,
        "dtype": dtype,
        "executable": False,
        "requires_sandbox": True,
        "not_executed": True,
        "body_mapping_required": True,
        "body_compatible": False,
        "body_name": None,
    }

    # Preserve chunk metadata when the policy emits action chunks.
    if len(shape) == 2:
        proposal["chunk_size"] = shape[0]
        proposal["action_dim"] = shape[1]

    return proposal


def _infer_action_type(shape: list[int]) -> str:
    """Classify an action as a raw vector or a chunked action tensor."""
    if len(shape) == 2 and shape[0] > 1:
        return "lerobot_action_chunk"
    return "raw_lerobot_action"


def _as_serializable_values(values: Any, shape: list[int]) -> Any:
    """Return a JSON-serializable representation, preserving chunk shape."""
    if values is None:
        return []

    # If the values already carry structure (list-of-lists), keep it when the
    # declared shape is multi-dimensional.
    if len(shape) >= 2 and isinstance(values, list) and values and isinstance(values[0], list):
        return _nested_floats(values)

    # Flatten only for plain vectors or scalar actions.
    return [float(v) for v in _flatten(values)]


def _nested_floats(values: Any) -> Any:
    """Recursively coerce a nested list/ndarray structure to plain floats."""
    if isinstance(values, (list, tuple)):
        return [_nested_floats(v) for v in values]
    if isinstance(values, dict):
        return {k: _nested_floats(v) for k, v in values.items()}
    return float(values)


def _flatten(values: Any) -> list[Any]:
    flat: list[Any] = []
    for v in values:
        if isinstance(v, (list, tuple)):
            flat.extend(_flatten(v))
        else:
            flat.append(v)
    return flat
