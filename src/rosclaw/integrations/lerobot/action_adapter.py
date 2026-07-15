"""Convert worker action output into a ROSClaw action proposal.

This module must not import torch or lerobot.
"""

from __future__ import annotations

from typing import Any

from rosclaw.integrations.lerobot.contracts import (
    ActionProposalV2,
    build_action_proposal_v2,
    validate_action_values,
)
from rosclaw.integrations.lerobot.worker_schema import WorkerAction


def adapt_action_to_proposal(
    action: WorkerAction | dict[str, Any] | None,
    *,
    policy_path: str = "",
    policy_metadata: dict[str, Any] | None = None,
    manifest_action_space: list[str] | None = None,
    session_id: str | None = None,
    step_index: int = 0,
    proposal_id: str = "proposal_000",
    runtime_id: str | None = None,
    processor_hash: str | None = None,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn a worker action into a ROSClaw action proposal dict.

    The returned proposal follows ``rosclaw.action_proposal.v2`` and always has
    ``executable=false`` and ``requires_sandbox=true`` as part of the P1/P4
    safety contract.

    Action chunks (e.g. shape ``[100, 14]``) are kept in their original shape
    rather than flattened, so downstream sandbox/body mapping can decide how to
    interpret the chunk.
    """
    if action is None:
        return ActionProposalV2(
            proposal_id=proposal_id,
            session_id=session_id,
            step_index=step_index,
            policy_path=policy_path,
            policy_hash=None,
            runtime_id=runtime_id,
            processor_hash=processor_hash,
            representation="unknown",
            reference_frame=None,
            values=[],
            shape=[],
            dtype="float32",
            names=[],
            units="unknown",
            chunk={"is_chunk": False, "length": 1},
            timing=timing or {},
            safety={
                "executable": False,
                "requires_sandbox": True,
                "not_executed": True,
                "body_mapping_required": True,
                "body_compatible": False,
                "body_name": None,
            },
        ).to_dict()

    if isinstance(action, dict):
        processed_action = dict(action)
        raw_action = action.get("raw_action")
    else:
        processed_action = action.to_dict()
        raw_action = None

    if "raw_action" in processed_action:
        raw_action = processed_action.pop("raw_action")

    # Fail-closed on NaN/Inf.
    values = processed_action.get("values", [])
    flat_values: list[float] = []
    if isinstance(values, list) and values and isinstance(values[0], list):
        flat_values = [float(v) for row in values for v in row]
    else:
        flat_values = [float(v) for v in values]
    ok, error = validate_action_values(flat_values)
    if not ok:
        return _blocked_proposal(
            proposal_id=proposal_id,
            session_id=session_id,
            step_index=step_index,
            policy_path=policy_path,
            runtime_id=runtime_id,
            processor_hash=processor_hash,
            timing=timing,
            error_code="invalid_policy_action",
            message=error or "invalid_policy_action",
        )

    proposal = build_action_proposal_v2(
        proposal_id=proposal_id,
        session_id=session_id,
        step_index=step_index,
        policy_path=policy_path,
        policy_metadata=policy_metadata or {},
        manifest_action_space=manifest_action_space,
        processed_action=processed_action,
        raw_action=raw_action if isinstance(raw_action, dict) else None,
        runtime_id=runtime_id,
        processor_hash=processor_hash,
        timing=timing,
        safety={
            "executable": False,
            "requires_sandbox": True,
            "not_executed": True,
            "body_mapping_required": True,
            "body_compatible": False,
            "body_name": None,
        },
    )
    return proposal.to_dict()


def _blocked_proposal(
    *,
    proposal_id: str,
    session_id: str | None,
    step_index: int,
    policy_path: str,
    runtime_id: str | None,
    processor_hash: str | None,
    timing: dict[str, Any] | None,
    error_code: str,
    message: str,
) -> dict[str, Any]:
    proposal = ActionProposalV2(
        proposal_id=proposal_id,
        session_id=session_id,
        step_index=step_index,
        policy_path=policy_path,
        policy_hash=None,
        runtime_id=runtime_id,
        processor_hash=processor_hash,
        representation="unknown",
        reference_frame=None,
        values=[],
        shape=[],
        dtype="float32",
        names=[],
        units="unknown",
        chunk={"is_chunk": False, "length": 1},
        timing=timing or {},
        safety={
            "executable": False,
            "requires_sandbox": True,
            "not_executed": True,
            "body_mapping_required": True,
            "body_compatible": False,
            "body_name": None,
            "error_code": error_code,
            "message": message,
        },
    )
    return proposal.to_dict()


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
