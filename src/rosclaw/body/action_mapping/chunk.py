"""Chunk handling helpers for body action mapping.

Policy action chunks are stored as flat vectors ordered
``[chunk_0_dim_0, ..., chunk_0_dim_N, chunk_1_dim_0, ...]``.
This module reshapes those vectors for per-chunk body mapping and validation.
"""

from __future__ import annotations

from typing import Any


def reshape_action_values(
    values: list[float],
    action_dim: int,
    chunk_size: int | None = None,
) -> list[list[float]]:
    """Reshape a flat action vector into ``[chunk][action_dim]`` rows."""
    chunks = chunk_size or 1
    expected = action_dim * chunks
    if len(values) != expected:
        raise ValueError(f"Expected {expected} values, got {len(values)}")
    return [values[i * action_dim : (i + 1) * action_dim] for i in range(chunks)]


def flatten_action_rows(rows: list[list[float]]) -> list[float]:
    """Flatten per-chunk rows back into a single list."""
    result: list[float] = []
    for row in rows:
        result.extend(row)
    return result


def chunk_from_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
    """Return normalized chunk metadata from a proposal dict."""
    chunk = proposal.get("chunk", {})
    if not isinstance(chunk, dict):
        return {"size": None, "is_chunked": False}
    return {
        "size": chunk.get("size"),
        "is_chunked": bool(chunk.get("size")),
        "metadata": chunk.get("metadata", {}),
    }
