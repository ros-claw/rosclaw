"""Lossless current-parent consolidation without dropping distinct experiences.

Exact repeated states reuse an existing memory row only when predictions are
also exactly equal. Every input sample retains an explicit row mapping. This
numeric training helper neither executes a policy nor verifies its evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rosclaw.growth.anchor_output_memory import AnchorOutputMemory


def consolidate_unique(
    memory: AnchorOutputMemory,
    observations: Any,
    predictions: Any,
    *,
    parent_policy_hash: str,
    evidence_hash: str,
) -> tuple[AnchorOutputMemory, dict[str, Any]]:
    """Return new memory and complete sample-to-row provenance; never mutate.

    The caller must independently qualify the parent, audit every trajectory,
    and seal this mapping into its evidence. A valid hash is not physical proof.
    Known-state guarantees remain local, not distributional or safety claims.
    """
    inherited = memory.to_dict()
    addition = AnchorOutputMemory(
        observations,
        predictions,
        bandwidth=inherited["bandwidth"],
        encoder_hash=memory.encoder_hash,
        parent_policy_hash=parent_policy_hash,
        evidence_hash=evidence_hash,
    ).to_dict()
    if len(addition["observations"][0]) != len(inherited["observations"][0]):
        raise ValueError("unchanged frozen observation dimension required")
    if len(addition["predictions"][0]) != memory.output_dimension:
        raise ValueError("unchanged parent prediction dimension required")
    old_x = np.asarray(inherited["observations"], dtype=np.float64)
    old_y = np.asarray(inherited["predictions"], dtype=np.float64)
    new_x = np.asarray(addition["observations"], dtype=np.float64)
    new_y = np.asarray(addition["predictions"], dtype=np.float64)
    old_x[old_x == 0] = 0.0
    new_x[new_x == 0] = 0.0
    seen = {row.tobytes(): i for i, row in enumerate(old_x)}
    appended_x, appended_y, mapping = [], [], []
    inherited_count = len(old_x)
    for x, y in zip(new_x, new_y, strict=True):
        key = x.tobytes()
        if key in seen:
            index = seen[key]
            expected = (
                old_y[index] if index < inherited_count else appended_y[index - inherited_count]
            )
            if not np.array_equal(y, expected):
                raise ValueError(
                    "conflicting exact-state parent predictions; no rounding or overwrite"
                )
        else:
            index = inherited_count + len(appended_x)
            seen[key] = index
            appended_x.append(x)
            appended_y.append(y)
        mapping.append(index)
    if appended_x:
        combined_x = np.concatenate((old_x, np.asarray(appended_x)))
        combined_y = np.concatenate((old_y, np.asarray(appended_y)))
    else:
        combined_x, combined_y = old_x, old_y
    # Constructor still enforces the total capacity; no distinct old row is
    # evicted, approximated, averaged, quantized or chosen by outcome ranking.
    result = AnchorOutputMemory(
        combined_x,
        combined_y,
        bandwidth=inherited["bandwidth"],
        encoder_hash=memory.encoder_hash,
        parent_policy_hash=parent_policy_hash,
        evidence_hash=evidence_hash,
        predecessor_memory_hash=inherited["memory_hash"],
    )
    provenance = {
        "schema": "rosclaw.growth.anchor_consolidation_mapping.v1",
        "inherited_memory_hash": inherited["memory_hash"],
        "consolidated_memory_hash": result.to_dict()["memory_hash"],
        "inherited_rows": inherited_count,
        "input_sample_count": len(new_x),
        "appended_unique_rows": len(appended_x),
        "exact_duplicate_samples": len(new_x) - len(appended_x),
        "sample_to_memory_row": mapping,
        "exact_duplicate_only": True,
        "rounding_used": False,
        "distinct_inherited_rows_evicted": 0,
        "evidence_independently_verified": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
    return result, provenance
