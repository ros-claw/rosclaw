"""Bounded historical feedback from an owning non-REAL canonical receipt.

Rendering changes neither execution, verification, admission nor retry policy.
The complete receipt remains in rosclawd under its original action identifier.
"""

from __future__ import annotations

import json
from typing import Any

MAX_FEEDBACK_BYTES = 3072


def historical_action_feedback(
    receipt: dict[str, Any],
    *,
    capability: str,
    action_id: str,
    mode: str,
    max_bytes: int = MAX_FEEDBACK_BYTES,
) -> str:
    """Expose actual recorded facts, never a fabricated current robot state.

    Unsupported/REAL or mismatched receipts retain the existing dispatch text.
    Oversized fields are explicitly omitted, not silently chopped or promoted.
    No raw authorization, grant, permit or lease payload is projected.
    """
    if (
        mode not in {"SIMULATION", "SHADOW"}
        or receipt.get("execution_mode") != mode
        or receipt.get("evidence_domain") not in {"SIMULATION", "SHADOW", "UNKNOWN"}
        or not action_id
        or receipt.get("action_id") != action_id
        or receipt.get("capability_id") != capability
    ):
        return ""

    max_bytes = min(MAX_FEEDBACK_BYTES, max_bytes)
    if max_bytes < 256:
        return ""
    payload_budget = max_bytes - 256

    def encoded(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    fields = (
        "action_id",
        "body_id",
        "body_snapshot_hash",
        "capability_id",
        "execution_mode",
        "evidence_domain",
        "final_state",
        "evidence_level",
        "trust_level",
        "verified",
        "finished_at",
    )
    result = {key: receipt[key] for key in fields if key in receipt}
    result.update(
        feedback_scope="historical_original_nonreal_action_receipt",
        current_state_asserted=False,
        evidence_promoted=False,
        receipt_ref="receipt://" + action_id,
    )
    try:
        if len(encoded(result).encode()) > min(1024, payload_budget):
            return "\nHistorical receipt feedback unavailable: oversized identity metadata."
    except (TypeError, ValueError):
        return "\nHistorical receipt feedback unavailable: non-JSON identity metadata."
    omitted: list[str] = []

    def include(key: str, value: Any, limit: int) -> bool:
        try:
            part = encoded(value)
            proposed = {**result, key: value}
            if len(part.encode()) > limit or len(encoded(proposed).encode()) > payload_budget:
                omitted.append(key)
                return False
        except (TypeError, ValueError):
            omitted.append(key)
            return False
        result[key] = value
        return True

    if "errors" in receipt:
        include("errors", receipt["errors"], 1024)
    if "observations" in receipt:
        observations = receipt["observations"]
        if isinstance(observations, list):
            result["recorded_observation_count"] = len(observations)
            selected = observations[-2:]
            included = include("last_recorded_observations", selected, 1536)
            result["omitted_observation_count"] = len(observations) - (
                len(selected) if included else 0
            )
        else:
            omitted.append("observations")
    for key in ("dispatch_result", "verification_result"):
        if key in receipt:
            include(key, receipt[key], 1024)
    result["inline_omitted_fields"] = omitted
    rendered = encoded(result)
    # Reserve space above for explicit omission metadata. Fail rendering alone
    # if an unexpected value exceeds the bound; never amend the action result.
    if len(rendered.encode()) > max_bytes:
        return "\nHistorical receipt feedback unavailable: bounded projection exceeded."
    text = "\n原回执历史反馈（非当前状态，不增加执行权限或提高证据等级）：\n" + rendered
    if len(text.encode()) > max_bytes:
        return "\nHistorical receipt feedback unavailable: bounded projection exceeded."
    return text
