"""Bounded new-RSK workbook views; full original tool results stay unchanged."""

import copy
import hashlib

from rosclaw.contracts.common import content_hash


def rsk_workbook_projection(result):
    if isinstance(result.get("proposal"), dict) and any(
        change.get("operation") == "REPLACE_WHOLE_ORIGINAL_FILE"
        for change in result["proposal"].get("changes", [])
    ):
        view = copy.deepcopy(result)
        for change in view["proposal"]["changes"]:
            if change.get("operation") != "REPLACE_WHOLE_ORIGINAL_FILE":
                continue
            for field in ["before", "value"]:
                text = change[field]
                change[field] = {
                    "original_utf8_bytes": len(text.encode()),
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "text_omitted_from_workbook": True,
                }
        view["full_original_result_hash"] = content_hash("rskfulltoolresult", result)
        return view
    if isinstance(result.get("experiment"), dict):
        experiment = result["experiment"]
        return {
            "status": result["status"],
            "experiment_hash": experiment["experiment_hash"],
            "graph_hash": experiment["graph_hash"],
            "plan_status": experiment["status"],
            "hypotheses": [
                {
                    "id": h["id"],
                    "predicate": h["predicate"],
                    "status": h["status"],
                    "targets": h["targets"][:8],
                    "targets_omitted": max(0, len(h["targets"]) - 8),
                    "supporting_evidence_count": len(h["supporting_evidence"]),
                    "contradicting_evidence_count": len(h["contradicting_evidence"]),
                }
                for h in experiment["hypotheses"]
            ],
            "probes": [
                {
                    "probe_id": p["request"]["id"],
                    "operation": p["request"]["operation"],
                    "provider": p["request"]["provider"],
                    "max_duration_sec": p["request"]["max_duration_sec"],
                    "discriminated_hypotheses": p["discriminated_hypotheses"],
                }
                for p in experiment["probes"]
            ],
            "deferred_hypothesis_ids": experiment["deferred_hypothesis_ids"],
            "budget_sec": experiment["budget_sec"],
            "starts_World": False,
            "execution_status": "PROPOSED_NOT_EXECUTED",
            "current_authority": "NONE",
            "full_original_result_hash": content_hash("rskfulltoolresult", result),
        }
    if isinstance(result.get("assessment"), dict):
        assessment = result["assessment"]
        return {
            "status": result["status"],
            "diagnostic_assessment_status": assessment["status"],
            "actual_new_physics_Worlds": result.get("actual_new_physics_Worlds"),
            "provider_scope": result.get("provider_scope"),
            "reasons": assessment["reasons"],
            "withdrawn_hypothesis_ids": assessment["withdrawn_hypothesis_ids"],
            "hypotheses": [
                {
                    "id": h["id"],
                    "predicate": h["predicate"],
                    "status": h["status"],
                    "supporting_evidence_count": len(h["supporting_evidence"]),
                    "contradicting_evidence_count": len(h["contradicting_evidence"]),
                }
                for h in assessment["hypotheses"]
            ],
            "original_case_rebound": False,
            "current_case_L1": "NOT_EVALUATED",
            "causal_verification": "NOT_ESTABLISHED",
            "current_authority": "NONE",
            "full_original_result_hash": content_hash("rskfulltoolresult", result),
        }
    return result
