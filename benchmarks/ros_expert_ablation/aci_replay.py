"""Scripted constraint baseline on original snapshots; no model/World calls."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from benchmarks.ros_expert_ablation.aci_operator_context import aci_from_capture


def deterministic_request(aci):
    """Select an evidence-supported typed option, never inspect gold case labels."""
    states = {c.id: c.status for c in aci.case.constraints}
    if states["F-TF-001"] == "VIOLATED":
        intent = {"kind": "enable_localization_tf"}
    elif states["F-SCAN-001"] == "VIOLATED":
        intent = {"kind": "resume_scan_bridge"}
    elif states["F-LIFE-001"] == "VIOLATED":
        managers = [
            (node, params)
            for node, params in aci.model.observations["node_parameters"].items()
            if "node_names" in params
        ]
        if len(managers) != 1:
            raise ValueError("unique original manager needed")
        manager = managers[0]
        namespace = manager[0].rsplit("/", 1)[0]
        names = list(manager[1]["node_names"])
        resolved = {name if name.startswith("/") else namespace + "/" + name for name in names}
        names += [
            node.rsplit("/", 1)[1]
            for node in aci.requirements["lifecycle_nodes"].values()
            if node not in resolved
        ]
        intent = {"kind": "activate_managed_nodes", "node_names": names}
    elif states["F-GEOM-001"] == "VIOLATED":
        intent = {
            "kind": "cover_urdf_envelope",
            "radius_m": math.ceil(aci.tools.minimum_radius_m * 10) / 10,
        }
    elif states["F-ROUTE-001"] == "VIOLATED":
        if len(aci.ownership_reviews) != 1:
            raise ValueError("owned staged endpoint evidence needed")
        intent = {
            "kind": "disable_owned_fixture_endpoint",
            "ownership_ref": next(iter(aci.ownership_reviews)),
        }
    else:
        intent = None
    if intent:
        return {
            "operation": "ros_engineer.propose_repair",
            "arguments": {"case_hash": aci.case.case_hash, "repair": intent},
        }
    names = aci.requirements["interface_names"]
    return {
        "operation": "ros_engineer.propose_onboarding",
        "arguments": {
            "case_hash": aci.case.case_hash,
            "interfaces": dict(
                zip(
                    ["mapping", "odometry", "lidar", "navigate_to_pose"],
                    [
                        names[k]
                        for k in [
                            "mapping.occupancy_map",
                            "state.odometry",
                            "sensing.lidar",
                            "navigation.navigate_to_pose",
                        ]
                    ],
                    strict=True,
                )
            ),
        },
    }


def replay(study, worktree, output):
    study = Path(study).resolve()
    output = Path(output).resolve()
    if output == study or study in output.parents:
        raise ValueError("output must be outside originals")
    output.mkdir(exist_ok=False)
    rows = []
    for p in sorted(study.glob("episode-*/original-result.json")):
        original = json.loads(p.read_text())
        before = next(
            Path(w["directory"])
            for w in original["original_operator_Worlds"]
            if w["returncode"] == 0 and w["case"] != "positive"
        )
        aci = aci_from_capture(before, worktree=worktree)
        request = deterministic_request(aci)
        result = aci.call(request)
        rows.append(
            {
                "ordinal": original["ordinal"],
                "original_label_for_review_only": original["case"],
                "request": request,
                "result": result,
            }
        )
    assert len(rows) == 54
    result = {
        "scope": "SCRIPTED_OFFLINE_CONSTRAINT_BASELINE_NOT_MODEL_OR_PHYSICAL_EVALUATION",
        "new_model_calls": 0,
        "new_Worlds": 0,
        "status_counts": dict(Counter(row["result"]["status"] for row in rows)),
        "rows": rows,
    }
    (output / "original-static-review.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("study")
    p.add_argument("worktree")
    p.add_argument("output")
    a = p.parse_args()
    print(replay(a.study, a.worktree, a.output)["status_counts"])


if __name__ == "__main__":
    main()
