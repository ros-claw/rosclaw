"""Offline RK-01 replay: original files -> graphs/plans, without ROS or model calls.

Run: python -m benchmarks.ros_expert_ablation.replay_operational_graph MANIFEST OUTPUT
The manifest lists explicit SHA256-pinned original system/native, URDF and
requirements inputs. Native replay uses the existing pure system-model builder.
Labels and expected verdicts are used only for review after compilation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot
from rosclaw.connectors.ros.engineering import compile_engineering_case
from rosclaw.connectors.ros.intelligence import RosSystemModel, build_system_model
from rosclaw.connectors.ros.intelligence.graph_contracts import RosOperationalGraphV1
from rosclaw.connectors.ros.intelligence.missing_evidence import describe_missing_evidence
from rosclaw.connectors.ros.intelligence.operational_graph import (
    compile_operational_graph,
    summarize_operational_graph,
)
from rosclaw.connectors.ros.intelligence.provenance_edges import pointer_value
from rosclaw.contracts.common import content_hash


def _read(ref):
    path = Path(ref["path"])
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 5_000_000:
        raise ValueError("bounded regular original file required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("original input SHA256 mismatch")
    return raw


def replay(manifest, output):
    path = Path(manifest)
    if path.is_symlink() or not 0 < path.stat().st_size <= 5_000_000:
        raise ValueError("bounded manifest required")
    original_manifest = path.read_bytes()
    inputs = json.loads(original_manifest)
    if inputs.get("schema_version") != "rosclaw.operational_graph_replay_inputs.v1":
        raise ValueError("unsupported graph replay manifest")
    entries = inputs.get("entries")
    if type(entries) is not list or not 1 <= len(entries) <= 128:
        raise ValueError("1..128 frozen entries required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for ordinal, entry in enumerate(entries, 1):
        original = _read(entry["observation"])
        observed = json.loads(original)
        if entry["format"] == "sealed_system_model":
            model = RosSystemModel.from_dict(observed)
        elif entry["format"] == "native_probe":
            graph = RosGraphSnapshot.from_dict(
                {
                    "ros_version": "ros2",
                    "distro": "jazzy",
                    "endpoint": "fixture://sealed-original-native-replay",
                    "captured_at": observed["captured_at"],
                    **observed["graph"],
                }
            )
            model = build_system_model(graph, native=observed)
        else:
            raise ValueError("unsupported original observation format")
        urdf = _read(entry["urdf"])
        requirements = json.loads(_read(entry["requirements"]))
        normalized_ref = (
            content_hash(
                "nativeprojection",
                {
                    "original_native_sha256": entry["observation"]["sha256"],
                    "snapshot_hash": model.snapshot_hash,
                },
            )
            if entry["format"] == "native_probe"
            else entry["observation"]["path"]
        )
        kwargs = {
            "source_ref": normalized_ref,
            "now": datetime.fromisoformat(entry["reviewed_at"]),
            "evidence_domain": "SEALED_REPLAY",
        }
        if entry.get("source_review"):
            kwargs["source_review"] = json.loads(_read(entry["source_review"]))
        for key in ("expected_run_id", "expected_body_hash"):
            if key in entry:
                kwargs[key] = entry[key]
        graph = compile_operational_graph(model, urdf, requirements, **kwargs)
        case = compile_engineering_case(model, urdf, requirements, **kwargs)
        expected = {c.id: c.status for c in case.constraints}
        actual = {f.id: f.status for f in graph.findings if f.id in expected}
        if actual != expected:
            raise AssertionError("existing Case invariant regression")
        if any(actual.get(k) != v for k, v in entry.get("expected_constraints", {}).items()):
            raise AssertionError("frozen historical verdict changed")
        # Independently resolve every graph evidence pointer against originals/Case.
        case_document, model_document = case.model_dump(mode="json"), model.to_dict()
        for ref in graph.provenance.values():
            document = case_document if ref.document_hash == case.case_hash else model_document
            if ref.document_hash not in {
                case.case_hash,
                model.snapshot_hash,
            } or ref.value_hash != content_hash("opvalue", pointer_value(document, ref.pointer)):
                raise AssertionError("original provenance pointer/hash mismatch")
        graph = RosOperationalGraphV1.from_dict(graph.model_dump(mode="json"))
        summary = summarize_operational_graph(graph)
        if len(json.dumps(summary, ensure_ascii=False, separators=(",", ":"))) > 6000:
            raise AssertionError("Agent projection budget exceeded")
        plan = describe_missing_evidence(graph)
        if sum(p.max_duration_sec for p in plan.requests) != plan.total_budget_sec:
            raise AssertionError("read-only plan budget mismatch")
        if entry["format"] == "native_probe":
            (output / f"{ordinal:03d}-normalized-system-model.json").write_text(
                json.dumps(model.to_dict(), sort_keys=True, indent=2) + "\n"
            )
        (output / f"{ordinal:03d}-case.json").write_text(
            json.dumps(case_document, sort_keys=True, indent=2) + "\n"
        )
        for suffix, data in (
            ("graph", graph.model_dump(mode="json")),
            ("plan", plan.model_dump(mode="json")),
            ("summary", summary),
        ):
            (output / f"{ordinal:03d}-{suffix}.json").write_text(
                json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
            )
        rows.append(
            {
                "ordinal": ordinal,
                "label_for_review_only": entry.get("label_for_review_only"),
                "source_ref": graph.source_ref,
                "original_observation_sha256": entry["observation"]["sha256"],
                "normalized_model_artifact": f"{ordinal:03d}-normalized-system-model.json"
                if entry["format"] == "native_probe"
                else None,
                "snapshot_hash": graph.snapshot_hash,
                "graph_hash": graph.graph_hash,
                "case_hash": graph.case_hash,
                "original_invariants": actual,
                "provenance_pointers_checked": len(graph.provenance),
                "entities": len(graph.entities),
                "dataflows": len(graph.dataflows),
                "findings": len(graph.findings),
                "summary_characters": len(
                    json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
                ),
                "probe_status": plan.status,
                "probes_proposed_not_executed": len(plan.requests),
                "new_finding_counts": {
                    s: sum(f.status == s for f in graph.findings if f.id not in expected)
                    for s in ("SATISFIED", "VIOLATED", "UNKNOWN")
                },
            }
        )
        for key in ("observation", "urdf", "requirements", "source_review"):
            if entry.get(key):
                _read(entry[key])
    if path.read_bytes() != original_manifest:
        raise AssertionError("input manifest changed during replay")
    result = {
        "status": "PASS_ORIGINAL_OFFLINE_GRAPH_REPLAY_NOT_NEW_PHYSICAL_ACCEPTANCE",
        "input_manifest_sha256": hashlib.sha256(original_manifest).hexdigest(),
        "entries": len(rows),
        "new_Worlds": 0,
        "new_model_calls": 0,
        "probes_executed": 0,
        "current_authority": "NONE",
        "rows": rows,
    }
    (output / "original-review.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    args = parser.parse_args()
    result = replay(args.manifest, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
