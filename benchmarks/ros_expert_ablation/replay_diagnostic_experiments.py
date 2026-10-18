"""Offline RK-02 proposals from SHA-pinned operational graphs; no probe execution.

Run: python -m benchmarks.ros_expert_ablation.replay_diagnostic_experiments MANIFEST OUTPUT
Each entry has a graph {path, sha256} and explicit bounded issue_ids. Review labels
are never passed to the hypothesis engine. Output directories are exclusive.
"""

import argparse
import hashlib
import json
from pathlib import Path

from rosclaw.connectors.ros.engineering.probe_planner import plan_diagnostic_probes
from rosclaw.connectors.ros.intelligence.graph_contracts import RosOperationalGraphV1

from .replay_operational_graph import _read


def replay(manifest, output):
    manifest = Path(manifest)
    if (
        manifest.is_symlink()
        or not manifest.is_file()
        or not 0 < manifest.stat().st_size <= 5_000_000
    ):
        raise ValueError("bounded regular manifest required")
    original = manifest.read_bytes()
    document = json.loads(original)
    if document.get("schema_version") != "rosclaw.diagnostic_replay_inputs.v1":
        raise ValueError("unsupported diagnostic replay manifest")
    entries = document.get("entries")
    if type(entries) is not list or not 1 <= len(entries) <= 128:
        raise ValueError("1..128 explicit replay entries required")
    # Validate all original bindings before creating output.
    sources = []
    for entry in entries:
        graph = RosOperationalGraphV1.from_dict(json.loads(_read(entry["graph"])))
        plan = plan_diagnostic_probes(
            graph,
            issue_ids=entry["issue_ids"],
            budget_sec=entry.get("budget_sec", 20),
            max_probes=entry.get("max_probes", 8),
        )
        sources.append((entry, graph, plan))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for ordinal, (entry, graph, plan) in enumerate(sources, 1):
        name = f"{ordinal:03d}-experiment.json"
        raw = (json.dumps(plan.model_dump(mode="json"), sort_keys=True, indent=2) + "\n").encode()
        (output / name).write_bytes(raw)
        rows.append(
            {
                "ordinal": ordinal,
                "graph_hash": graph.graph_hash,
                "input_sha256": entry["graph"]["sha256"],
                "output": name,
                "output_sha256": hashlib.sha256(raw).hexdigest(),
                "status": plan.status,
                "hypotheses": len(plan.hypotheses),
                "probes_proposed": len(plan.probes),
                "probe_execution": "NOT_RUN",
                "execution_authority": "NONE",
            }
        )
    for entry, _, _ in sources:
        _read(entry["graph"])
    if manifest.read_bytes() != original:
        raise ValueError("manifest changed during replay")
    summary = {
        "schema_version": "rosclaw.diagnostic_replay_review.v1",
        "manifest_sha256": hashlib.sha256(original).hexdigest(),
        "entries": rows,
        "scope": "OFFLINE_OBSERVATIONAL_PROPOSALS_NOT_NEW_NATIVE_OR_PHYSICAL_ACCEPTANCE",
    }
    (output / "review.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    args = parser.parse_args()
    result = replay(args.manifest, args.output)
    print(json.dumps({"entries": len(result["entries"]), "probe_execution": "NOT_RUN"}))


if __name__ == "__main__":
    main()
