"""Audit saved G5 turns without invoking models, SDKs or changing verdicts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def reference(path, root):
    raw = path.read_bytes()
    return {"path": str(path.relative_to(root)), "sha256": sha(raw), "bytes": len(raw)}


def shown_result(record):
    result = record["tool_result"]
    raw = json.dumps(result, ensure_ascii=False)
    if len(raw) > 16000:
        return {
            "status": "ORIGINAL_TOOL_RESULT_STORED_READ_WITH_READ_FILE",
            "file": f"tool-result-{record['turn']:02d}.json",
            "characters": len(raw),
        }
    return result


def audit(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("audit output must be outside immutable input tree")
    output.mkdir(parents=True, exist_ok=False)
    registration = json.loads((root / "preregistration.json").read_text())
    inputs = {
        str(p.relative_to(root)): sha(p.read_bytes())
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.is_symlink()
    }
    episodes = []
    all_turns = []
    operations = Counter()
    errors = Counter()
    expected = Counter(
        (case, seed, arm)
        for case in (
            "j100_navigation_onboarding",
            "map_to_odom_missing",
            "scan_stale",
            "nav2_lifecycle_inactive",
            "footprint_undersized",
            "duplicate_velocity_route",
        )
        for seed in (101, 202, 303)
        for arm in ("A", "B", "C")
    )
    if Counter((a["case"], a["seed"], a["arm"]) for a in registration["allocation"]) != expected:
        raise ValueError("54 distinct preregistered paired cells required")
    for allocation in registration["allocation"]:
        directory = root / f"episode-{allocation['ordinal']:02d}"
        row = json.loads((directory / "original-result.json").read_text())
        for key, value in allocation.items():
            if row[key] != value:
                raise ValueError("allocation/result identity mismatch")
        model_dir = directory / "model-episode"
        trace_path = model_dir / "original-tool-trace.json"
        trace = json.loads(trace_path.read_text())
        if len(trace) != len(row["raw_model_calls"]):
            raise ValueError("model/trace join is not one-to-one")
        turns = []
        reads = defaultdict(list)
        operation_counts = Counter()
        reconstructed = []
        first_proposal = None
        first_test = None
        last_elapsed = 0
        last_prefix = ""
        for index, (record, call) in enumerate(zip(trace, row["raw_model_calls"], strict=True)):
            call_dir = model_dir / f"model-turn-{index:02d}"
            request = (call_dir / "request.txt").read_text()
            prefix, transcript_text = request.rsplit("\nTRANSCRIPT\n", 1)
            if json.loads(transcript_text) != reconstructed:
                raise ValueError("saved prompt differs from independently replayed transcript")
            last_prefix = prefix
            events = [
                json.loads(line)
                for line in (call_dir / "original-events.jsonl").read_text().splitlines()
                if line.strip()
            ]
            messages = [
                e["item"]["text"]
                for e in events
                if e.get("type") == "item.completed"
                and e.get("item", {}).get("type") == "agent_message"
            ]
            decision = json.loads(messages[-1])
            if decision != record["original_decision"]:
                raise ValueError("raw model/trace decision mismatch")
            actual_usage = [e["usage"] for e in events if e.get("type") == "turn.completed"]
            if actual_usage != [call["outcome"]["usage"]]:
                raise ValueError("raw usage/record mismatch")
            op = decision["operation"]
            args = json.loads(decision["arguments"])
            result = record["tool_result"]
            raw = json.dumps(result, ensure_ascii=False)
            rejected = str(result.get("status", "")).startswith("REJECTED_")
            operation_counts[op] += 1
            operations[op] += 1
            if rejected:
                errors[result.get("error", result["status"])] += 1
            overlap = 0
            interval = None
            if op == "read_file" and not rejected and "text" in result:
                start = result["offset"]
                end = start + len(result["text"])
                key = (result["file"], result.get("pointer", ""))
                # Union of prior intervals avoids double-counting repeated overlaps.
                clipped = sorted(
                    (max(start, a), min(end, b))
                    for a, b in reads[key]
                    if max(start, a) < min(end, b)
                )
                merged = []
                for a, b in clipped:
                    if merged and a <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], b)
                    else:
                        merged.append([a, b])
                overlap = sum(b - a for a, b in merged)
                reads[key].append((start, end))
                interval = [start, end]
            if (
                op in {"propose_patch", "select_interfaces"}
                and not rejected
                and first_proposal is None
            ):
                first_proposal = {
                    "turn_one_based": index + 1,
                    "elapsed_model_phase_sec": record["elapsed_sec"],
                    "operation": op,
                    "meaning": "statically accepted proposal; not correct diagnosis or physical acceptance",
                }
            if op in {"run_tests", "finish"} and row["tests"] and first_test is None:
                first_test = {
                    "turn_one_based": index + 1,
                    "elapsed_model_phase_sec": record["elapsed_sec"],
                }
            delivered = shown_result(record)
            # The runner's special C rejection bypasses large-result offloading.
            if result.get("status") == "REJECTED_MISSING_TRAINING_MEMORY_QUERY":
                delivered = result
            turn = {
                "ordinal": row["ordinal"],
                "arm": row["arm"],
                "case": row["case"],
                "seed": row["seed"],
                "turn_one_based": index + 1,
                "operation": op,
                "arguments": args,
                "request_bytes": len(request.encode()),
                "request_sha256": sha(request.encode()),
                "tool_result_characters": len(raw),
                "tool_result_utf8_bytes": len(raw.encode()),
                "delivered_result_bytes": len(json.dumps(delivered, ensure_ascii=False).encode()),
                "result_offloaded": len(raw) > 16000,
                "rejected": rejected,
                "error": result.get("error"),
                "read_interval_characters": interval,
                "repeat_overlap_characters": overlap,
                "elapsed_model_phase_sec": record["elapsed_sec"],
                "previous_tool_plus_next_model_and_host_interval_sec": record["elapsed_sec"]
                - last_elapsed,
                "timing_note": "original record stamps occur after model decision and before tool; isolated model/tool durations unavailable",
                "raw_usage": actual_usage[0],
                "memory_refs_used": decision.get("memory_refs_used", []),
                "trace_ref": reference(trace_path, root),
                "raw_events_ref": reference(call_dir / "original-events.jsonl", root),
            }
            turns.append(turn)
            all_turns.append(turn)
            last_elapsed = record["elapsed_sec"]
            reconstructed.extend([{"model": decision}, {"tool_result": delivered}])
        hypothetical = (
            last_prefix + "\nTRANSCRIPT\n" + json.dumps(reconstructed, ensure_ascii=False)
        )
        if row["status"] == "CONTEXT_BYTE_BUDGET_EXHAUSTED" and len(hypothetical.encode()) <= 48000:
            raise ValueError("context exit is not supported by replayed next prompt")
        if row["status"] == "MODEL_OR_TOOL_BUDGET_EXHAUSTED" and len(turns) != 12:
            raise ValueError("tool exhaustion without 12 original calls")
        successful_memory = row.get("memory_causality_trace", [])
        admitted = {
            r["memory_ref"]
            for q in successful_memory
            for r in q["retrieval_and_admission"].get("admitted", [])
        }
        used = {ref for turn in turns for ref in turn["memory_refs_used"]}
        if not used.issubset(admitted):
            raise ValueError("unadmitted Memory reference")
        episodes.append(
            {
                **allocation,
                "status": row["status"],
                "verified_success": row["verified_success"],
                "operation_counts": dict(operation_counts),
                "first_static_proposal": first_proposal,
                "first_validation_request": first_test,
                "request_bytes_max": max(t["request_bytes"] for t in turns),
                "replayed_next_prompt_bytes": len(hypothetical.encode()),
                "repeated_read_turns": sum(t["repeat_overlap_characters"] > 0 for t in turns),
                "repeated_read_characters": sum(t["repeat_overlap_characters"] for t in turns),
                "rejected_turns": sum(t["rejected"] for t in turns),
                "result_offloads": sum(t["result_offloaded"] for t in turns),
                "memory_attempts": operation_counts["query_memory"],
                "actual_memory_queries": len(successful_memory),
                "admitted_memory_aliases": sorted(admitted),
                "model_referenced_memory_aliases": sorted(used),
                "capture_sec": row["capture_sec"],
                "model_and_tools_phase_sec": row["elapsed_sec"],
                "total_wall_sec": row["total_elapsed_sec"],
                "model_call_duration_sec": None,
                "physical_validation_duration_sec": None,
                "duration_note": "capture and enclosing phase clocks measured; isolated call/validation durations not recorded; no mtime substitution",
                "new_World_count": len(row["original_operator_Worlds"]),
                "tests": row["tests"],
                "original_result_ref": reference(directory / "original-result.json", root),
            }
        )
    if len(episodes) != 54 or len(all_turns) != 546:
        raise ValueError("unexpected original population")
    groups = {}
    for arm in ("A", "B", "C"):
        selected = [e for e in episodes if e["arm"] == arm]
        groups[arm] = {
            "episodes": len(selected),
            "model_calls": sum(sum(e["operation_counts"].values()) for e in selected),
            "static_proposal_episodes": sum(
                e["first_static_proposal"] is not None for e in selected
            ),
            "validation_request_episodes": sum(
                e["first_validation_request"] is not None for e in selected
            ),
            "verified_successes": sum(e["verified_success"] for e in selected),
            "median_capture_sec": median(e["capture_sec"] for e in selected),
            "median_model_and_tools_phase_sec": median(
                e["model_and_tools_phase_sec"] for e in selected
            ),
            "median_total_wall_sec": median(e["total_wall_sec"] for e in selected),
        }
    report = {
        "schema_version": "rosclaw.g5.forensics.v1",
        "source": registration["source"],
        "status": "PASS_ORIGINAL_RECORD_REPLAY_NOT_MODEL_BENEFIT",
        "population": 54,
        "raw_model_turns": len(all_turns),
        "grain": "one original model decision and joined tool outcome per turn",
        "operations": dict(operations),
        "exit_statuses": dict(Counter(e["status"] for e in episodes)),
        "rejected_reasons": dict(errors),
        "arm_summary": groups,
        "episodes": episodes,
        "turns": all_turns,
        "timing_missingness": "isolated SDK preparation/model/tool/physical intervals were not fully instrumented; enclosing phases only",
        "originals_modified": False,
        "rerun_models_or_Worlds": False,
        "v1_done": False,
    }
    (output / "G5_V1_FAILURE_TAXONOMY.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "original-input-sha256.json").write_text(json.dumps(inputs, indent=2) + "\n")
    for name, digest in inputs.items():
        if sha((root / name).read_bytes()) != digest:
            raise ValueError("input changed during audit")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    r = audit(args.input, args.output)
    print(
        json.dumps(
            {
                k: r[k]
                for k in [
                    "status",
                    "population",
                    "raw_model_turns",
                    "operations",
                    "exit_statuses",
                    "arm_summary",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
