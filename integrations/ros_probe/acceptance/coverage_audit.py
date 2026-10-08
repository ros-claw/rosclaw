"""Read-only plan/execution audit of an existing SIM episode; never dispatches."""

import argparse
import gzip
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path

from rosclaw.connectors.ros.diagnosis.coverage_audit import (
    digest,
    plan_projection,
    read_audit,
    trajectory_metrics,
)
from rosclaw.connectors.ros.verification.coverage import CleaningPose, CoverageVerifier


def load(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            text = stream.read()
    else:
        text = path.read_text()
    return json.loads(text)


def evidence_source(directory, recorded_path):
    """Prefer reconstructed public bytes over a legacy machine-specific path."""
    recorded = Path(recorded_path)
    for local in [directory / "actions" / recorded.name, directory / recorded.name]:
        for candidate in [local, Path(str(local) + ".gz")]:
            if candidate.is_file():
                return candidate
    if recorded.is_file():
        return recorded
    raise ValueError("cannot locate original mission evidence by canonical artifact name")


def audit(directory, output):
    receipt = load(directory / "golden-coverage.receipt.json")["receipt"]
    source = evidence_source(directory, receipt["verification_result"]["evidence_artifact"]["path"])
    evidence = load(source)
    original_bytes = (
        gzip.decompress(source.read_bytes()) if source.suffix == ".gz" else source.read_bytes()
    )
    if (
        hashlib.sha256(original_bytes).hexdigest()
        != receipt["verification_result"]["evidence_artifact"]["sha256"]
    ):
        raise ValueError("canonical evidence hash mismatch")
    grid, trace = evidence["grid"], evidence["trajectory"]
    verifier = CoverageVerifier(**grid)
    events = []
    source_files = [source, directory / "golden-coverage.receipt.json"]
    audit_complete = True
    for path in sorted((directory / "actions").glob("coverage-audit-*.jsonl")):
        rows = read_audit(path)
        if not rows or rows[0].get("action_id") != receipt["action_id"]:
            continue
        events.extend(rows)
        summary = load(Path(str(path) + ".summary.json"))
        audit_complete &= summary["complete"]
        source_files.extend([path, Path(str(path) + ".summary.json")])
    if not events:
        audit_complete = False
    progress = next((r for r in events if r["kind"] == "coverage_progress"), None)
    ended = next(
        (
            r
            for r in events
            if r["kind"] == "goal_ended" and r["payload"]["nav_goal_id"] == receipt["action_id"]
        ),
        None,
    )
    primary = next((r for r in events if r["kind"] == "primary_completed"), None)
    # Preserve the original no-boundary checkpoint definition for old episodes.
    main_count = (
        ended["payload"].get("consumed_samples")
        if primary and ended
        else progress["payload"]["consumed_samples"]
        if progress
        else ended["payload"].get("consumed_samples")
        if ended
        else None
    )
    primary_count = primary["payload"]["consumed_samples"] if primary else main_count
    starts = [(0, "MAIN_COVERAGE", receipt["action_id"])]
    for row in events:
        if row["kind"] == "goal_started" and row["payload"].get("stage") in (
            "REPAIR",
            "BOUNDARY_PASS",
        ):
            starts.append(
                (
                    min(len(trace), max(0, row["payload"]["sample_offset"])),
                    row["payload"]["stage"],
                    row["payload"]["nav_goal_id"],
                )
            )
    starts.sort()
    segments = []
    before = 0
    main_cells = None
    primary_cells = None
    historic_checkpoint_indices = []
    repairs = receipt["verification_result"].get("recovery_attempts", [])
    checkpoint_ratio = repairs[0]["coverage_before"] if repairs else None
    offsets = {s[0] for s in starts} | {len(trace)}
    if main_count is not None:
        offsets.add(main_count)
    if primary_count is not None:
        offsets.add(primary_count)
    cursor = 0
    for end in sorted(offsets):
        if end == 0:
            continue
        for index, pose in enumerate(trace[cursor:end], cursor):
            verifier.observe(CleaningPose(**pose), frame_id=evidence["frame_id"])
            if (
                main_count is None
                and checkpoint_ratio is not None
                and (len(verifier.visits) / len(verifier.accessible) == checkpoint_ratio)
            ):
                historic_checkpoint_indices.append(index)
                main_cells = set(verifier.visits)
        if end == primary_count:
            primary_cells = set(verifier.visits)
        if end == main_count:
            main_cells = set(verifier.visits)
        active = max((s for s in starts if s[0] <= cursor), key=lambda s: s[0])
        segments.append(
            {
                "stage": active[1] if events else "UNKNOWN",
                "nav_goal_id": active[2] if events else None,
                "sample_start": cursor,
                "sample_end_exclusive": end,
                "new_covered_cells": len(verifier.visits) - before,
                "coverage_ratio": len(verifier.visits) / len(verifier.accessible),
                **trajectory_metrics(trace[max(0, cursor - 1) : end]),
                "phase_boundary_note": "consumer sample offset; waiting included",
            }
        )
        before, cursor = len(verifier.visits), end
    saved = directory / (
        source.name.removesuffix(".gz").removesuffix(".json") + ".verification.json"
    )
    if not saved.exists():
        saved = source.parent / saved.name
    saved_equal = verifier.result() == load(saved)["coverage"] if saved.exists() else None
    if saved_equal is False:
        raise ValueError("saved canonical verifier differs from exact replay")
    if verifier.result()["coverage_ratio"] != receipt["verification_result"]["coverage_ratio"]:
        raise ValueError("canonical coverage differs from exact replay")
    plans = []
    plan_rows = []
    for path in sorted(directory.glob("plan-events-*.jsonl")):
        plan_rows.extend(read_audit(path))
        source_files.append(path)
        summary_path = Path(str(path) + ".summary.json")
        if summary_path.exists():
            source_files.append(summary_path)
            audit_complete &= load(summary_path)["complete"]
        else:
            audit_complete = False
    freeze_path = directory / "source-freeze.json"
    freeze = load(freeze_path) if freeze_path.exists() else None
    binding_errors = []
    if freeze is None or freeze.get("working_tree_dirty") is not False:
        binding_errors.append("missing or dirty source freeze")
    elif not events or any(
        any(row.get(k) != freeze.get(k) for k in ["run_id", "git_sha", "map_hash", "geometry_hash"])
        or row.get("body_snapshot_hash") != receipt["body_snapshot_hash"]
        for row in events
    ):
        binding_errors.append(
            "daemon event provenance differs from source freeze or canonical Body"
        )
    if freeze and any(row.get("run_id") != freeze["run_id"] for row in plan_rows):
        binding_errors.append("plan observer belongs to another run")
    audit_complete &= not binding_errors
    for name in [
        "source-freeze.json",
        "snapshot.json",
        "measured_map.json",
        "execution_config.json",
        "nav2.yaml",
        "body.json",
        "robot.urdf",
        "witness.jsonl",
        "world.sdf",
        "fixture_profile.json",
        "experiment.json",
        "protocol.json",
        "golden-localize.receipt.json",
        "golden-remember.receipt.json",
    ]:
        path = directory / name
        if path.exists():
            source_files.append(path)
    # Header timestamps on old Nav2 paths may be absent/zero. Use observed
    # wall capture intervals, and explicitly retain ambiguity as UNKNOWN.
    main_start = next(
        (
            r
            for r in events
            if r["kind"] == "goal_started" and r["payload"].get("stage") == "MAIN_COVERAGE"
        ),
        None,
    )
    main_end = next(
        (
            r
            for r in events
            if r["kind"] == "goal_ended" and r["payload"]["nav_goal_id"] == receipt["action_id"]
        ),
        None,
    )
    for row in plan_rows:
        if row["kind"] == "path" and row["payload"]["topic"] == "/coverage_server/coverage_plan":
            payload = row["payload"]
            bound = bool(
                main_start
                and main_end
                and datetime.fromisoformat(main_start["captured_at"])
                <= datetime.fromisoformat(row["captured_at"])
                <= datetime.fromisoformat(main_end["captured_at"])
            )
            projection = plan_projection(grid, payload["poses"], frame_id=payload["frame_id"])
            plans.append(
                {
                    "plan_id": payload["plan_id"],
                    "event_hash": row["artifact_sha256"],
                    "nav_goal_id": receipt["action_id"] if bound else None,
                    "binding_method": "single_serialized_main_goal_capture_interval"
                    if bound
                    else "UNKNOWN",
                    **projection,
                }
            )
    predicted = set().union(*(set(p["predicted_cells"]) for p in plans if p["nav_goal_id"]))
    missed = verifier.accessible - (main_cells if main_cells is not None else set(verifier.visits))
    attribution = {}
    for cell in sorted(missed):
        label = (
            "NOT_IN_PLANNED_SWATH"
            if audit_complete and predicted and cell not in predicted
            else "UNKNOWN"
        )
        attribution.setdefault(label, []).append(cell)
    feedback = {}
    for row in events:
        if row["kind"] != "nav_feedback":
            continue
        payload = row["payload"]
        entry = feedback.setdefault(
            payload["nav_goal_id"],
            {
                "observed_feedback_count": 0,
                "max_number_of_recoveries": None,
                "last_distance_remaining": None,
            },
        )
        entry["observed_feedback_count"] += 1
        values = payload["values"]
        count = values.get("number_of_recoveries")
        if type(count) is int and count >= 0:
            entry["max_number_of_recoveries"] = max(count, entry["max_number_of_recoveries"] or 0)
        if type(values.get("distance_remaining")) in (int, float):
            entry["last_distance_remaining"] = values["distance_remaining"]
    summary = {
        "schema_version": "rosclaw.coverage_causal_audit.v1",
        "evidence_role": "historical_diagnostic_replay_not_new_physical_episode",
        "audit_complete": bool(audit_complete and plans),
        "source_binding_errors": binding_errors,
        "source_commit": freeze.get("git_sha") if freeze else None,
        "run_id": freeze.get("run_id") if freeze else None,
        "body_snapshot_hash": receipt["body_snapshot_hash"],
        "denominator_hash": digest(grid),
        "fixed_denominator_cells": len(verifier.accessible),
        "canonical_verifier_replay_equal": saved_equal,
        "observed_final_coverage_ratio": verifier.result()["coverage_ratio"],
        "main_sample_count": main_count,
        "primary_before_repair_sample_count": primary_count,
        "primary_before_repair_coverage_ratio": len(primary_cells) / len(verifier.accessible)
        if primary_cells is not None
        else None,
        "boundary_nav_goal_result": receipt["verification_result"].get("boundary_action_result"),
        "main_nav_goal_result": receipt["verification_result"].get("initial_action_result"),
        "historical_checkpoint_ratio": checkpoint_ratio if main_count is None else None,
        "historical_checkpoint_sample_range": [
            historic_checkpoint_indices[0],
            historic_checkpoint_indices[-1],
        ]
        if historic_checkpoint_indices
        else None,
        "historical_checkpoint_note": "Exact coverage replay; phase time is not identified by an old receipt alone",
        "main_observed_coverage_ratio": len(main_cells) / len(verifier.accessible)
        if main_cells is not None
        else None,
        "main_metrics": trajectory_metrics(trace[:main_count]) if main_count is not None else None,
        "total_metrics": trajectory_metrics(trace),
        "plans": plans,
        "first_pass_missed_attribution": attribution,
        "attribution_note": "Not-in-plan means outside recorded ideal coverage path sweep, not proof of unreachable geometry; planned-but-missed remains UNKNOWN without causal evidence.",
        "feedback_events": sum(r["kind"] == "nav_feedback" for r in events),
        "feedback_by_goal": feedback,
        "controller_diagnostic_events": {
            kind: sum(row["kind"] == kind for row in plan_rows)
            for kind in ["collision_monitor_state", "velocity_command"]
        },
        "segment_count": len(segments),
        "limitations": [
            "Plan is diagnostic only, never measured coverage credit",
            "No controller/collision-monitor state or unexposed internal route stage is inferred",
            "Historical runs without events have UNKNOWN stage/heading evidence",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan-versus-execution.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "coverage-segment-metrics.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in segments)
    )
    manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_overlay(
        output / "plan_execution_overlay.svg",
        grid,
        trace,
        main_count,
        main_cells,
        plan_rows,
        summary,
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in {"plans", "first_pass_missed_attribution"}
            },
            indent=2,
        )
    )


def write_overlay(path, grid, trace, main_count, main_cells, plan_rows, summary):
    """Portable diagnostic figure; original evidence is never modified."""
    scale = 600 / max(grid["width"] * grid["resolution"], grid["height"] * grid["resolution"])

    def xy(x, y):
        return (45 + (x - grid["origin"][0]) * scale, 665 - (y - grid["origin"][1]) * scale)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="740" viewBox="0 0 1040 740">',
        '<rect width="1040" height="740" fill="#fafaf8"/>',
        '<g font-family="sans-serif" fill="#172a3a">',
        '<text x="45" y="32" font-size="21">Coverage plan vs observed first pass (SIM)</text>',
    ]
    covered = main_cells if main_cells is not None else set()
    for cell in grid["accessible_cells"]:
        row, col = divmod(cell, grid["width"])
        x, y = xy(
            grid["origin"][0] + col * grid["resolution"],
            grid["origin"][1] + (row + 1) * grid["resolution"],
        )
        size = scale * grid["resolution"]
        color = "#b8dfc1" if cell in covered else "#f5d8a6"
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{size:.2f}" height="{size:.2f}" fill="{color}"/>'
        )
    for row in plan_rows:
        payload = row["payload"]
        if row["kind"] == "path" and payload["topic"] == "/coverage_server/coverage_plan":
            points = " ".join(
                f"{x:.2f},{y:.2f}" for x, y in [xy(p["x"], p["y"]) for p in payload["poses"]]
            )
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="#2153a0" stroke-width="3"/>'
            )
    if main_count is not None:
        points = " ".join(
            f"{x:.2f},{y:.2f}" for x, y in [xy(p["x"], p["y"]) for p in trace[:main_count]]
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="#9b365e" stroke-width="1.6"/>'
        )
    labels = [
        "Green: observed first-pass credit",
        "Orange: first-pass missed cells",
        "Blue: recorded coverage plan",
        "Purple: observed first-pass path",
        f"Final coverage: {summary['observed_final_coverage_ratio'] * 100:.4f}%",
        "Prediction is never measured credit",
        f"Audit complete: {summary['audit_complete']}",
    ]
    if summary["main_observed_coverage_ratio"] is not None:
        labels.insert(4, f"First pass: {summary['main_observed_coverage_ratio'] * 100:.4f}%")
    for i, label in enumerate(labels):
        parts.append(f'<text x="675" y="{100 + i * 34}" font-size="15">{html.escape(label)}</text>')
    parts.append("</g></svg>")
    path.write_text("\n".join(parts) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit(args.directory.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
