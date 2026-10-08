"""Read-only paired SIM statistics; failed/missing seeds remain visible."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import median

METRICS = ("measured_distance_m", "sim_duration_sec")


def _quantile(values, q):
    rows = sorted(values)
    at = (len(rows) - 1) * q
    low = math.floor(at)
    return rows[low] + (rows[min(low + 1, len(rows) - 1)] - rows[low]) * (at - low)


def analyze_pairs(pairs, *, expected_seeds, profile, phase, bootstrap_samples=2000):
    """Body-specific estimates; no successful-only subset or synthesized rows."""
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("expected seeds must be nonempty and unique")
    if type(bootstrap_samples) is not int or not 100 <= bootstrap_samples <= 10000:
        raise ValueError("bootstrap sample count must be bounded")
    selected = []
    seen = set()
    for pair in pairs:
        if pair["profile"] != profile or pair["phase"] != phase:
            continue
        seed = pair["seed"]
        if seed not in expected_seeds:
            raise ValueError("unregistered seed in selected series")
        if seed in seen:
            raise ValueError("duplicate seed: attempts cannot replace failed pairs")
        seen.add(seed)
        selected.append(pair)
    selected.sort(key=lambda p: p["seed"])
    missing = sorted(set(expected_seeds) - seen)
    failures = []
    validated = []
    identities = set()
    run_ids = set()
    for pair in selected:
        problems = []
        rows = pair.get("runs", [])
        if len(rows) != 2 or {r.get("arm") for r in rows} != {"baseline", "candidate"}:
            problems.append("two distinct arms required")
        else:
            arms = {r["arm"]: r for r in rows}
            strategy = pair.get("candidate_repair_strategy", "greedy")
            deadline = pair.get("mission_timeout_sec")
            protocol = pair.get("protocol_sha256")
            if type(deadline) is not int or not 60 <= deadline <= 1800 or not protocol:
                problems.append("mission deadline or protocol binding missing")
            identities.add(
                (
                    pair["source_commit"],
                    pair["image_id"],
                    pair["candidate"],
                    strategy,
                    deadline,
                    protocol,
                )
            )
            for r in rows:
                expected_preset = "baseline" if r["arm"] == "baseline" else pair["candidate"]
                expected_strategy = "greedy" if r["arm"] == "baseline" else strategy
                if (
                    r.get("preset") != expected_preset
                    or r.get("repair_strategy", "greedy") != expected_strategy
                ):
                    problems.append("arm preset or repair strategy differs from frozen pair")
                if r.get("complete_independent_observations") is not True:
                    problems.append("complete independent observations not recorded")
                if (
                    type(r.get("mission_timeout_sec")) is not int
                    or r["mission_timeout_sec"] != deadline
                ):
                    problems.append("arm mission deadline differs from frozen pair")
                if any(
                    r.get(k) != pair[k] for k in ["profile", "seed", "source_commit", "image_id"]
                ):
                    problems.append("run differs from frozen pair")
                rid = r.get("run_id")
                if not rid or rid in run_ids:
                    problems.append("run identity missing or reused")
                run_ids.add(rid)
                if r.get("status") != "PASS" or r.get("audit_complete") is not True:
                    problems.append("physical/audit gate not passed")
                if any(
                    type(r.get(k)) is not int or r[k] != 0
                    for k in ["collision_count", "trace_gaps"]
                ):
                    problems.append("contact/gap gate not passed")
                for key, limit in [
                    ("post_cleanup_displacement_m", 0.01),
                    ("post_cleanup_yaw_change_rad", 0.03),
                ]:
                    value = r.get(key)
                    if (
                        type(value) not in (int, float)
                        or not math.isfinite(value)
                        or not 0 <= value <= limit
                    ):
                        problems.append("independent stop gate not passed")
                coverage = r.get("coverage_ratio")
                if (
                    type(coverage) not in (int, float)
                    or not math.isfinite(coverage)
                    or not 0.98 <= coverage <= 1
                ):
                    problems.append("coverage gate not passed")
                for metric in METRICS:
                    value = r.get(metric)
                    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                        problems.append(f"invalid measured {metric}")
            if not problems:
                validated.append(arms)
        if problems:
            failures.append({"seed": pair["seed"], "reasons": sorted(set(problems))})
    if len(identities) > 1:
        failures.append(
            {
                "seed": None,
                "reasons": [
                    "series source/image/candidate/repair/deadline/protocol changed; not a frozen series"
                ],
            }
        )
    eligible = not missing and not failures and len(validated) == len(expected_seeds)
    stats = {}
    if eligible:
        # Resample paired indices, never independently reshuffle the two arms.
        rng = random.Random(0)
        indices = [
            tuple(rng.randrange(len(validated)) for _ in validated)
            for _ in range(bootstrap_samples)
        ]
        for metric in METRICS:
            baseline = [p["baseline"][metric] for p in validated]
            candidate = [p["candidate"][metric] for p in validated]
            ratio = 1 - median(candidate) / median(baseline)
            paired = [1 - c / b for b, c in zip(baseline, candidate, strict=True)]
            boots = [
                1 - median(candidate[i] for i in ix) / median(baseline[i] for i in ix)
                for ix in indices
            ]
            stats[metric] = {
                "baseline_median": median(baseline),
                "candidate_median": median(candidate),
                "reduction_of_medians": ratio,
                "median_paired_reduction": median(paired),
                "paired_bootstrap_percentile_95": [
                    _quantile(boots, 0.025),
                    _quantile(boots, 0.975),
                ],
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": 0,
            }
    return {
        "schema_version": "rosclaw.paired_efficiency_analysis.v1",
        "profile": profile,
        "phase": phase,
        "evidence_role": "statistics_from_recorded_pair_summaries_not_new_physics",
        "expected_seeds": expected_seeds,
        "recorded_seeds": sorted(seen),
        "missing_seeds": missing,
        "failed_pairs": failures,
        "pair_success_rate": len(validated) / len(expected_seeds),
        "complete_frozen_series": eligible,
        "metrics": stats,
        "median_30_percent_target": eligible
        and all(s["reduction_of_medians"] >= 0.3 for s in stats.values()),
        "v1_done": False,
        "uncertainty_note": "Paired percentile bootstrap, linear quantiles; assumes independent fresh runs. Small-sample uncertainty is reported, not a guarantee of improvement.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--profile", choices=["waffle", "burger"], required=True)
    parser.add_argument("--phase", choices=["pilot", "evaluation"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    pairs = [json.loads(p.read_text()) for p in args.pairs]
    report = analyze_pairs(
        pairs,
        expected_seeds=protocol[f"{args.phase}_seeds"],
        profile=args.profile,
        phase=args.phase,
    )
    if args.phase == "evaluation":
        freeze = protocol.get("evaluation_freeze")
        if not freeze or any(
            p["source_commit"] != freeze["source_commit"]
            or p["candidate"] != freeze["selected_presets"].get(args.profile)
            or p["image_id"] != freeze.get("image_id")
            or p.get("mission_timeout_sec")
            != freeze.get("mission_timeout_sec", {}).get(args.profile)
            or p.get("candidate_repair_strategy", "greedy")
            != freeze.get("selected_repair_strategies", {}).get(args.profile, "greedy")
            for p in pairs
            if p["profile"] == args.profile and p["phase"] == args.phase
        ):
            raise ValueError("evaluation series differs from predeclared freeze")
    report["input_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.protocol, *args.pairs]
    }
    report["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "profile",
                    "complete_frozen_series",
                    "missing_seeds",
                    "median_30_percent_target",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
