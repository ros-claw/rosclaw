"""Closed same-model experimental allocation and evidence validation, no ROS."""

import hashlib
import json
import random
from collections import Counter

CASES = (
    "j100_navigation_onboarding",
    "map_to_odom_missing",
    "scan_stale",
    "nav2_lifecycle_inactive",
    "footprint_undersized",
    "duplicate_velocity_route",
)
SEEDS = (101, 202, 303)
ARMS = ("A", "B", "C")


def allocate(order_seed=1013):
    """Paired tasks stay together; order within every pair is randomized."""
    rng = random.Random(order_seed)
    pairs = [(case, seed) for case in CASES for seed in SEEDS]
    rng.shuffle(pairs)
    result = []
    for pair, (case, seed) in enumerate(pairs, 1):
        arms = list(ARMS)
        rng.shuffle(arms)
        for arm in arms:
            result.append(
                {
                    "ordinal": len(result) + 1,
                    "pair": pair,
                    "case": case,
                    "seed": seed,
                    "arm": arm,
                    "episode_id": f"{case}-{seed}-{arm}",
                }
            )
    return result


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def admit_memory(record, *, evaluation_case, evaluation_runs, urdf_sha256):
    """Reject evaluation answers and unbound training facts before context entry.

    Same-case G4 fixes are not admissible evaluation priors. Cross-case verified
    examples may inform workflow; retrieval alone is not a model benefit.
    """
    return (
        record.get("independently_verified") is True
        and record.get("claim_scope") == "SIMULATION_ONLY"
        and record.get("training_only") is True
        and record.get("fault_case_id") in CASES
        and record["fault_case_id"] != evaluation_case
        and record.get("source_urdf_sha256") == urdf_sha256
        and isinstance(record.get("episode_sha256"), str)
        and len(record["episode_sha256"]) == 64
        and bool(record.get("world_runs"))
        and not set(record["world_runs"]).intersection(evaluation_runs)
    )


def usage_from_events(events):
    """Require actual terminal provider usage; never coerce absence to zero."""
    turns = [e for e in events if e.get("type") == "turn.completed"]
    if len(turns) != 1:
        raise ValueError("exactly one original completed model turn required")
    usage = turns[0].get("usage")
    fields = ("input_tokens", "cached_input_tokens", "output_tokens")
    if not isinstance(usage, dict) or any(
        type(usage.get(k)) is not int or usage[k] < 0 for k in fields
    ):
        raise ValueError("actual provider usage missing or malformed")
    if usage["cached_input_tokens"] > usage["input_tokens"]:
        raise ValueError("cached input tokens exceed input tokens")
    return {k: usage[k] for k in fields}


def validate_complete_results(rows):
    expected = Counter((case, seed, arm) for case in CASES for seed in SEEDS for arm in ARMS)
    if Counter((r["case"], r["seed"], r["arm"]) for r in rows) != expected:
        raise ValueError("54 distinct paired actual episodes required")
    for row in rows:
        if not row.get("raw_model_calls") or not row.get("actual_usage"):
            raise ValueError("model evidence missing; not zero usage")
        usage_from_events([{"type": "turn.completed", "usage": row["actual_usage"]}])
        if row.get("verified_success") is True and not all(
            row.get(k)
            for k in ("new_world_run", "fresh_l1_ref", "canonical_receipts", "independent_stop_ref")
        ):
            raise ValueError("success lacks original canonical physics evidence")
    return True
