"""Operator SIM collector fault trigger; no robot command or SDK transport."""

import argparse
import hashlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

from backend_probe_world import bounded_source
from observations import latest_completed_observation
from owned_collection_pause import OwnedCollectionPause
from probe_scene_geometry import decode_scene_json
from qualified_backend_episode import qualified_projection


def eligible_cleaning_sample(sample, snapshot, brush_binding, *, unix_time, monotonic_time):
    """Only fresh measured Brush ON with an actual healthy source can trigger."""
    if type(sample) is not dict or type(snapshot) is not dict:
        raise ValueError("typed original pre-fault observations required")
    if any(
        type(sample.get(k)) not in (int, float) or not math.isfinite(sample[k])
        for k in ("time_sec", "lease_remaining_sec", "physics_collision_count")
    ) or any(
        type(snapshot.get(k)) not in (int, float) or not math.isfinite(snapshot[k])
        for k in ("robot_sim_time_sec", "sampled_monotonic_sec")
    ):
        raise ValueError("finite original pre-fault clock and actuator source required")
    receipt = datetime.fromisoformat(sample["captured_at"])
    if (
        receipt.tzinfo is None
        or not 0 <= unix_time - receipt.timestamp() < 0.3
        or not 0 <= monotonic_time - snapshot["sampled_monotonic_sec"] < 0.15
        or abs(sample["time_sec"] - snapshot["robot_sim_time_sec"]) > 0.15
        or sample.get("evidence_domain") != "GAZEBO_PHYSICS"
        or sample.get("observation_complete") is not True
        or "brush_source_fault" not in sample
        or sample["brush_source_fault"] is not None
        or "physics_source_fault" not in sample
        or sample["physics_source_fault"] is not None
        or sample.get("brush_source_binding") != brush_binding
        or sample["physics_collision_count"] != 0
        or snapshot.get("source_fault") is not None
        or snapshot.get("live_source_constraint_satisfied") is not True
        or snapshot.get("robot_collision_count") != 0
        or snapshot.get("probe_completed_cache_cycles", 0) < 1
    ):
        raise ValueError("fresh actual same-time healthy source before collector fault required")
    return sample.get("cleaning_enabled") is True and sample["lease_remaining_sec"] > 0


def run(directory, scenario_path):
    root = Path(directory)
    raw = bounded_source(scenario_path, 65536)
    spec = decode_scene_json(raw)
    plan = decode_scene_json(bounded_source(root / "backend-stack-source-plan.json"))
    binding = decode_scene_json(bounded_source(root / "brush_binding.json"))
    policy_path = root / "backend-collection-pause-policy.json"
    policy_raw = bounded_source(policy_path, 65536)

    # Validation creates no process or signal. Signals are owned by backend_stack.
    class NoChildren:
        children = ()

    OwnedCollectionPause(root, NoChildren(), plan, policy_path)
    if (
        type(spec) is not dict
        or spec.get("schema_version") != "rosclaw.dynamic_fixture_scenario.v1"
        or spec.get("case") != "D5"
        or spec.get("fault_kind") != "OWNED_COLLECTION_PAUSE"
        or spec.get("run_id") != plan["run_id"]
        or type(spec.get("mission_id")) is not str
        or not spec["mission_id"]
        or type(spec.get("pause_wall_sec")) is not int
        or any(binding.get(k) != plan[k] for k in ("run_id", "body_snapshot_hash"))
        or spec.get("pause_wall_sec") != decode_scene_json(policy_raw)["pause_wall_sec"]
        or type(spec.get("introduce_after_cleaning_sim_sec")) is not int
        or not 1 <= spec["introduce_after_cleaning_sim_sec"] <= 120
        or type(spec.get("wall_timeout_sec")) is not int
        or not 60 <= spec["wall_timeout_sec"] <= 1920
    ):
        raise ValueError("exact frozen D5 collector scenario required")
    deadline, first_on = time.monotonic() + spec["wall_timeout_sec"], None
    with (root / "dynamic-scenario-events.jsonl").open("x", buffering=1) as log:

        def emit(kind, **values):
            log.write(
                json.dumps(
                    {
                        "kind": kind,
                        "captured_at": datetime.now(UTC).isoformat(),
                        "scenario_sha256": hashlib.sha256(raw).hexdigest(),
                        "run_id": spec["run_id"],
                        "mission_id": spec["mission_id"],
                        "physical_acceptance": "NOT_VERIFIED",
                        **values,
                    }
                )
                + "\n"
            )

        emit("SCENARIO_STARTED", policy=spec)
        try:
            while time.monotonic() < deadline:
                if (
                    bounded_source(scenario_path, 65536) != raw
                    or bounded_source(policy_path, 65536) != policy_raw
                ):
                    raise ValueError("frozen collector fixture changed")
                sample = latest_completed_observation(root / "witness.jsonl")
                projection = qualified_projection(root)
                enabled = eligible_cleaning_sample(
                    sample,
                    projection["snapshot"],
                    binding,
                    unix_time=time.time(),
                    monotonic_time=time.monotonic(),
                )
                if first_on is None and enabled:
                    first_on = sample["time_sec"]
                    emit("FIRST_ENABLED_CLEANING", sim_time_sec=first_on)
                if (
                    enabled
                    and first_on is not None
                    and sample["time_sec"] - first_on >= spec["introduce_after_cleaning_sim_sec"]
                ):
                    with (root / "backend-collection-pause-before.json").open("x") as out:
                        json.dump(
                            {
                                "sample": sample,
                                "backend_projection": projection,
                                "policy_sha256": hashlib.sha256(policy_raw).hexdigest(),
                                "physical_acceptance": "NOT_VERIFIED",
                            },
                            out,
                            indent=2,
                        )
                    with (root / "backend-collection-pause-request.json").open("x") as out:
                        json.dump(
                            {"fixture_policy_sha256": hashlib.sha256(policy_raw).hexdigest()}, out
                        )
                    emit(
                        "PERTURBATION_COMPLETE_REQUIRES_NATIVE_AND_CREDIT_VALIDATION",
                        fixture_request_written=True,
                        actual_pause_confirmed=False,
                        sim_time_sec=sample["time_sec"],
                        stop_proof="NOT_MEASURED",
                    )
                    return
                time.sleep(0.02)
            raise TimeoutError("frozen collector scenario deadline exhausted before trigger")
        except Exception as error:
            emit("SCENARIO_FAILED", error=type(error).__name__ + ": " + str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("/evidence"))
    parser.add_argument("--scenario", type=Path, required=True)
    args = parser.parse_args()
    run(args.directory, args.scenario)


if __name__ == "__main__":
    main()
