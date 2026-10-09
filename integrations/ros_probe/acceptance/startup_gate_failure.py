"""Original SIM startup gate diagnostics; no retry, readiness or stop claim."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from observations import latest_completed_observation


def retain_startup_failure(directory):
    root = Path(directory)
    report = {
        "schema_version": "rosclaw.SIM_startup_gate_failure.v1",
        "captured_at": datetime.now(UTC).isoformat(),
        "root_cause": "UNKNOWN",
        "physical_stop_proof": "NOT_MEASURED",
        "readiness": False,
        "authorization": False,
        "missing_requirements": [],
        "original_log_snapshots": {},
    }
    try:
        sample = latest_completed_observation(root / "witness.jsonl")
        age = (datetime.now(UTC) - datetime.fromisoformat(sample["captured_at"])).total_seconds()
        report["original_last_completed_witness"] = sample
        report["original_witness_age_sec"] = age
        for name, okay in {
            "fresh_independent_witness": 0 <= age < 0.5,
            "complete_independent_observations": sample.get("observation_complete") is True,
            "zero_observed_collisions": sample.get("collision_count") == 0,
            "brush_OFF_before_mission": sample.get("cleaning_enabled") is False,
        }.items():
            if not okay:
                report["missing_requirements"].append(name)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["missing_requirements"].append("decodable_independent_observation")
        report["observation_error"] = str(error)
    if not (root / "measured_map.json").is_file():
        report["missing_requirements"].append("original_measured_map")
    for name in ("nav2.log", "coverage_lifecycle.log"):
        path = root / name
        try:
            if path.is_symlink():
                raise ValueError("owned regular original startup log required")
            with path.open("rb") as source:
                raw = source.read(64_000_001)
            if len(raw) > 64_000_000:
                raise ValueError("bounded startup log snapshot required")
            report["original_log_snapshots"][name] = {
                "original_size_bytes": len(raw),
                "original_sha256": hashlib.sha256(raw).hexdigest(),
                "original_tail_utf8": raw[-4096:].decode("utf-8", errors="replace"),
                "actual_live_lifecycle_state": "NOT_MEASURED",
            }
            if b"Managed nodes are active" not in raw:
                report["missing_requirements"].append("startup_completion_marker:" + name)
        except (OSError, ValueError) as error:
            report["missing_requirements"].append("readable_original_startup_log:" + name)
            report["original_log_snapshots"][name] = {"error": str(error)}
    with (root / "startup-gate-failure.json").open("x") as output:
        json.dump(report, output, indent=2)
    return report
