"""Bounded host-side diagnostics from existing SDK public failure artifacts.

No process/ROS access, repair selection, threshold changes or success promotion.
A recorded runtime error is a diagnostic clue, not independent physical proof.
"""

import hashlib
import json
import stat
from pathlib import Path


def failure_feedback(world, *, root):
    root = Path(root).resolve()
    world = Path(world).resolve()
    world.relative_to(root)
    refs = []

    def read(relative, *, as_json=True):
        path = world / relative
        if not path.exists():
            return None
        try:
            path.resolve().relative_to(world)
        except ValueError:
            raise ValueError("bounded original SDK diagnostic path required") from None
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 2_000_000:
            raise ValueError("bounded original SDK public artifact required")
        with path.open("rb") as stream:
            raw = stream.read(2_000_001)
        if len(raw) > 2_000_000 or len(raw) != metadata.st_size:
            raise ValueError("original diagnostic artifact changed during bounded read")
        refs.append(
            {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(raw).hexdigest()}
        )
        return json.loads(raw) if as_json else raw.decode(errors="replace")

    l1 = read("actual-live-L1-readiness-review.json")
    journey = read("execution/governed-journey-original-outcome.json")
    daemon = read("execution/governed-daemon-original.log", as_json=False)
    blocked = []
    unknown = []
    for line in (daemon or "").splitlines():
        marker = "current ROS/source expired or rejected: "
        if marker not in line:
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(line.split(marker, 1)[1])
            health = value["health"]
            for key, target in (("blocked_reasons", blocked), ("unknown_reasons", unknown)):
                for reason in health.get(key, []):
                    if type(reason) is str and reason not in target:
                        target.append(reason[:200])
        except (ValueError, KeyError, TypeError):
            continue
    goals = (journey or {}).get("goals", [])
    ready = (l1 or {}).get("status") == "READY_FOR_SIM_TEST"
    return {
        "verified_success": False,
        "failure_stage": "CANONICAL_EXECUTION_NOT_COMPLETED"
        if ready
        else "BOOTSTRAP_OR_L1_NOT_ACCEPTED",
        "initial_L1": (l1 or {}).get("status", "UNKNOWN_NOT_RECORDED"),
        "initial_source_binding": (l1 or {})
        .get("source_binding", {})
        .get("result", "UNKNOWN_NOT_RECORDED"),
        "canonical_goals": [
            {
                "action_id": row.get("action_id"),
                "final_state": row.get("final_state"),
                "physical_stop_verified": row.get("verification", {}).get("physical_stop_verified"),
            }
            for row in goals[:2]
        ],
        "recorded_runtime_health_blocked_reasons": blocked[:8],
        "recorded_runtime_health_unknown_reasons": unknown[:8],
        "next_check": (
            "review effective published footprint, padding and TF time alignment; static radius alone is not continuous L1; recompile allowed staging and explicitly revalidate"
            if any(reason.startswith("EFFECTIVE_FOOTPRINT_UNDERSIZES_BODY:") for reason in blocked)
            else "review original L1 and canonical failure artifacts before another explicitly budgeted validation"
        ),
        "diagnostic_scope": "RECORDED_RUNTIME_DIAGNOSTIC_NOT_INDEPENDENT_STOP_OR_CAUSAL_PROOF",
        "original_diagnostic_refs": refs,
        "current_authority": "NONE",
        "safety_thresholds_changed": False,
    }


def safe_failure_feedback(world, *, root):
    """A diagnostic read failure cannot replace the original failed SDK verdict."""
    try:
        return failure_feedback(world, root=root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {
            "verified_success": False,
            "failure_stage": "SDK_WORLD_FAILED_DIAGNOSTIC_UNAVAILABLE",
            "diagnostic_error_type": type(error).__name__,
            "initial_L1": "UNKNOWN_NOT_REVIEWED",
            "current_authority": "NONE",
            "next_check": "operator review of retained original SDK failure artifacts required",
        }
