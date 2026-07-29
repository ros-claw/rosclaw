"""A/B/C real-machine canary (PR-EVO-HW-4, 真机自进化v2 §Phase 6).

Arms:

* A — no memory, no patch (baseline identity);
* B — fixed manual cooldown (run2's static control: 60 s every 30 rounds);
* C — operator-approved canary of the selected VALIDATED candidate
  (mechanical application on REAL hands + REAL camera, full PatchProof;
  §Phase 7: with too little evidence for autonomous APPLY, candidates
  enter an operator-approved canary — this IS that path).

Design rules (§Phase 6): same rounds, same base seed stream, arm order
randomized AND interleaved so day-level thermal drift cancels across
arms to first order, camera/model pinned (realsense + the contract's
own verification), and no unpromoted memory crosses arms (C uses only
the experiment namespace's validated candidate — never shared-corpus
memory).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

ARM_A = "A_no_memory"
ARM_B = "B_fixed_cooldown"
ARM_C = "C_candidate_canary"
ARMS = (ARM_A, ARM_B, ARM_C)

# Driver group per arm (the workspace runner's group names).
ARM_DRIVER_GROUP = {
    ARM_A: "no_memory",
    ARM_B: "fixed_cooldown",
    ARM_C: "candidate_canary",
}


@dataclass(frozen=True)
class ArmSlot:
    block: int
    arm: str
    seed: int
    driver_group: str


def build_canary_schedule(
    *,
    blocks: int,
    seed: int,
    base_seed: int,
) -> list[ArmSlot]:
    """Seeded interleaved schedule: each block runs A, B, C once in a
    shuffled order (deterministic for a given seed).

    §Phase 6 相同手势 Seed: within one block ALL arms share the same seed —
    the three arms face the IDENTICAL gesture sequence, so the dominant
    gesture-mix noise is paired away across arms.  Seeds differ between
    blocks (independent draws)."""
    rng = random.Random(seed)
    schedule: list[ArmSlot] = []
    for block in range(blocks):
        order = list(ARMS)
        rng.shuffle(order)
        block_seed = base_seed + block
        for arm in order:
            schedule.append(
                ArmSlot(
                    block=block,
                    arm=arm,
                    seed=block_seed,
                    driver_group=ARM_DRIVER_GROUP[arm],
                )
            )
    return schedule


def select_canary_candidate(
    validated: list[dict[str, Any]],
    *,
    baseline_regime: str,
    exclude_ids: set[str] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Pick ONE validated candidate for the canary with a recorded reason.

    Rule: in a thermal/tracking-degradation regime prefer a cooldown-class
    candidate (the failure mode the cooldown addresses); otherwise prefer
    a pose-recovery candidate; C0 (empty) is never canaried — it is the
    baseline identity, not an intervention.  ``exclude_ids`` filters out
    candidates that already have canary evidence — re-running a tested
    candidate wastes hardware time; the ladder walks to the next untried
    one.
    """
    import json as _json

    excluded = exclude_ids or set()
    candidates = []
    for row in validated:
        if str(row.get("candidate_id")) in excluded:
            continue
        changes = row.get("changes") or {}
        if isinstance(changes, str):
            changes = _json.loads(changes)
        if changes:  # skip C0
            candidates.append({**row, "changes": changes})
    if not candidates:
        return None, "no untried non-empty validated candidate"
    degraded = baseline_regime in (
        "THERMAL_DRIFT",
        "TRACKING_DEGRADATION",
        "THERMAL_TRACKING_DEGRADATION",
    )
    def cooldown_key(row: dict[str, Any]) -> float:
        changes = row["changes"]
        return float(changes.get("inter_round_cooldown_sec") or 0.0)

    cooldowns = [c for c in candidates if cooldown_key(c) > 0 or c["changes"].get("cooldown_every_n_rounds")]
    if degraded and cooldowns:
        # The most conservative effective cooldown wins the canary.
        pick = min(
            cooldowns,
            key=lambda c: (
                cooldown_key(c) if cooldown_key(c) > 0 else 2.5,
            ),
        )
        return pick, f"regime {baseline_regime} → conservative cooldown candidate"
    if cooldowns and not degraded:
        pose = [c for c in candidates if c["changes"].get("neutral_pose_between_blocks") or c["changes"].get("rehome_between_blocks")]
        if pose:
            return pose[0], f"regime {baseline_regime} → pose-recovery candidate"
    return candidates[0], "first validated candidate (no regime-specific rule matched)"


def select_explicit_candidate(
    validated: list[dict[str, Any]],
    candidate_id: str,
) -> tuple[dict[str, Any] | None, str]:
    """Operator-directed selection, bypassing the untried ladder.

    Used for statistical-power top-ups: a candidate that already has
    canary evidence but too few arm-C sessions for the promotion gate's
    min_sessions floor can be re-canaried explicitly.  The reason string
    discloses the operator direction — every canary must record WHY this
    candidate ran (an operator-chosen candidate is not the ladder's
    recommendation).
    """
    for row in validated:
        if str(row.get("candidate_id")) == candidate_id:
            return row, f"operator-directed top-up: {candidate_id}"
    return None, (
        f"candidate {candidate_id} is not VALIDATED "
        "(unknown id, rolled back, or already promoted/decided)"
    )
