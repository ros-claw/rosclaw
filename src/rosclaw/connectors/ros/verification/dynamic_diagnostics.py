"""Derived temporal exposure/revisit diagnostics; no independent attestation."""

from .coverage import CleaningPose, CoverageVerifier
from .mission import replay_coverage
from .occupancy import OccupancyAccounting


def analyze_dynamic_credit(evidence):
    """Check observed blocked-brush exposure and actual free-cell revisits.

    Exact public replay validates the complete occupancy/brush chains first.
    This calculation cannot replace canonical receipts, source admission,
    Native task success, contacts or independent physical-stop observations.
    Previously valid credit is never removed by subsequent occupancy.
    """
    coverage, temporal = replay_coverage(evidence)
    if temporal is None or not temporal["complete"]:
        raise ValueError("complete versioned temporal mission evidence required")
    verifier = CoverageVerifier(**evidence["grid"])
    accounting = OccupancyAccounting(
        verifier, mission_id=evidence["mission_id"], **evidence["occupancy_binding"]
    )
    brush_only = CoverageVerifier(**evidence["grid"])
    previous_occupied = set()
    awaiting_revisit = set()
    exposed = set()
    pending_exposed = set()
    revisited = set()
    false_credit = set()
    exposure_samples = 0
    withdrawals = []
    for pose, snapshot in zip(evidence["trajectory"], evidence["occupancy_samples"], strict=True):
        occupied = set(snapshot["occupancy"]["occupied_cells"])
        before = set(verifier.visits)
        withdrawn_unclean = (previous_occupied - occupied) - before
        awaiting_revisit.update(withdrawn_unclean)
        if withdrawn_unclean:
            withdrawals.append(
                {"sim_time_sec": pose["time_sec"], "cells": sorted(withdrawn_unclean)}
            )
        accounting.observe_sample(
            {**pose, **snapshot, "observation_complete": True, "collision_count": 0}
        )
        added = set(verifier.visits) - before
        false_credit.update(added & occupied)
        # Counterfactual sampled footprint only; never interpolate or credit it.
        brush_only.visits.clear()
        brush_only.previous = None
        brush_only.last_footprint.clear()
        brush_only.observe(CleaningPose(**pose), frame_id=evidence["frame_id"], interpolate=False)
        overlap = set(brush_only.visits) & occupied
        if overlap:
            exposure_samples += 1
            exposed.update(overlap)
            pending_exposed.update(overlap - before)
        actual_revisits = added & awaiting_revisit
        if actual_revisits - set(brush_only.visits) or actual_revisits & occupied:
            raise ValueError("withdrawal credit is not an actual enabled free-cell revisit")
        revisited.update(actual_revisits)
        awaiting_revisit.difference_update(actual_revisits)
        previous_occupied = occupied
    replayed = verifier.result()
    if replayed != coverage or accounting.result() != temporal:
        raise ValueError("dynamic diagnostic replay differs from public verifier")
    return {
        "schema_version": "rosclaw.dynamic_credit_diagnostics.v1",
        "evidence_role": "derived_from_retained_temporal_artifact_not_new_physics_or_attestation",
        "occupancy_samples": temporal["occupancy_sample_count"],
        "fixed_denominator_cells": temporal["fixed_denominator_cells"],
        "blocked_enabled_brush_exposure_samples": exposure_samples,
        "blocked_enabled_brush_exposure_cells": sorted(exposed),
        "previously_unclean_blocked_enabled_brush_exposure_cells": sorted(pending_exposed),
        "previously_unclean_exposed_cells_actually_revisited_free": sorted(
            pending_exposed & revisited
        ),
        "false_new_credit_while_occupied_cells": sorted(false_credit),
        "withdrawals_of_previously_unclean_cells": withdrawals,
        "actual_enabled_free_revisit_cells": sorted(revisited),
        "withdrawn_cells_awaiting_actual_revisit": sorted(awaiting_revisit),
        "d6_calculation_exposure_present": bool(exposed),
        "d6_calculation_zero_false_credit": bool(exposed) and not false_credit,
        "d2_calculation_withdrawal_and_actual_revisit_present": bool(withdrawals)
        and bool(revisited),
        "requires_canonical_receipts_native_task_and_physical_stop": True,
        "physical_acceptance": "NOT_VERIFIED",
        "coverage": coverage,
    }
