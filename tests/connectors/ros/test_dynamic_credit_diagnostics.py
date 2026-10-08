"""Temporal diagnostics do not convert synthetic replay into Native acceptance."""

from copy import deepcopy

import pytest

from rosclaw.connectors.ros.verification.dynamic_diagnostics import analyze_dynamic_credit
from tests.connectors.ros.test_temporal_mission_runtime import evidence, sample


def test_blocked_enabled_brush_and_withdrawal_require_actual_later_revisit():
    rows = [sample(0, 0, 0.005, [0]), sample(0.1, 1, 0.015, [])]
    result = analyze_dynamic_credit(evidence(rows))
    assert result["d6_calculation_zero_false_credit"] is True
    assert result["blocked_enabled_brush_exposure_cells"] == [0]
    assert result["false_new_credit_while_occupied_cells"] == []
    assert result["withdrawn_cells_awaiting_actual_revisit"] == [0]
    assert result["d2_calculation_withdrawal_and_actual_revisit_present"] is False
    rows.append(sample(0.2, 2, 0.005, []))
    result = analyze_dynamic_credit(evidence(rows))
    assert result["actual_enabled_free_revisit_cells"] == [0]
    assert result["d2_calculation_withdrawal_and_actual_revisit_present"] is True
    assert result["physical_acceptance"] == "NOT_VERIFIED"
    assert result["requires_canonical_receipts_native_task_and_physical_stop"] is True


def test_prior_valid_credit_preserved_during_occupation_is_not_false_new_credit():
    result = analyze_dynamic_credit(
        evidence([sample(0, 0, 0.005, []), sample(0.1, 1, 0.005, [0]), sample(0.2, 2, 0.015, [])])
    )
    assert result["d6_calculation_zero_false_credit"] is True
    assert result["withdrawals_of_previously_unclean_cells"] == []
    assert result["coverage"]["coverage_ratio"] == 1


def test_no_blocked_exposure_cannot_claim_d6_even_if_all_credit_is_valid():
    result = analyze_dynamic_credit(evidence([sample(0, 0, 0.005, []), sample(0.1, 1, 0.015, [])]))
    assert (
        not result["d6_calculation_exposure_present"]
        and not result["d6_calculation_zero_false_credit"]
    )


@pytest.mark.parametrize("fault", ["missing", "time", "hash"])
def test_bad_complete_source_chain_refused(fault):
    source = deepcopy(evidence([sample(0, 0, 0.005, [0]), sample(0.1, 1, 0.015, [])]))
    if fault == "missing":
        source["occupancy_samples"].pop()
    elif fault == "time":
        source["occupancy_samples"][0]["occupancy"]["sim_time_sec"] = 0.01
    else:
        source["occupancy_samples"][0]["occupancy_hash"] = "changed"
    with pytest.raises(ValueError):
        analyze_dynamic_credit(source)
