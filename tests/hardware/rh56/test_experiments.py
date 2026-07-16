"""Experiments 1-8 (plan §9): real shadow, no-op, micro motion, gestures,
OK contact, repeatability, fault injection.

All skipped except on the hardware runner with the execute acknowledgement.
"""

from __future__ import annotations

import pytest

from tests.hardware.rh56.conftest import requires_execute_ack, requires_rh56_hw

pytestmark = [pytest.mark.rh56_hw, requires_rh56_hw]


@pytest.mark.rh56_shadow
def test_experiment1_real_shadow_1000_steps():
    pytest.skip("pending physical RH56 bring-up (Experiment 1)")


@requires_execute_ack
@pytest.mark.rh56_execute
def test_experiment2_noop_execution_20_rounds():
    pytest.skip("pending physical RH56 bring-up (Experiment 2)")


@requires_execute_ack
@pytest.mark.rh56_execute
def test_experiment3_micro_index_motion_10_cycles():
    pytest.skip("pending physical RH56 bring-up (Experiment 3)")


@requires_execute_ack
@pytest.mark.rh56_execute
def test_experiment4_50_raw_motion_10_cycles():
    pytest.skip("pending physical RH56 bring-up (Experiment 4)")


@requires_execute_ack
@pytest.mark.rh56_execute
def test_experiment5_multi_finger_non_contact():
    pytest.skip("pending physical RH56 bring-up (Experiment 5)")


@requires_execute_ack
@pytest.mark.rh56_execute
def test_experiment6_ok_contact_validated_pose():
    pytest.skip("pending physical RH56 bring-up (Experiment 6)")


@requires_execute_ack
@pytest.mark.rh56_execute
def test_experiment7_ok_repeatability_10_rounds():
    pytest.skip("pending physical RH56 bring-up (Experiment 7)")


@requires_execute_ack
@pytest.mark.rh56_execute
def test_experiment8_fault_injection_matrix():
    pytest.skip("pending physical RH56 bring-up (Experiment 8)")
