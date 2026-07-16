"""Experiment 0 (plan §9): transport read-only verification.

Skipped everywhere except the hardware runner.  Validates device path, slave
id, baudrate, 6-actuator reads and 1000 consecutive read cycles.
"""

from __future__ import annotations

import pytest

from tests.hardware.rh56.conftest import requires_rh56_hw

pytestmark = [pytest.mark.rh56_hw, requires_rh56_hw]


def test_experiment0_read_1000_rounds():
    """Read-only: 1000 consecutive reads at >= 99.9% success, stable dims."""
    pytest.skip("pending physical RH56 bring-up (Experiment 0)")


def test_experiment0_device_and_slave_id():
    pytest.skip("pending physical RH56 bring-up (Experiment 0)")
