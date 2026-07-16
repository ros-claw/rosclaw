"""RH56 hardware tests (plan §14).

Default CI must NOT run these.  They only run on the hardware runner:

    ROSCLAW_RH56_HW=1 pytest tests/hardware/rh56/ -m rh56_hw -q

Execution tests additionally require an explicit operator acknowledgement:

    ROSCLAW_RH56_EXECUTE_ACK=YES
"""

from __future__ import annotations

import os

import pytest

RH56_HW = os.environ.get("ROSCLAW_RH56_HW") == "1"
RH56_EXECUTE_ACK = os.environ.get("ROSCLAW_RH56_EXECUTE_ACK") == "YES"

requires_rh56_hw = pytest.mark.skipif(not RH56_HW, reason="ROSCLAW_RH56_HW != 1")
requires_execute_ack = pytest.mark.skipif(
    not (RH56_HW and RH56_EXECUTE_ACK),
    reason="requires ROSCLAW_RH56_HW=1 and ROSCLAW_RH56_EXECUTE_ACK=YES",
)
