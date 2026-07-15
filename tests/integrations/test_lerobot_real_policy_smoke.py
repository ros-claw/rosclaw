"""Optional real-policy smoke test.

Run only when a local LeRobot policy directory is provided via the
``ROSCLAW_LEROBOT_SMOKE_POLICY`` environment variable and a real LeRobot
runtime is configured in the user's ROSClaw home.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rosclaw.integrations.lerobot.smoke_policy import (
    SmokePolicyOptions,
    run_smoke_policy_sync,
)

_REAL_POLICY = os.environ.get("ROSCLAW_LEROBOT_SMOKE_POLICY")
_REAL_ROSCLAW_HOME = Path.home() / ".rosclaw"


@pytest.mark.skipif(
    not _REAL_POLICY,
    reason="Set ROSCLAW_LEROBOT_SMOKE_POLICY to a local policy directory",
)
@pytest.mark.skipif(
    not (_REAL_ROSCLAW_HOME / "integrations" / "lerobot.yaml").exists(),
    reason="No LeRobot runtime configured in ~/.rosclaw",
)
def test_real_policy_smoke(monkeypatch):
    """End-to-end smoke test against a real LeRobot policy."""
    # The session-scoped conftest uses a temp ROSCLAW_HOME; for this real test
    # we need the user's actual ROSClaw home where the LeRobot runtime is registered.
    monkeypatch.setenv("ROSCLAW_HOME", str(_REAL_ROSCLAW_HOME))

    options = SmokePolicyOptions(
        policy_path=_REAL_POLICY,
        device="cpu",
        allow_network=False,
        timeout_sec=300,
    )
    report = run_smoke_policy_sync(options)

    assert report.status == "ok", report.error
    assert report.stages["inspect"] == "ok"
    assert report.stages["load_test"] == "ok"
    assert report.stages["infer"] == "ok"
    assert report.action_proposal is not None
    assert report.action_proposal["not_executed"] is True
    assert report.action_proposal["requires_sandbox"] is True
    assert report.action_proposal["executable"] is False
    assert report.action_proposal.get("shape") in ([14], [100, 14])
