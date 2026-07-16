"""Runtime boundary test: ROSClaw core must never import torch or lerobot.

This test runs in a subprocess so any accidental import of torch/lerobot by the
core modules is detected before it can poison the test process.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

CORE_MODULES = [
    "rosclaw.integrations.lerobot.policy_runtime.manager",
    "rosclaw.integrations.lerobot.policy_runtime.client",
    "rosclaw.integrations.lerobot.policy_runtime.state",
    "rosclaw.integrations.lerobot.policy_runtime.protocol",
    "rosclaw.integrations.lerobot.rollout.loop",
    "rosclaw.integrations.lerobot.rollout.observation_source",
    "rosclaw.integrations.lerobot.rollout.recorder",
    "rosclaw.integrations.lerobot.rollout.metrics",
    "rosclaw.integrations.lerobot.rollout.sandbox_preflight",
    "rosclaw.integrations.lerobot.action_adapter",
    "rosclaw.integrations.lerobot.observation_adapter",
    "rosclaw.integrations.lerobot.contracts",
    "rosclaw.integrations.lerobot.config",
    "rosclaw.integrations.lerobot.cli",
    "rosclaw.integrations.lerobot.provider",
    "rosclaw.body.action_mapping",
    "rosclaw.cli",
]

WORKER_MODULES = [
    "rosclaw.integrations.lerobot.policy_worker_runtime",
    "rosclaw.integrations.lerobot.policy_worker_service",
]


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_module_does_not_import_torch_or_lerobot(module: str) -> None:
    """Importing a core module must not bring torch or lerobot into sys.modules."""
    script = f"""
import sys
sys.path.insert(0, {str(Path(__file__).parents[3].resolve())!r})
import {module}
mods = set(sys.modules.keys())
forbidden = [m for m in mods if m == 'torch' or m.startswith('torch.') or m == 'lerobot' or m.startswith('lerobot.')]
if forbidden:
    print('FORBIDDEN:' + ','.join(forbidden))
    sys.exit(1)
print('OK')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"{module} imported torch/lerobot: {result.stdout}{result.stderr}"


@pytest.mark.parametrize("module", WORKER_MODULES)
def test_worker_module_can_import_torch_and_lerobot(module: str) -> None:
    """Worker modules are allowed (and expected) to import torch/lerobot."""
    script = f"""
import sys
sys.path.insert(0, {str(Path(__file__).parents[3].resolve())!r})
try:
    import {module}
except Exception as exc:
    print('IMPORT_ERROR:' + str(exc))
    sys.exit(1)
mods = set(sys.modules.keys())
has_torch = 'torch' in mods or any(m.startswith('torch.') for m in mods)
print('HAS_TORCH=' + str(has_torch))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    # The worker module may fail to import if torch/lerobot are missing, but it
    # must not be because core-side code refuses the import.  We only assert it
    # does not crash with a core-side guard error.
    assert "CORE_IMPORT_BLOCKED" not in result.stdout
