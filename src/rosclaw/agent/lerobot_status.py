"""LeRobot integration detection for runtime status and context snapshots.

Detection contract (rosclaw_lerobot_终稿 §5.2):

1. Never import torch/lerobot into the core process — state comes from the
   stored integration config and the presence of the reference policy/plugin
   artifacts, not from importing the frameworks.
2. Never raise: any failure degrades to ``configured: false``.
3. Never leak: no tokens, API keys, home paths, permit ids, or private repo
   credentials in the returned payload.

States: ``ready`` | ``degraded`` | ``not_configured`` | ``runtime_missing`` |
``policy_missing``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rosclaw.body.rh56.resources import rh56_reference_policy_path
from rosclaw.firstboot.workspace import get_rosclaw_home

_REFERENCE_POLICY_DIR = Path("policies") / "rh56_reference_policy_v1"
_PLUGIN_MODULE = "lerobot_policy_rosclaw_rh56"


def detect_lerobot_integration() -> dict[str, Any]:
    """Detect LeRobot bridge state without importing torch/lerobot."""
    try:
        return _detect()
    except Exception:  # noqa: BLE001
        return {"configured": False, "state": "not_configured"}


def _detect() -> dict[str, Any]:
    from rosclaw.integrations.lerobot.config import get_configured_lerobot_runtime

    runtime = get_configured_lerobot_runtime()
    base: dict[str, Any] = {
        "reference_policy": "rosclaw_rh56_reference",
        "supported_bodies": ["inspire_rh56_left", "inspire_rh56_right"],
        "supported_modes": [
            "proposal_only",
            "shadow",
            "single_step_receding_horizon",
        ],
        "agent_action_entry": "mcp.request_action",
        "direct_execution_allowed": False,
    }
    if not runtime or not runtime.get("python_executable"):
        return {"configured": False, "state": "not_configured", **base}

    python_exe = str(runtime.get("python_executable", ""))
    state = str(runtime.get("state", "unknown"))
    lerobot_version = runtime.get("lerobot_version")

    # Prefer the bundled/source artifact, then a separately downloaded cache.
    try:
        bundled_policy = rh56_reference_policy_path()
    except FileNotFoundError:
        bundled_policy = None
    cache_policy = get_rosclaw_home() / "cache" / "lerobot" / _REFERENCE_POLICY_DIR
    policy_dir = bundled_policy if bundled_policy is not None else cache_policy
    policy_ok = (policy_dir / "policy_contract.yaml").exists()

    if not Path(python_exe).exists():
        state = "runtime_missing"
    elif not policy_ok:
        state = "policy_missing"
    elif state not in ("ready", "degraded"):
        state = "degraded"

    return {
        "configured": True,
        "bridge_version": "1.0.1",
        "state": state,
        "lerobot_version": lerobot_version,
        "worker_python": runtime.get("python_version"),
        "reference_policy_present": policy_ok,
        **base,
    }
