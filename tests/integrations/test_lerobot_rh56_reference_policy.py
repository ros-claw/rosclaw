"""P5-A reference policy integration test (real persistent worker).

Loads the LeRobot worker's bundled RH56 reference policy into the persistent runtime,
verifies the authoritative action contract metadata, runs WARMUP and several
task-mode inferences, and checks the resulting proposals carry
``semantic_source=explicit_policy_contract`` and ``authoritative=True``.
"""

from __future__ import annotations

import pytest

from rosclaw.body.rh56.resources import rh56_reference_policy_path
from rosclaw.integrations.lerobot.action_adapter import adapt_action_to_proposal
from rosclaw.integrations.lerobot.policy_runtime.manager import PersistentRuntimeManager

POLICY_PATH = rh56_reference_policy_path()

pytestmark = pytest.mark.skipif(
    not (POLICY_PATH / "config.json").exists(),
    reason="rh56 reference policy artifact missing",
)

EXPECTED_NAMES = ["little", "ring", "middle", "index", "thumb", "thumb_rot"]


@pytest.fixture
def runtime(real_lerobot_runtime_config):
    from rosclaw.integrations.lerobot.config import get_configured_lerobot_runtime

    configured = get_configured_lerobot_runtime()
    if not configured or not configured.get("python_executable"):
        pytest.skip("LeRobot runtime not configured")
    manager = PersistentRuntimeManager(
        python_executable=str(configured["python_executable"]),
        policy_path=str(POLICY_PATH),
        device="cpu",
    )
    state = manager.start()
    if state.state != "ready":
        pytest.skip(f"worker failed to start: {state.error}")
    yield manager
    manager.stop()


def _load(runtime: PersistentRuntimeManager) -> dict:
    response = runtime.call(
        "LOAD_POLICY",
        {"policy_path": str(POLICY_PATH), "device": "cpu", "allow_network": False},
    )
    assert response.get("status") == "ok", response
    return response["policy_metadata"]


def test_reference_policy_authoritative_metadata(runtime) -> None:
    metadata = _load(runtime)
    action = metadata["output_features"]["action"]
    assert action["names"] == EXPECTED_NAMES
    assert action["representation"] == "joint_position"
    assert action["unit"] == "raw_device_unit"
    assert metadata.get("action_contract_hash", "").startswith("sha256:")
    contract = metadata["extra"]["action_contract"]
    assert contract["policy_id"] == "rh56_reference_policy_v1"
    assert contract["authoritative"] is True


def test_reference_policy_warmup_and_infer(runtime) -> None:
    metadata = _load(runtime)
    warmup = runtime.call(
        "WARMUP",
        {"observation": {"observation.state": [1000.0] * 6, "task": "open_hand"}, "iterations": 1},
    )
    assert warmup.get("status") == "ok", warmup

    runtime.call("CREATE_SESSION", {"session_id": "s_ref"})
    try:
        for step, (task, check) in enumerate(
            [
                ("hold_current", lambda v: v[0] == 980.0),
                ("open_hand", lambda v: all(x == 1000.0 for x in v)),
                ("micro_index_flex", lambda v: v[3] <= 1000.0),
            ]
        ):
            response = runtime.call(
                "INFER",
                {
                    "session_id": "s_ref",
                    "observation": {
                        "observation.state": [980.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
                        "task": task,
                    },
                    "step_index": step,
                },
            )
            assert response.get("status") == "ok", (task, response)
            processed = response["processed_action"]
            flat = [x for row in processed["values"] for x in (row if isinstance(row, list) else [row])]
            assert len(flat) == 6
            assert check(flat), (task, flat)

            proposal = adapt_action_to_proposal(
                processed,
                policy_path=str(POLICY_PATH),
                policy_metadata=metadata,
                session_id="s_ref",
                step_index=step,
            )
            assert proposal["representation"] == "joint_position"
            assert proposal["action"]["units"] == "raw_device_unit"
            assert proposal["action"]["names"] == EXPECTED_NAMES
            assert proposal["semantic_source"] == "explicit_policy_contract"
            assert proposal["authoritative"] is True
            assert "error_code" not in proposal["safety"]
    finally:
        runtime.call("CLOSE_SESSION", {"session_id": "s_ref"})
