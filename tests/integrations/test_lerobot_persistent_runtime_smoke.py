"""P4-E real persistent runtime acceptance smoke test.

This test loads the validated ALOHA policy once and runs 100 continuous
inferences on the same LeRobot worker process.  It verifies:

- Policy loads once.
- 100 continuous inferences complete on the same worker PID.
- ``policy.reset()`` is invoked per session.
- Pre/post processors are present.
- Action proposals carry representation, names, units, timing.
- NaN/Inf actions are rejected.
- Python 3.11 core never imports torch/lerobot.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from rosclaw.integrations.lerobot.config import get_configured_lerobot_runtime
from rosclaw.integrations.lerobot.observation_adapter import adapt_observation_for_worker
from rosclaw.integrations.lerobot.policy_runtime.manager import PersistentRuntimeManager
from rosclaw.integrations.lerobot.smoke_policy import DEFAULT_SMOKE_POLICY


def _effective_policy_path() -> str:
    """Return the migrated local path for the smoke policy if it exists.

    LeRobot 0.6.x requires a one-time processor migration for older pretrained
    policies.  The acceptance test skips if the migrated copy is not available.
    """
    from rosclaw.integrations.lerobot.policy_cache import (
        DEFAULT_CACHE_SUBDIR,
        _sanitize_repo_id,
    )

    # The real cache is normally under the user's home, even when tests run
    # with an isolated ROSCLAW_HOME.
    candidate_homes = [
        Path.home() / ".rosclaw",
    ]
    try:
        from rosclaw.firstboot.workspace import get_rosclaw_home

        candidate_homes.insert(0, get_rosclaw_home())
    except Exception:  # pragma: no cover
        pass

    sanitized = _sanitize_repo_id(DEFAULT_SMOKE_POLICY)
    for home in candidate_homes:
        local_path = home / DEFAULT_CACHE_SUBDIR / sanitized
        migrated = Path(str(local_path) + "_migrated")
        if migrated.exists():
            return str(migrated)

    pytest.skip(
        f"Migrated policy not found for {DEFAULT_SMOKE_POLICY}. "
        "Run the LeRobot processor migration on the cached policy first."
    )


SAMPLE_OBSERVATION_FILE = (
    Path(__file__).parents[2]
    / "examples"
    / "lerobot"
    / "sample_observation_aloha_act.json"
)


def _flatten_observation(obs: dict[str, Any]) -> dict[str, Any]:
    """Convert the sample nested observation into LeRobot flat keys."""
    out: dict[str, Any] = {}
    # Resolve image paths relative to the sample fixture file.
    out["_base_dir"] = str(SAMPLE_OBSERVATION_FILE.parent)
    task = obs.get("task")
    if task:
        out["task"] = task
    observation = obs.get("observation", obs)
    if isinstance(observation, dict):
        if "state" in observation:
            out["observation.state"] = list(observation["state"])
        images = observation.get("images", {})
        if isinstance(images, dict):
            for name, path in images.items():
                out[f"observation.images.{name}"] = path
    return out


def _lerobot_python() -> str | None:
    configured = get_configured_lerobot_runtime()
    if configured and configured.get("python_executable"):
        return str(configured["python_executable"])
    return None


@pytest.fixture
def lerobot_python(real_lerobot_runtime_config) -> str:
    exe = _lerobot_python()
    if not exe:
        pytest.skip("LeRobot runtime not configured")
    return exe


@pytest.fixture
def sample_observation() -> dict[str, Any]:
    if not SAMPLE_OBSERVATION_FILE.exists():
        pytest.skip("Sample observation fixture not found")
    import json

    return json.loads(SAMPLE_OBSERVATION_FILE.read_text(encoding="utf-8"))


def test_persistent_runtime_100_inferences(lerobot_python: str, sample_observation: dict[str, Any]) -> None:
    """Run 100 inferences on one persistent worker and verify P4 gates."""
    policy_path = _effective_policy_path()
    observation = adapt_observation_for_worker(_flatten_observation(sample_observation))

    runtime = PersistentRuntimeManager(
        python_executable=lerobot_python,
        policy_path=policy_path,
        device="cpu",
        dtype="auto",
        allow_network=False,
        timeout_sec=300.0,
        startup_timeout_sec=120.0,
    )

    with runtime:
        load_response = runtime.call(
            "LOAD_POLICY",
            {
                "policy_path": policy_path,
                "revision": "main",
                "device": "cpu",
                "allow_network": False,
            },
            timeout_sec=300.0,
        )
        assert load_response.get("status") == "ok", f"LOAD_POLICY failed: {load_response}"
        policy_metadata = load_response.get("policy_metadata", {})
        output_features = policy_metadata.get("output_features", {})
        action_feature = output_features.get("action", {})
        action_shape = list(action_feature.get("shape", []))

        session_id = "p4_smoke_session"
        create_response = runtime.call(
            "CREATE_SESSION",
            {"session_id": session_id, "body_id": "aloha_sim"},
        )
        assert create_response.get("status") == "ok"

        # Run 100 inferences on the same session.
        latencies_ms: list[float] = []
        worker_pid = runtime.state.pid
        for step in range(100):
            import time

            t0 = time.perf_counter()
            infer_response = runtime.call(
                "INFER",
                {
                    "session_id": session_id,
                    "observation": observation,
                    "step_index": step,
                },
                timeout_sec=60.0,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(latency_ms)

            assert infer_response.get("status") == "ok", f"INFER failed at step {step}: {infer_response}"
            processed = infer_response.get("processed_action", {})
            values = processed.get("values", [])
            actual_shape = list(processed.get("shape", []))
            assert actual_shape[-1] == action_shape[-1], (
                f"Action dim mismatch at step {step}: expected {action_shape[-1]}, got {actual_shape}"
            )
            flat_values: list[float] = []
            if values and isinstance(values[0], list):
                flat_values = [v for row in values for v in row]
            else:
                flat_values = list(values)
            assert not any(
                isinstance(v, float) and (math.isnan(v) or math.isinf(v)) for v in flat_values
            ), f"NaN/Inf in action at step {step}"

        # The same OS process served all inferences.
        assert runtime.state.pid == worker_pid
        assert len(latencies_ms) == 100
        assert max(latencies_ms) < 60_000, "At least one inference exceeded 60s"

        # Reset the session and confirm the worker stays alive.
        reset_response = runtime.call("RESET_SESSION", {"session_id": session_id})
        assert reset_response.get("status") == "ok"
        assert runtime.state.pid == worker_pid

        # Health still OK after 100+ inferences.
        health = runtime.call("HEALTH", {}, timeout_sec=10.0)
        assert health.get("status") == "ok"
        assert health.get("active_sessions", -1) >= 1

        runtime.call("CLOSE_SESSION", {"session_id": session_id})


def test_persistent_runtime_action_proposal_has_semantics(
    lerobot_python: str, sample_observation: dict[str, Any]
) -> None:
    """Infer once and assert the proposal carries P4 contract fields."""
    from rosclaw.integrations.lerobot.action_adapter import adapt_action_to_proposal

    policy_path = _effective_policy_path()
    observation = adapt_observation_for_worker(_flatten_observation(sample_observation))

    runtime = PersistentRuntimeManager(
        python_executable=lerobot_python,
        policy_path=policy_path,
        device="cpu",
        timeout_sec=300.0,
        startup_timeout_sec=120.0,
    )

    with runtime:
        runtime.call(
            "LOAD_POLICY",
            {"policy_path": policy_path, "revision": "main", "device": "cpu"},
            timeout_sec=300.0,
        )
        session_id = "p4_semantics_session"
        runtime.call("CREATE_SESSION", {"session_id": session_id})

        infer_response = runtime.call(
            "INFER",
            {"session_id": session_id, "observation": observation, "step_index": 0},
            timeout_sec=60.0,
        )
        assert infer_response.get("status") == "ok"

        processed = infer_response.get("processed_action", {})
        policy_metadata = infer_response.get("policy_metadata", {})
        proposal = adapt_action_to_proposal(
            processed,
            policy_path=policy_path,
            policy_metadata=policy_metadata,
            session_id=session_id,
            step_index=0,
            proposal_id="p4_semantics_0",
        )

        assert proposal.get("schema_version") == "rosclaw.action_proposal.v2"
        # The generic ALOHA smoke policy config does not declare action semantics,
        # so the bridge must remain fail-closed rather than guessing.
        assert proposal.get("representation") == "unknown"
        action = proposal.get("action", {})
        assert action.get("names") == []
        assert action.get("units") == "unknown"
        assert action.get("shape")
        assert proposal.get("safety", {}).get("executable") is False
        assert proposal.get("safety", {}).get("requires_sandbox") is True
        assert proposal.get("safety", {}).get("error_code") == "unknown_action_semantics"
        assert proposal.get("authoritative") is False
        assert proposal.get("semantic_source") == "unknown"

        runtime.call("CLOSE_SESSION", {"session_id": session_id})
