"""Tests that provider and sandbox events are linked in a practice episode."""

from __future__ import annotations

import json
from types import SimpleNamespace

from rosclaw.cli import cmd_practice_run


class TestPracticeProviderSandboxLinkage:
    """Verify provider reasoning is grounded in the captured frame and sandbox allows it."""

    def test_provider_event_references_captured_frame(
        self,
        linked_realsense_workspace,
        fake_realsense_skill,
        tmp_path,
    ):
        output_root = tmp_path / "episode"
        args = SimpleNamespace(
            robot="d405_lab_01",
            robot_type=None,
            task=None,
            skill="realsense_capture_rgbd",
            provider="cosmos-reason2-lan",
            capability="vlm.risk_assessment",
            output_root=str(output_root),
            data_root=None,
            workspace=str(linked_realsense_workspace),
            json=False,
        )

        assert cmd_practice_run(args) == 0

        session_dir = next((output_root / "sessions").iterdir())
        events = [
            json.loads(line)
            for line in (session_dir / "raw" / "events.jsonl").read_text().strip().splitlines()
        ]

        camera = next(ev for ev in events if ev["source"] == "camera")
        sandbox = next(ev for ev in events if ev["source"] == "sandbox")

        rgb_ref = camera["payload"]["rgb_ref"]
        assert rgb_ref.endswith("color_000001.png")
        provider = next(
            ev
            for ev in events
            if ev["source"] == "provider" and ev["event_type"] == "provider.result"
        )
        assert provider["payload"]["input_summary"]["image"] == rgb_ref
        assert provider["payload"]["provider_id"] == "cosmos-reason2-lan"

        # Sandbox must allow the perception-only action and reference the skill.
        assert sandbox["payload"]["decision"] == "ALLOW"
        assert sandbox["payload"]["action_id"] == "realsense_capture_rgbd"
        assert sandbox["payload"]["requested_action"]["type"] == "provider_reasoning"
        assert "perception-only" in sandbox["payload"]["reason"].lower()

    def test_sandbox_blocks_actuator_action_for_perception_only_body(
        self,
        linked_realsense_workspace,
        tmp_path,
    ):
        """Sanity check that the underlying policy would block motion commands."""
        from rosclaw.cli import _evaluate_sandbox_policy

        result = _evaluate_sandbox_policy(
            "realsense_d405",
            {"type": "move_base"},
        )
        assert result["decision"] == "BLOCK"
        assert "actuator" in result["reason"].lower()
