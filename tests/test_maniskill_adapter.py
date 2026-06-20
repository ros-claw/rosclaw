"""
test_maniskill_adapter.py — ROSClaw + ManiSkill integration tests.

Validates that ManiSkill benchmark tasks can be wrapped as ROSClaw tasks
and produce auditable Practice/Memory traces.

Supported tasks: PickCube, StackCube, PegInsertionSide

Usage:
    PYTHONPATH=src python -m pytest tests/test_maniskill_adapter.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

maniskill = pytest.importorskip(
    "mani_skill",
    reason="ManiSkill not installed. Run: pip install mani-skill",
)

pytestmark = pytest.mark.integration

from rosclaw.mcp_drivers.maniskill_adapter import ManiSkillAdapter  # noqa: E402


class TestManiSkillAdapter:
    """ManiSkill adapter integration tests."""

    @pytest.fixture(scope="class")
    def pickcube_adapter(self):
        """Create PickCube adapter (headless)."""
        adapter = ManiSkillAdapter(
            task="PickCube",
            robot="panda",
            render_mode=None,
        )
        yield adapter
        adapter.close()

    def test_adapter_import(self):
        """ManiSkill adapter module imports correctly."""
        from rosclaw.mcp_drivers.maniskill_adapter import ManiSkillAdapter
        assert ManiSkillAdapter is not None

    def test_adapter_supported_tasks(self):
        """Adapter knows supported tasks."""
        tasks = ManiSkillAdapter._SUPPORTED_TASKS
        assert "PickCube" in tasks
        assert "StackCube" in tasks
        assert "PegInsertion" in tasks

    def test_pickcube_reset(self, pickcube_adapter):
        """PickCube environment resets and returns observation."""
        result = pickcube_adapter.reset()
        assert "observation" in result
        assert "info" in result

    def test_pickcube_random_rollout(self, pickcube_adapter):
        """Random policy rollout records steps."""
        pickcube_adapter.reset()

        for _ in range(10):
            action = pickcube_adapter.sample_action()
            step_result = pickcube_adapter.step(action)
            assert "reward" in step_result
            assert "terminated" in step_result
            assert "truncated" in step_result
            if step_result["terminated"] or step_result["truncated"]:
                break

        state = pickcube_adapter.get_state()
        assert state["steps"] > 0
        assert "total_reward" in state

    def test_pickcube_episode_trace(self, pickcube_adapter):
        """Episode trace contains ROSClaw-compatible fields."""
        pickcube_adapter.reset()

        for _ in range(5):
            action = pickcube_adapter.sample_action()
            pickcube_adapter.step(action)

        trace = pickcube_adapter.record_episode(success=False)
        assert trace["task"] == "PickCube"
        assert trace["robot"] == "panda"
        assert trace["episode_id"].startswith("maniskill_PickCube_")
        assert trace["steps"] == 5
        assert "total_reward" in trace
        assert "step_traces" in trace
        assert isinstance(trace["success"], bool)

    def test_pickcube_observation_space(self, pickcube_adapter):
        """Observation space info is available for Provider routing."""
        info = pickcube_adapter.get_observation_space()
        assert info["obs_mode"] == "state_dict"
        assert info["control_mode"] == "pd_joint_pos"
        assert "action_space" in info


class TestManiSkillCrossTask:
    """Multi-task adapter creation."""

    def test_stackcube_adapter(self):
        """StackCube adapter initializes."""
        adapter = ManiSkillAdapter(task="StackCube", render_mode=None)
        result = adapter.reset()
        assert "observation" in result
        adapter.close()

    def test_peg_insertion_adapter(self):
        """PegInsertion adapter initializes."""
        adapter = ManiSkillAdapter(task="PegInsertion", render_mode=None)
        result = adapter.reset()
        assert "observation" in result
        adapter.close()
