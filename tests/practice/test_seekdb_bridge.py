"""Tests for the ROSClaw -> rosclaw_practice SeekDB bridge."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("rosclaw_practice")

from rosclaw.core.types import PraxisEvent, RobotState
from rosclaw.practice.seekdb_bridge import SeekDBBridge


@pytest.fixture
def robot_state():
    return RobotState(
        timestamp=1.0,
        joint_positions=np.array([0.1, 0.2, 0.3]),
        joint_velocities=np.array([0.0, 0.0, 0.0]),
        joint_torques=np.array([0.0, 0.0, 0.0]),
        joint_names=["j1", "j2", "j3"],
    )


@pytest.fixture
def praxis_event(robot_state):
    return PraxisEvent(
        event_id="evt_123",
        event_type="success",
        timestamp=1718875200.0,
        robot_id="ur5e_lab_01",
        agent_instruction="抓取红色水杯",
        cot_trace=["检测目标", "规划路径", "执行抓取"],
        initial_state=robot_state,
        final_state=robot_state,
        trajectory=[[0.1, 0.2], [0.2, 0.3]],
        mcap_path="/data/rosclaw/mcap/evt_123/evt_123.mcap",
        error_details=None,
        duration_sec=1.5,
        metadata={"reward": 1.0},
    )


class TestSeekDBBridge:
    def test_convert_maps_fields(self, praxis_event):
        bridge = SeekDBBridge()
        converted = bridge._convert(praxis_event)

        assert converted.practice_id == "evt_123"
        assert converted.timestamp == "2024-06-20T09:20:00Z"
        assert converted.robot_id == "ur5e_lab_01"
        assert converted.cognitive_context.semantic_intent == "抓取红色水杯"
        assert converted.cognitive_context.llm_cot == "检测目标\n规划路径\n执行抓取"
        assert converted.physical_feedback.status == "SUCCESS"
        assert converted.physical_feedback.reward == 1.0
        assert converted.physical_feedback.error_log == ""
        assert converted.data_pointers.mcap_path == "/data/rosclaw/mcap/evt_123/evt_123.mcap"

    def test_convert_default_reward_and_empty_mcap(self, robot_state):
        event = PraxisEvent(
            event_id="evt_456",
            event_type="failure",
            timestamp=0.0,
            robot_id="ur5e_lab_02",
            agent_instruction="插入U盘",
            cot_trace=[],
            initial_state=robot_state,
            final_state=None,
            trajectory=[],
            mcap_path=None,
            error_details="gripper fault",
            duration_sec=0.0,
            metadata={},
        )
        bridge = SeekDBBridge()
        converted = bridge._convert(event)

        assert converted.physical_feedback.status == "FAILURE"
        assert converted.physical_feedback.reward == 0.0
        assert converted.physical_feedback.error_log == "gripper fault"
        assert converted.data_pointers.mcap_path == ""

    def test_commit_posts_to_seekdb(self, praxis_event):
        bridge = SeekDBBridge()
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_response) as mock_post:
            bridge.commit(praxis_event)

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["json"]["table"] == "praxis_events"
        assert call_kwargs["json"]["data"]["practice_id"] == "evt_123"
        assert call_kwargs["timeout"] == 2.0

    def test_commit_fallback_on_failure(self, praxis_event):
        fallback_dir = tempfile.mkdtemp()
        bridge = SeekDBBridge(fallback_dir=fallback_dir)

        with patch("requests.post", side_effect=ConnectionError(" SeekDB offline")):
            bridge.commit(praxis_event)

        fallback_files = [f for f in os.listdir(fallback_dir) if f.endswith(".json")]
        assert len(fallback_files) == 1

        with open(os.path.join(fallback_dir, fallback_files[0]), encoding="utf-8") as f:
            saved = json.load(f)

        assert saved["practice_id"] == "evt_123"
        assert saved["physical_feedback"]["status"] == "SUCCESS"

        os.remove(os.path.join(fallback_dir, fallback_files[0]))
        os.rmdir(fallback_dir)

    @pytest.mark.asyncio
    async def test_commit_async(self, praxis_event):
        bridge = SeekDBBridge()
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_response) as mock_post:
            await bridge.commit_async(praxis_event)

        mock_post.assert_called_once()

    def test_commit_with_outbox_enqueues_without_http(self, praxis_event, tmp_path):
        from rosclaw.storage.outbox import OutboxStore

        outbox = OutboxStore(db_path=str(tmp_path / "outbox.sqlite"))
        bridge = SeekDBBridge(outbox=outbox)
        try:
            with patch("requests.post") as mock_post:
                bridge.commit(praxis_event)
            mock_post.assert_not_called()
            stats = outbox.stats()
            assert stats["total"] == 1
            assert stats["pending"] == 1
        finally:
            bridge.close()
            outbox.close()

    def test_bridge_owned_worker_uses_custom_interval_and_batch_size(self, tmp_path):
        from rosclaw.storage.outbox import OutboxStore

        outbox = OutboxStore(db_path=str(tmp_path / "outbox.sqlite"))
        bridge = SeekDBBridge(
            outbox=outbox,
            outbox_interval_sec=2.5,
            outbox_batch_size=42,
        )
        try:
            assert bridge._owned_worker is not None
            assert bridge._owned_worker._interval_sec == 2.5
            assert bridge._owned_worker._batch_size == 42
        finally:
            bridge.close()
            outbox.close()
