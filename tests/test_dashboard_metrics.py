"""Tests for DashboardMetrics sense aggregation."""

from __future__ import annotations

import pytest

from rosclaw.core.event_topics import EventTopics
from rosclaw.dashboard.metrics import DashboardMetrics


class TestDashboardSenseMetrics:
    def test_get_body_sense_stats_before_record(self):
        metrics = DashboardMetrics()
        stats = metrics.get_body_sense_stats()
        assert stats["available"] is False
        assert stats["overall_status"] == "unknown"
        assert stats["blocked_capabilities"] == []
        assert stats["degraded_capabilities"] == []

    def test_record_body_sense_and_stats(self):
        metrics = DashboardMetrics()
        sense = {
            "robot_id": "g1_lab_01",
            "overall_status": "not_ready",
            "blocked_capabilities": ["kick_ball"],
            "degraded_capabilities": ["walk_slow"],
            "risk_summary": {"thermal_risk": "high", "battery_risk": "low"},
            "recommended_actions": ["cooldown"],
            "main_reasons": ["joint_overheat"],
            "timestamp": 1.0,
        }
        metrics.record_body_sense(sense)

        stats = metrics.get_body_sense_stats()
        assert stats["available"] is True
        assert stats["robot_id"] == "g1_lab_01"
        assert stats["overall_status"] == "not_ready"
        assert stats["blocked_capabilities"] == ["kick_ball"]
        assert stats["degraded_capabilities"] == ["walk_slow"]
        assert stats["risk_summary"]["thermal_risk"] == "high"
        assert stats["recommended_actions"] == ["cooldown"]
        assert stats["main_reasons"] == ["joint_overheat"]
        assert stats["history_count"] == 1

    def test_record_body_sense_ignores_non_dict(self):
        metrics = DashboardMetrics()
        metrics.record_body_sense(None)  # type: ignore[arg-type]
        metrics.record_body_sense("not-a-dict")  # type: ignore[arg-type]
        assert metrics.get_body_sense_stats()["available"] is False

    def test_body_sense_history_trimming(self):
        metrics = DashboardMetrics(max_history=3)
        for i in range(5):
            metrics.record_body_sense({"overall_status": "ready", "timestamp": float(i)})
        assert len(metrics._body_sense_history) == 3
        stats = metrics.get_body_sense_stats()
        assert stats["history_count"] == 3
        assert stats["timestamp"] == 4.0

    def test_snapshot_contains_sense(self):
        metrics = DashboardMetrics()
        metrics.record_body_sense({"overall_status": "ready"})
        snapshot = metrics.snapshot()
        assert "sense" in snapshot
        assert snapshot["sense"]["available"] is True
        assert snapshot["sense"]["overall_status"] == "ready"

    def test_known_topics_include_sense(self):
        metrics = DashboardMetrics()
        for topic in [
            EventTopics.SENSE_STATE_UPDATED,
            EventTopics.SENSE_BODY_UPDATED,
            EventTopics.SENSE_EVENT_DETECTED,
            EventTopics.SENSE_READINESS_UPDATED,
            EventTopics.SENSE_CAPABILITY_BLOCKED,
            EventTopics.SENSE_CAPABILITY_DEGRADED,
        ]:
            assert topic in metrics._KNOWN_TOPICS
