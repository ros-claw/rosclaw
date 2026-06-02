"""Tests for BasicCritic — event-driven success detection and reward computation."""

import pytest

from rosclaw.core.event_bus import EventBus, Event
from rosclaw.critic.basic_critic import BasicCritic


class TestBasicCriticLifecycle:
    def test_passive_mode_no_event_bus(self, caplog):
        import logging
        critic = BasicCritic("test_bot", event_bus=None)
        with caplog.at_level(logging.INFO, logger="rosclaw.critic.basic_critic"):
            critic.initialize()
        assert "passive mode" in caplog.text
        critic.stop()

    def test_active_mode_with_event_bus(self, caplog):
        import logging
        bus = EventBus()
        critic = BasicCritic("test_bot", event_bus=bus)
        with caplog.at_level(logging.INFO, logger="rosclaw.critic.basic_critic"):
            critic.initialize()
        assert "Initialized for test_bot" in caplog.text
        critic.stop()

    def test_stop_without_event_bus(self):
        critic = BasicCritic("test_bot", event_bus=None)
        critic.initialize()
        critic.stop()
        assert critic._subscribed_topics == []

    def test_stop_unsubscribes(self):
        bus = EventBus()
        critic = BasicCritic("test_bot", event_bus=bus)
        critic.initialize()
        # Topics are normalized to rosclaw.* namespace
        assert bus.subscriber_count("rosclaw.skill.execution.complete") == 1
        assert bus.subscriber_count("rosclaw.praxis.completed") == 1
        assert bus.subscriber_count("rosclaw.praxis.failed") == 1
        assert bus.subscriber_count("rosclaw.sandbox.action.blocked") == 1
        assert bus.subscriber_count("rosclaw.safety.violation") == 1
        critic.stop()
        assert bus.subscriber_count("rosclaw.skill.execution.complete") == 0
        assert bus.subscriber_count("rosclaw.praxis.completed") == 0


class TestBasicCriticSkillComplete:
    def test_success_judgment(self):
        bus = EventBus()
        critic = BasicCritic("bot", event_bus=bus)
        critic.initialize()
        critic._on_skill_complete(Event(
            topic="skill.execution.complete",
            payload={"result": {"status": "success"}, "episode_id": "ep1"},
        ))
        assert len(critic._judgments) == 1
        assert critic._judgments[0]["status"] == "SUCCESS"
        assert critic._judgments[0]["reward"] == 1.0
        critic.stop()

    def test_failure_timeout_judgment(self):
        bus = EventBus()
        critic = BasicCritic("bot", event_bus=bus)
        critic.initialize()
        critic._on_skill_complete(Event(
            topic="skill.execution.complete",
            payload={
                "result": {"status": "failure", "error": "operation timeout"},
                "episode_id": "ep2",
            },
        ))
        assert critic._judgments[0]["status"] == "FAILED"
        assert critic._judgments[0]["reward"] == -0.3
        assert "timeout" in critic._judgments[0]["reason"]
        critic.stop()

    def test_failure_non_timeout_judgment(self):
        bus = EventBus()
        critic = BasicCritic("bot", event_bus=bus)
        critic.initialize()
        critic._on_skill_complete(Event(
            topic="skill.execution.complete",
            payload={
                "result": {"status": "failure", "error": "collision detected"},
                "episode_id": "ep3",
            },
        ))
        assert critic._judgments[0]["status"] == "FAILED"
        assert critic._judgments[0]["reward"] == -1.0
        assert critic._judgments[0]["reason"] == "collision detected"
        critic.stop()

    def test_non_dict_payload(self):
        bus = EventBus()
        critic = BasicCritic("bot", event_bus=bus)
        critic.initialize()
        critic._on_skill_complete(Event(
            topic="skill.execution.complete",
            payload="not a dict",
        ))
        # Non-dict payload -> status="unknown" -> no judgment recorded
        assert len(critic._judgments) == 0
        critic.stop()


class TestBasicCriticPraxisEvents:
    def test_praxis_completed(self):
        bus = EventBus()
        critic = BasicCritic("bot", event_bus=bus)
        critic.initialize()
        critic._on_praxis_completed(Event(
            topic="praxis.completed",
            payload={
                "outcome": {"reward": 0.8},
                "practice_id": "pr1",
            },
        ))
        assert critic._judgments[0]["status"] == "SUCCESS"
        assert critic._judgments[0]["reward"] == 0.8
        critic.stop()

    def test_praxis_failed(self):
        bus = EventBus()
        critic = BasicCritic("bot", event_bus=bus)
        critic.initialize()
        critic._on_praxis_failed(Event(
            topic="praxis.failed",
            payload={
                "outcome": {"reward": -0.7},
                "practice_id": "pr2",
                "error_log": "grasp slipped",
            },
        ))
        assert critic._judgments[0]["status"] == "FAILED"
        assert critic._judgments[0]["reward"] == -0.7
        assert critic._judgments[0]["reason"] == "grasp slipped"
        critic.stop()

    def test_praxis_non_dict_payload(self):
        bus = EventBus()
        critic = BasicCritic("bot", event_bus=bus)
        critic.initialize()
        critic._on_praxis_completed(Event(topic="praxis.completed", payload=None))
        assert critic._judgments[0]["episode_id"] == "unknown"
        critic.stop()


class TestBasicCriticFirewallAndSafety:
    def test_firewall_blocked(self):
        bus = EventBus()
        critic = BasicCritic("bot", event_bus=bus)
        critic.initialize()
        critic._on_firewall_blocked(Event(
            topic="firewall.action_blocked",
            payload={
                "episode_id": "ep4",
                "reason": "joint limit exceeded",
            },
        ))
        assert critic._judgments[0]["status"] == "BLOCKED"
        assert critic._judgments[0]["reward"] == -0.5
        assert critic._judgments[0]["reason"] == "joint limit exceeded"
        critic.stop()

    def test_safety_violation_with_list(self):
        bus = EventBus()
        critic = BasicCritic("bot", event_bus=bus)
        critic.initialize()
        critic._on_safety_violation(Event(
            topic="safety.violation",
            payload={
                "episode_id": "ep5",
                "violations": ["collision", "workspace breach"],
            },
        ))
        assert critic._judgments[0]["status"] == "BLOCKED"
        assert critic._judgments[0]["reason"] == "collision"
        critic.stop()

    def test_safety_violation_empty_list(self):
        bus = EventBus()
        critic = BasicCritic("bot", event_bus=bus)
        critic.initialize()
        critic._on_safety_violation(Event(
            topic="safety.violation",
            payload={
                "episode_id": "ep6",
                "violations": [],
            },
        ))
        assert critic._judgments[0]["reason"] == "safety violation"
        critic.stop()

    def test_safety_violation_non_dict_payload(self):
        bus = EventBus()
        critic = BasicCritic("bot", event_bus=bus)
        critic.initialize()
        critic._on_safety_violation(Event(topic="safety.violation", payload="raw"))
        assert critic._judgments[0]["episode_id"] == "unknown"
        critic.stop()


class TestBasicCriticJudge:
    def test_judge_without_event_bus(self):
        critic = BasicCritic("bot", event_bus=None)
        critic.initialize()
        critic._judge("ep1", "SUCCESS", 1.0, {"extra": True})
        assert len(critic._judgments) == 1
        # No event bus → no published events
        critic.stop()

    def test_judge_publishes_events(self):
        bus = EventBus()
        critic = BasicCritic("bot", event_bus=bus)
        critic.initialize()
        received = []
        bus.subscribe("rosclaw.critic.success.detected", lambda e: received.append(e.topic))
        bus.subscribe("rosclaw.critic.judgment", lambda e: received.append(e.topic))
        critic._judge("ep1", "SUCCESS", 1.0, {})
        assert received.count("rosclaw.critic.success.detected") == 1
        assert received.count("rosclaw.critic.judgment") == 1
        critic.stop()


class TestBasicCriticStats:
    def test_get_stats_empty(self):
        critic = BasicCritic("bot", event_bus=None)
        stats = critic.get_stats()
        assert stats == {"total": 0, "success_rate": 0.0, "avg_reward": 0.0}

    def test_get_stats_mixed(self):
        bus = EventBus()
        critic = BasicCritic("bot", event_bus=bus)
        critic.initialize()
        critic._judge("ep1", "SUCCESS", 1.0, {})
        critic._judge("ep2", "FAILED", -1.0, {})
        critic._judge("ep3", "BLOCKED", -0.5, {})
        stats = critic.get_stats()
        assert stats["total"] == 3
        assert stats["success_rate"] == pytest.approx(1 / 3)
        assert stats["avg_reward"] == pytest.approx(-0.5 / 3, abs=1e-3)
        assert stats["by_status"]["SUCCESS"] == 1
        assert stats["by_status"]["FAILED"] == 1
        assert stats["by_status"]["BLOCKED"] == 1
        critic.stop()

    def test_get_stats_all_success(self):
        bus = EventBus()
        critic = BasicCritic("bot", event_bus=bus)
        critic.initialize()
        critic._judge("ep1", "SUCCESS", 1.0, {})
        critic._judge("ep2", "SUCCESS", 1.0, {})
        stats = critic.get_stats()
        assert stats["success_rate"] == 1.0
        assert stats["avg_reward"] == 1.0
        critic.stop()
