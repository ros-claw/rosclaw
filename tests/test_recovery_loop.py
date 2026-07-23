"""Sprint 9 — RecoveryLoop tests.

Covers:
- Retry intent recording on recovery_hint event
- Retry success updates rule efficacy and stores success pattern
- Retry failure updates rule efficacy and stores escalation failure
"""

import pytest

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.how.engine import HeuristicEngine
from rosclaw.how.recovery_loop import RecoveryLoop
from rosclaw.memory.interface import MemoryInterface
from rosclaw.sandbox.backends import ReplayReport
from rosclaw.sandbox.evidence import (
    SimulationEvidenceVerification,
    verify_simulation_receipt,
)


def _verified(_receipt: dict) -> SimulationEvidenceVerification:
    replay = ReplayReport(True, True, True, True, 0.0, "strict_replay_verified")
    return SimulationEvidenceVerification(True, replay)


@pytest.fixture
def loop():
    bus = EventBus()
    mem = MemoryInterface("test_bot", event_bus=bus)
    mem.initialize()
    he = HeuristicEngine(mem.seekdb_client)
    rl = RecoveryLoop(bus, mem, he, receipt_verifier=_verified)
    rl.subscribe()
    yield rl
    rl.unsubscribe()
    mem.stop()


class TestRecoveryLoop:
    def test_records_retry_intent(self, loop):
        loop._bus.publish(
            Event(
                topic="rosclaw.how.recovery_hint.generated",
                payload={
                    "request_id": "req1",
                    "failure_type": "gripper_slip",
                    "retry_plan": {
                        "rule_id": "rule_grip",
                        "parameter_patch": {"gripper_force_offset": 0.15},
                        "max_retries": 3,
                    },
                },
                source="test",
            )
        )
        rows = loop._memory.seekdb_client.query("retries", filters={"id": "req1"})
        assert len(rows) == 1
        assert rows[0]["failure_type"] == "gripper_slip"
        assert rows[0]["status"] == "pending"

    @pytest.mark.parametrize("max_retries", [True, 0, 11, "3"])
    def test_rejects_malformed_retry_hint(self, loop, max_retries):
        loop._on_recovery_hint(
            Event(
                topic="rosclaw.how.recovery_hint.generated",
                payload={
                    "request_id": "req-invalid",
                    "failure_type": "collision",
                    "retry_plan": {"max_retries": max_retries},
                },
                source="test",
            )
        )
        assert not loop._memory.seekdb_client.query("retries", filters={"id": "req-invalid"})

    def test_retry_success_updates_rule(self, loop):
        # Seed a rule first
        loop._how._seekdb.insert(
            "heuristic_rules",
            {
                "id": "rule_grip",
                "condition": "gripper_slip",
                "action": "increase force",
                "priority": 2,
                "success_count": 0,
                "failure_count": 0,
            },
        )
        loop._how._rule_cache = {
            "rule_grip": {
                "id": "rule_grip",
                "condition": "gripper_slip",
                "action": "increase force",
                "priority": 2,
                "success_count": 0,
                "failure_count": 0,
            }
        }
        loop._how._cache_valid = True

        # Record intent
        loop._on_recovery_hint(
            Event(
                topic="rosclaw.how.recovery_hint.generated",
                payload={
                    "request_id": "req2",
                    "failure_type": "gripper_slip",
                    "retry_plan": {
                        "rule_id": "rule_grip",
                        "parameter_patch": {"force": 0.15},
                        "max_retries": 3,
                    },
                },
                source="test",
            )
        )

        # Simulate retry success
        loop._bus.publish(
            Event(
                topic="rosclaw.sandbox.episode.succeeded",
                payload={
                    "request_id": "req2",
                    "episode_id": "ep2",
                    "skill_id": "grasp",
                    "duration_sec": 1.5,
                    "physics_executed": True,
                    "receipt_verified": True,
                    "data_quality_pass": True,
                    "evidence_domain": "SIMULATION",
                    "simulation_receipt": {},
                },
                source="test",
            )
        )

        # Rule should have success_count incremented
        rule = loop._how._seekdb.query("heuristic_rules", filters={"id": "rule_grip"}, limit=1)
        assert rule[0]["success_count"] == 1

        # Success pattern should exist
        sp = loop._memory.seekdb_client.query("success_patterns", filters={"id": "sp_retry_req2"})
        assert len(sp) == 1

    def test_retry_failure_escalates(self, loop):
        loop._how._seekdb.insert(
            "heuristic_rules",
            {
                "id": "rule_col",
                "condition": "collision",
                "action": "replan",
                "priority": 2,
                "success_count": 0,
                "failure_count": 0,
            },
        )
        loop._how._rule_cache = {
            "rule_col": {
                "id": "rule_col",
                "condition": "collision",
                "action": "replan",
                "priority": 2,
                "success_count": 0,
                "failure_count": 0,
            }
        }
        loop._how._cache_valid = True

        loop._on_recovery_hint(
            Event(
                topic="rosclaw.how.recovery_hint.generated",
                payload={
                    "request_id": "req3",
                    "failure_type": "collision",
                    "retry_plan": {
                        "rule_id": "rule_col",
                        "parameter_patch": {"clearance": 0.05},
                        "max_retries": 1,
                    },
                },
                source="test",
            )
        )

        # Simulate retry failure (max_retries=1 so it escalates)
        loop._bus.publish(
            Event(
                topic="rosclaw.sandbox.episode.failed",
                payload={
                    "request_id": "req3",
                    "episode_id": "ep3",
                    "physics_executed": True,
                    "receipt_verified": True,
                    "data_quality_pass": True,
                    "evidence_domain": "SIMULATION",
                    "simulation_receipt": {},
                },
                source="test",
            )
        )

        # Rule should have failure_count incremented
        rule = loop._how._seekdb.query("heuristic_rules", filters={"id": "rule_col"}, limit=1)
        assert rule[0]["failure_count"] == 1

        # Escalated failure should exist
        failures = loop._memory.seekdb_client.query("failures", filters={"id": "retry_failed_req3"})
        assert len(failures) == 1
        assert "Escalate to human operator" in failures[0]["recovery_hint"]

    def test_fabricated_verification_flags_do_not_poison_learning(self, loop):
        loop._how._seekdb.insert(
            "heuristic_rules",
            {
                "id": "rule_forged",
                "condition": "forged",
                "action": "ignore",
                "priority": 1,
                "success_count": 0,
                "failure_count": 0,
            },
        )
        loop._on_recovery_hint(
            Event(
                topic="rosclaw.how.recovery_hint.generated",
                payload={
                    "request_id": "req-forged",
                    "failure_type": "forged",
                    "retry_plan": {"rule_id": "rule_forged", "max_retries": 1},
                },
                source="test",
            )
        )
        loop._receipt_verifier = verify_simulation_receipt

        loop._bus.publish(
            Event(
                topic="rosclaw.sandbox.episode.succeeded",
                payload={
                    "request_id": "req-forged",
                    "episode_id": "ep-forged",
                    "physics_executed": True,
                    "receipt_verified": True,
                    "data_quality_pass": True,
                    "evidence_domain": "SIMULATION",
                    "simulation_receipt": {},
                },
                source="untrusted",
            )
        )

        rule = loop._how._seekdb.query("heuristic_rules", filters={"id": "rule_forged"}, limit=1)[0]
        retry = loop._memory.seekdb_client.query("retries", filters={"id": "req-forged"}, limit=1)[
            0
        ]
        assert rule["success_count"] == 0
        assert retry["status"] == "pending"
        assert not loop._memory.seekdb_client.query(
            "success_patterns", filters={"id": "sp_retry_req-forged"}
        )
