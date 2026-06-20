"""Tests for sense-aware context adapters (Phase 5)."""

from __future__ import annotations

import pytest

from rosclaw.core.event_bus import EventBus
from rosclaw.sense.adapters.auto_context import AutoContextAdapter
from rosclaw.sense.adapters.how_context import HowContextAdapter
from rosclaw.sense.adapters.memory_writer import MemoryWriterAdapter
from rosclaw.sense.adapters.practice_writer import PracticeWriterAdapter
from rosclaw.sense.adapters.sandbox_context import SandboxContextAdapter
from rosclaw.sense.adapters.skill_requirements import SkillRequirementsAdapter
from rosclaw.sense.config import SenseConfig
from rosclaw.sense.runtime import SenseRuntime


@pytest.fixture
def sense_runtime():
    bus = EventBus()
    cfg = SenseConfig(
        robot_id="g1_lab_01", collector="mock", extra={"scenario": "kick_not_ready"}
    )
    runtime = SenseRuntime(cfg, event_bus=bus, robot_id="g1_lab_01")
    runtime.initialize()
    runtime.tick()
    yield runtime
    runtime.stop()


@pytest.fixture
def ready_runtime():
    bus = EventBus()
    cfg = SenseConfig(
        robot_id="g1_lab_01", collector="mock", extra={"scenario": "normal"}
    )
    runtime = SenseRuntime(cfg, event_bus=bus, robot_id="g1_lab_01")
    runtime.initialize()
    runtime.tick()
    yield runtime
    runtime.stop()


class TestSkillRequirementsAdapter:
    def test_no_runtime_returns_context(self):
        adapter = SkillRequirementsAdapter(sense_runtime=None)
        ctx = {"task": "kick_ball"}
        assert adapter.apply(ctx) is ctx

    def test_enriches_with_readiness(self, sense_runtime):
        adapter = SkillRequirementsAdapter(sense_runtime=sense_runtime)
        ctx = {"task": "kick_ball"}
        out = adapter.apply(ctx)
        assert out is not ctx
        assert "body_sense_check" in out
        assert out["body_sense_check"]["task"] == "kick_ball"
        assert out["body_sense_check"]["status"] == "not_ready"
        assert any("temperature" in r.lower() for r in out["body_sense_check"]["reasons"])

    def test_no_task_returns_context(self, sense_runtime):
        adapter = SkillRequirementsAdapter(sense_runtime=sense_runtime)
        ctx = {"requirements": {}}
        assert adapter.apply(ctx) is ctx


class TestSandboxContextAdapter:
    def test_no_runtime_returns_context(self):
        adapter = SandboxContextAdapter(sense_runtime=None)
        ctx = {"action": "step"}
        assert adapter.apply(ctx) is ctx

    def test_adds_body_sense_snapshot(self, sense_runtime):
        adapter = SandboxContextAdapter(sense_runtime=sense_runtime)
        ctx = {"action": "step"}
        out = adapter.apply(ctx)
        assert "body_sense_snapshot" in out
        assert out["body_sense_snapshot"]["overall_status"] == "not_ready"


class TestPracticeWriterAdapter:
    def test_no_runtime_returns_context(self):
        adapter = PracticeWriterAdapter(sense_runtime=None)
        ctx = {"phase": "start"}
        assert adapter.apply(ctx) is ctx

    def test_start_phase(self, sense_runtime):
        adapter = PracticeWriterAdapter(sense_runtime=sense_runtime)
        ctx = {"phase": "start"}
        out = adapter.apply(ctx)
        assert "body_sense_start" in out
        assert out["body_sense_start"]["overall_status"] == "not_ready"

    def test_end_phase(self, sense_runtime):
        adapter = PracticeWriterAdapter(sense_runtime=sense_runtime)
        ctx = {"phase": "end"}
        out = adapter.apply(ctx)
        assert "body_sense_end" in out


class TestMemoryWriterAdapter:
    def test_no_runtime_returns_context(self):
        adapter = MemoryWriterAdapter(sense_runtime=None)
        ctx = {"description": "fail"}
        assert adapter.apply(ctx) is ctx

    def test_flags_body_condition_failure(self, sense_runtime):
        adapter = MemoryWriterAdapter(sense_runtime=sense_runtime)
        ctx = {"description": "fail"}
        out = adapter.apply(ctx)
        assert out["body_condition_failure"] is True
        assert "body_sense_evidence" in out

    def test_ready_state_not_flagged(self, ready_runtime):
        adapter = MemoryWriterAdapter(sense_runtime=ready_runtime)
        ctx = {"description": "success"}
        out = adapter.apply(ctx)
        assert out["body_condition_failure"] is False


class TestHowContextAdapter:
    def test_no_runtime_returns_context(self):
        adapter = HowContextAdapter(sense_runtime=None)
        ctx = {"task": "kick_ball"}
        assert adapter.apply(ctx) is ctx

    def test_adds_block_reasons(self, sense_runtime):
        adapter = HowContextAdapter(sense_runtime=sense_runtime)
        ctx = {"task": "kick_ball"}
        out = adapter.apply(ctx)
        assert "body_readiness" in out
        assert "body_block_reasons" in out
        reasons = out["body_block_reasons"]
        assert any("kick_ball" in r for r in reasons)
        assert any("temperature" in r.lower() for r in reasons)


class TestAutoContextAdapter:
    def test_no_runtime_returns_context(self):
        adapter = AutoContextAdapter(sense_runtime=None)
        ctx = {"task": "kick_ball"}
        assert adapter.apply(ctx) is ctx

    def test_flags_body_condition_failure(self, sense_runtime):
        adapter = AutoContextAdapter(sense_runtime=sense_runtime)
        ctx = {"task": "kick_ball"}
        out = adapter.apply(ctx)
        assert out["body_condition_failure"] is True
        assert "body_sense_snapshot" in out

    def test_ready_state_not_flagged(self, ready_runtime):
        adapter = AutoContextAdapter(sense_runtime=ready_runtime)
        ctx = {"task": "kick_ball"}
        out = adapter.apply(ctx)
        assert out["body_condition_failure"] is False
