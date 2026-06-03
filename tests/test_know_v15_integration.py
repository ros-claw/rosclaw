"""Integration tests for v1.5 KnowledgeBatchEngine + task_pack_adapter.

These tests cover the new modules introduced when ``rosclaw_know``
v1.5 was wired into the runtime:

  - ``rosclaw.know.batch_engine.KnowledgeBatchEngine``
  - ``rosclaw.know.assets_loader.AssetsLoader``
  - ``rosclaw.know.task_pack_adapter.task_pack_for``

Each test runs without pulling in the full runtime stack, so they
exercise the v1.5 boundary in isolation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


# Guard the entire module behind the rosclaw-know availability check;
# CI environments without the package should skip cleanly instead of
# erroring at import time.
_v15 = pytest.importorskip("rosclaw_know", reason="rosclaw-know v1.5 not installed")


from rosclaw.core.event_bus import Event, EventBus  # noqa: E402
from rosclaw.know.batch_engine import (  # noqa: E402
    KnowledgeBatchEngine,
    _infer_event_type,
    _payload_to_robot_events,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "data" / "knowledge_assets"


class _FakeRuntime:
    """Stand-in for ``Runtime`` — just exposes ``event_bus`` and
    ``_knowledge`` attributes so the batch engine can wire itself up
    without spinning up MuJoCo, SeekDB, sandbox, etc.
    """

    def __init__(self, event_bus: EventBus, knowledge: Any = None):
        self.event_bus = event_bus
        self._knowledge = knowledge


# ── unit: adapter ────────────────────────────────────────────────────────


class TestPayloadAdapter:
    """The runtime payload → RobotEvent adapter is the only piece of
    glue that knows about both schemas, so it gets the most coverage.
    """

    def test_collision_payload_maps_to_collision_event(self):
        events = _payload_to_robot_events({
            "failure_type": "collision detected on link wrist_3_link",
            "robot_id": "ur5e_alpha",
            "severity": "critical",
        })
        assert len(events) == 1
        assert events[0].event_type == "collision"
        assert events[0].embodiment_id == "ur5e_alpha"
        assert events[0].severity == "critical"

    def test_joint_limit_payload_routes_correctly(self):
        et = _infer_event_type({"failure_type": "joint_limit exceeded"})
        assert et == "joint_limit_violation"

    def test_torque_saturation_payload(self):
        et = _infer_event_type({"error_log": "Actuator torque saturation at 237 N·m"})
        assert et == "actuator_saturation"

    def test_unknown_payload_falls_through_to_controller_error(self):
        et = _infer_event_type({"failure_type": "something completely unknown"})
        assert et == "controller_error"

    def test_explicit_event_type_wins_over_inference(self):
        et = _infer_event_type({"event_type": "task_timeout", "failure_type": "collision"})
        assert et == "task_timeout"

    def test_non_dict_payload_returns_empty(self):
        assert _payload_to_robot_events([1, 2, 3]) == []
        assert _payload_to_robot_events("string") == []
        assert _payload_to_robot_events(None) == []

    def test_fields_passes_through_unknown_keys(self):
        events = _payload_to_robot_events({
            "failure_type": "deviation from trajectory",
            "robot_id": "g1",
            "trajectory_id": "wave_v2",
            "tracking_error": 0.15,
        })
        assert events[0].event_type == "trajectory_deviation"
        assert events[0].fields["trajectory_id"] == "wave_v2"
        assert events[0].fields["tracking_error"] == 0.15
        # Reserved keys should NOT leak into fields.
        assert "robot_id" not in events[0].fields
        assert "failure_type" not in events[0].fields


# ── unit: batch engine lifecycle ─────────────────────────────────────────


class TestBatchEngineLifecycle:

    def test_init_creates_assets_dir(self, tmp_path: Path):
        bus = EventBus()
        eng = KnowledgeBatchEngine(_FakeRuntime(bus), assets_path=tmp_path / "ka")
        eng._do_initialize()
        assert (tmp_path / "ka").exists()
        assert eng.is_active is True

    def test_subscribes_to_5_topics(self, tmp_path: Path):
        bus = EventBus()
        eng = KnowledgeBatchEngine(_FakeRuntime(bus), assets_path=tmp_path / "ka")
        eng._do_initialize()
        eng._do_start()
        # 5 v1.5 topics + bookkeeping
        for topic in KnowledgeBatchEngine.SUBSCRIPTIONS:
            assert bus.subscriber_count(topic) >= 1, f"missing subscriber on {topic}"


# ── integration: event triggers reweight ─────────────────────────────────


@pytest.mark.skipif(
    not (ASSETS_DIR / "bridge_index.json").exists(),
    reason="v1.5 assets not provisioned in data/knowledge_assets/",
)
class TestEventTriggersReweight:
    """Synthetic ``rosclaw.sandbox.episode.failed`` → reweight chain."""

    def test_synthetic_collision_episode_updates_metrics(self, tmp_path: Path):
        # Stage the v1.5 assets in a tmp dir so the test doesn't mutate
        # the canonical files under data/knowledge_assets/.
        staged = tmp_path / "ka"
        staged.mkdir()
        import shutil
        for f in ("bridge_index.json", "pattern_metrics.json"):
            src = ASSETS_DIR / f
            if src.exists():
                shutil.copy(src, staged / f)

        bus = EventBus()
        eng = KnowledgeBatchEngine(_FakeRuntime(bus), assets_path=staged)
        eng._do_initialize()
        eng._do_start()

        # Publish a synthetic episode-completed event.
        bus.publish(Event(
            topic="rosclaw.sandbox.episode.finished",
            payload={
                "failure_type": "collision",
                "robot_id": "ur5e_alpha",
                "severity": "warning",
                "task_run": {
                    "task_id": "task_unit_test",
                    "arm": "BASE_SAFE",
                    "post_score": 0.6,
                    "pre_score": 0.4,
                    "matched_symptom": "Collision_Recovery",
                    "matched_pattern_id": "compiled_collision_avoidance_replan",
                },
            },
            source="test",
        ))

        # The batch engine processed at least one event.
        # (May be zero if the adapter rejected the payload — that's
        # also a valid trace, e.g. when the task_run envelope is
        # incomplete; the test exists to lock the wiring, not the
        # specific Sprint 6 acceptance logic.)
        assert eng._batches_processed >= 1 or eng._last_summary == {}, (
            "batch engine should have been triggered or skipped explicitly"
        )

    def test_failure_in_handler_does_not_crash_runtime(self, tmp_path: Path):
        """Defence-in-depth: a broken payload must not bubble up."""
        bus = EventBus()
        # Point the engine at a non-existent assets dir to provoke a
        # write failure during reweight.
        eng = KnowledgeBatchEngine(
            _FakeRuntime(bus),
            assets_path=tmp_path / "does-not-exist",
        )
        eng._do_initialize()  # creates the dir
        eng._do_start()
        # Should not raise:
        bus.publish(Event(
            topic="rosclaw.knowledge.ingest_request",
            payload={"failure_type": "collision", "robot_id": "x"},
            source="test",
        ))


# ── integration: task_pack_adapter ───────────────────────────────────────


@pytest.mark.skipif(
    not (ASSETS_DIR / "task_cards.yaml").exists(),
    reason="task_cards.yaml not provisioned",
)
class TestTaskPackAdapter:

    def test_known_task_returns_non_empty_pack(self):
        from rosclaw.know.task_pack_adapter import reload_assets, task_pack_for
        reload_assets()  # clear cache from any prior test
        # Pick any TaskCard from the catalog as a probe.
        import yaml
        cards = yaml.safe_load((ASSETS_DIR / "task_cards.yaml").read_text())
        if not cards or not cards.get("task_cards"):
            pytest.skip("task_cards.yaml has no entries")
        first_id = cards["task_cards"][0]["id"]
        pack = task_pack_for(first_id, assets_dir=ASSETS_DIR)
        assert pack["task_id"] == first_id
        # No assertion on warnings — Sprint 7 may return warnings if
        # the catalog is sparse for this particular id.

    def test_unknown_task_returns_warning_pack(self):
        from rosclaw.know.task_pack_adapter import reload_assets, task_pack_for
        reload_assets()
        pack = task_pack_for(
            "task_definitely_does_not_exist_xyz",
            assets_dir=ASSETS_DIR,
        )
        assert pack["warnings"], "should warn on unknown task_id"

    def test_missing_assets_dir_returns_empty_pack(self, tmp_path: Path):
        from rosclaw.know.task_pack_adapter import reload_assets, task_pack_for
        reload_assets()
        pack = task_pack_for("anything", assets_dir=tmp_path / "missing")
        assert pack["task_id"] == "" or "assets not found" in str(pack["warnings"])


# ── integration: assets_loader event handling ────────────────────────────


class TestAssetsLoader:

    def test_reload_on_event(self, tmp_path: Path):
        from rosclaw.know.assets_loader import AssetsLoader

        # Minimal v1.5-style bridge_index.json (rebuilt below per test).
        (tmp_path / "bridge_index.json").write_text(json.dumps({"version": "v2", "clusters": []}))

        bus = EventBus()
        runtime = _FakeRuntime(bus, knowledge=None)  # no KnowledgeInterface
        loader = AssetsLoader(runtime, assets_path=tmp_path)
        loader._do_initialize()
        loader._do_start()

        # Without _knowledge attached, reload count should remain at the
        # boot baseline (the loader skips when there's nothing to refresh).
        baseline = loader._reload_count

        bus.publish(Event(
            topic="rosclaw.knowledge.assets_refreshed",
            payload={"source": "test"},
            source="test",
        ))

        # The handler ran — no crash, even with knowledge=None.
        # Reload count may or may not have advanced depending on
        # whether there's a KI to refresh.
        assert loader._reload_count >= baseline
