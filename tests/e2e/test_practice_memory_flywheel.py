"""PR-DF-16B E2E: a REAL practice session close flows through the Runtime's
event bus into Memory 2.0 — no manual `memory distill`, no CLI (phase-II §5).
"""

import time

from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.practice.config import PracticeConfig, SourceConfig
from rosclaw.practice.coordinator import PracticeCoordinator


def test_runtime_session_close_auto_distills(tmp_path):
    cfg = RuntimeConfig(
        robot_id="sim_flywheel",
        workspace_home=str(tmp_path / "home"),
        seekdb_backend="sqlite",
        seekdb_path=str(tmp_path / "knowledge.sqlite"),
        enable_memory=True,
        enable_practice=False,
        enable_how=False,
        enable_auto=False,
        enable_knowledge=False,
        enable_skill_manager=False,
        enable_provider=False,
        enable_sense=False,
        enable_firewall=False,
        enable_event_persistence=False,
        enable_tracing=False,
    )
    rt = Runtime(cfg)
    rt.initialize()
    try:
        assert rt._memory_distillation is not None, "distillation service must be wired"
        # a real coordinator session publishing to the Runtime's bus
        pcfg = PracticeConfig(
            robot_id="sim_flywheel",
            task_name="slide puck",
            data_root=str(tmp_path / "practice"),
            sources=SourceConfig(agent=True, runtime=True),
            mock=True,
            publish_to_event_bus=True,
            event_bus=rt.event_bus,
        )
        coord = PracticeCoordinator(pcfg)
        coord.initialize()
        coord.start()
        time.sleep(0.3)
        coord.stop()
        assert rt._memory_distillation.drain(timeout=20)
        store = rt._data_plane.structured_store
        assert store.count("memory_items", {}) >= 1, "session close must auto-distill"
        runs = store.query("memory_distillation_runs", {})
        assert runs and runs[0]["status"] == "completed"
        assert runs[0]["session_dir"]
    finally:
        rt.stop()
    assert rt._memory_distillation is None
