"""Tests for ReportGenerator."""
import shutil

from rosclaw.auto.config import AutoConfig
from rosclaw.auto.engine.auto_engine import AutoEngine
from rosclaw.auto.reports import ReportGenerator


class TestReportGenerator:
    """AUTO-REPORT-001~005: Report generation tests."""

    def test_generate_markdown_contains_summary(self):
        store_path = "./.rosclaw_auto_test_report_md"
        shutil.rmtree(store_path, ignore_errors=True)
        engine = AutoEngine(config=AutoConfig(local_store_path=store_path))
        gen = ReportGenerator(engine)

        task = engine.create_task("pick_cube", "panda", "pick_v1")
        engine.promote_champion("pick_v1.5", task.id, "sim", {"success_rate": 0.76}, "", "", "")

        md = gen.generate_markdown(task.id)
        assert "# Evolution Report:" in md
        assert "pick_cube" in md
        assert "Champions" in md
        assert "pick_v1.5" in md

    def test_generate_markdown_contains_deadends(self):
        store_path = "./.rosclaw_auto_test_report_de"
        shutil.rmtree(store_path, ignore_errors=True)
        engine = AutoEngine(config=AutoConfig(local_store_path=store_path))
        gen = ReportGenerator(engine)

        task = engine.create_task("pick_cube", "panda", "pick_v1")
        engine.register_deadend(task.id, "increase_force", "too dangerous", ["sandbox_rejected"])

        md = gen.generate_markdown(task.id)
        assert "DeadEnds" in md
        assert "increase_force" in md

    def test_generate_markdown_contains_lineage(self):
        store_path = "./.rosclaw_auto_test_report_lineage"
        shutil.rmtree(store_path, ignore_errors=True)
        engine = AutoEngine(config=AutoConfig(local_store_path=store_path))
        gen = ReportGenerator(engine)

        task = engine.create_task("pick_cube", "panda", "pick_v1")
        engine.promote_champion("pick_v1", task.id, "baseline", {}, "", "", "")
        engine.promote_champion("pick_v1.5", task.id, "sim", {"sr": 0.76}, "pick_v1", "p1", "e1")

        md = gen.generate_markdown(task.id)
        assert "Skill Lineage" in md
        assert "pick_v1.5" in md

    def test_generate_champion_card(self):
        store_path = "./.rosclaw_auto_test_report_card"
        shutil.rmtree(store_path, ignore_errors=True)
        engine = AutoEngine(config=AutoConfig(local_store_path=store_path))
        gen = ReportGenerator(engine)

        task = engine.create_task("pick_cube", "panda", "pick_v1")
        engine.promote_champion("pick_v1.5", task.id, "sim", {"success_rate": 0.76}, "pick_v1", "p1", "e1")

        md = gen.generate_champion_card_markdown(task.id, "sim")
        assert "Champion Skill Card" in md
        assert "pick_v1.5" in md
        assert "sim" in md.lower()

    def test_generate_markdown_empty_task(self):
        store_path = "./.rosclaw_auto_test_report_empty"
        shutil.rmtree(store_path, ignore_errors=True)
        engine = AutoEngine(config=AutoConfig(local_store_path=store_path))
        gen = ReportGenerator(engine)

        task = engine.create_task("pick_cube", "panda", "pick_v1")
        md = gen.generate_markdown(task.id)
        assert "Evolution Report" in md
