"""Tests for DashboardExporter."""

import shutil

from rosclaw.auto.config import AutoConfig
from rosclaw.auto.dashboard import DashboardExporter
from rosclaw.auto.engine.auto_engine import AutoEngine


class TestDashboardExporter:
    """AUTO-DASH-001~005: Dashboard data export tests."""

    def test_export_summary(self, store_test_champion):
        store_path = "./.rosclaw_auto_test_dash_summary"
        shutil.rmtree(store_path, ignore_errors=True)
        engine = AutoEngine(config=AutoConfig(local_store_path=store_path))
        exporter = DashboardExporter(engine)

        task = engine.create_task("pick_cube", "panda", "pick_v1")
        store_test_champion(engine, "pick_v1.5", task.id, "sim", {"sr": 0.76})
        engine.register_deadend(task.id, "bad_dir", "test", [])

        data = exporter.export(task.id)
        assert "summary" in data
        assert data["summary"]["total_tasks"] == 1
        assert data["summary"]["total_champions"] == 1
        assert data["summary"]["total_deadends"] == 1
        assert "generated_at" in data

    def test_export_tasks(self):
        store_path = "./.rosclaw_auto_test_dash_tasks"
        shutil.rmtree(store_path, ignore_errors=True)
        engine = AutoEngine(config=AutoConfig(local_store_path=store_path))
        exporter = DashboardExporter(engine)

        task = engine.create_task("pick_cube", "panda", "pick_v1")
        data = exporter.export(task.id)
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["name"] == "pick_cube"

    def test_export_champions(self, store_test_champion):
        store_path = "./.rosclaw_auto_test_dash_champs"
        shutil.rmtree(store_path, ignore_errors=True)
        engine = AutoEngine(config=AutoConfig(local_store_path=store_path))
        exporter = DashboardExporter(engine)

        task = engine.create_task("pick_cube", "panda", "pick_v1")
        store_test_champion(
            engine,
            "pick_v1.5",
            task.id,
            "sim",
            {"success_rate": 0.76},
            "pick_v1",
            "p1",
            "e1",
        )

        data = exporter.export(task.id)
        assert len(data["champions"]) == 1
        assert data["champions"][0]["skill_id"] == "pick_v1.5"
        assert data["champions"][0]["level"] == "sim"

    def test_export_lineage_graph(self, store_test_champion):
        store_path = "./.rosclaw_auto_test_dash_lineage"
        shutil.rmtree(store_path, ignore_errors=True)
        engine = AutoEngine(config=AutoConfig(local_store_path=store_path))
        exporter = DashboardExporter(engine)

        task = engine.create_task("pick_cube", "panda", "pick_v1")
        engine.promote_champion("pick_v1", task.id, "baseline", {}, "", "", "")
        store_test_champion(
            engine, "pick_v1.5", task.id, "sim", {"sr": 0.76}, "pick_v1", "p1", "e1"
        )

        data = exporter.export(task.id)
        lineage = data["lineage"]
        assert "nodes" in lineage
        assert "edges" in lineage
        assert len(lineage["nodes"]) >= 2
        assert len(lineage["edges"]) >= 1
        assert lineage["edges"][0]["from"] == "pick_v1"
        assert lineage["edges"][0]["to"] == "pick_v1.5"

    def test_export_timeline(self):
        store_path = "./.rosclaw_auto_test_dash_timeline"
        shutil.rmtree(store_path, ignore_errors=True)
        engine = AutoEngine(config=AutoConfig(local_store_path=store_path))
        exporter = DashboardExporter(engine)

        task = engine.create_task("pick_cube", "panda", "pick_v1")
        engine.create_experiment("prop_1", "patch_1", task.name, "pick_v1", "pick_v1_c1")

        # Timeline filters by task name (experiments store task.name not task.id)
        data = exporter.export(task.name)
        assert len(data["timeline"]) >= 1
        assert data["timeline"][0]["status"] == "pending"

    def test_export_json_roundtrip(self):
        store_path = "./.rosclaw_auto_test_dash_json"
        shutil.rmtree(store_path, ignore_errors=True)
        engine = AutoEngine(config=AutoConfig(local_store_path=store_path))
        exporter = DashboardExporter(engine)

        task = engine.create_task("pick_cube", "panda", "pick_v1")
        json_str = exporter.export_json(task.id)
        assert "summary" in json_str
        assert "champions" in json_str
        assert "lineage" in json_str
