"""WP-P0-2 红测试（总纲 §5.2）：SessionCatalogV1 产品索引。

红测试先行——当前 pi_session_bindings 只有安全关系（session/
mission/body/mode/binding 状态），没有标题/摘要/任务状态/成本/
归档/搜索字段；首页/选择器只能全量扫 JSONL。

红线：产品字段不得塞回 pi_session_bindings（binding 是安全关系，
Catalog 是产品检索投影）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from tests.agentd.test_pi_tool_bridge import _setup


def _write_session(home: Path, session_id: str, name: str, first_message: str) -> None:
    sessions_dir = home / "agent" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = sessions_dir / f"2026-08-10T10-00-00_{session_id}.jsonl"
    lines = [
        json.dumps({"type": "session", "id": session_id,
                    "timestamp": "2026-08-10T10:00:00Z"}),
        json.dumps({"type": "session_info", "name": name}),
        json.dumps({"type": "message",
                    "message": {"role": "user", "content": first_message}}),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestCatalogBackfill:
    async def test_refresh_backfills_from_jsonl_and_bindings(
        self, tmp_path: Path
    ) -> None:
        """refresh：JSONL 扫描 + binding 投影（mission/body/mode）落库。"""
        service, mission = await _setup(tmp_path)  # 绑定 pi_1 → mission
        _write_session(tmp_path, "pi_1", "", "我想画五角星")
        from rosclaw.agentd.session_catalog import SessionCatalog

        catalog = SessionCatalog(service._store.connection)
        catalog.refresh(tmp_path)
        rows = catalog.list()
        assert len(rows) == 1
        row = rows[0]
        assert row["session_id"] == "pi_1"
        assert row["mission_id"] == mission.mission_id
        # body/mode 从 binding 投影（不硬编码——读权威绑定值）。
        binding_row = service._store.connection.execute(
            "SELECT body_id, execution_mode FROM pi_session_bindings "
            "WHERE pi_session_id = 'pi_1'"
        ).fetchone()
        assert row["body_id"] == binding_row[0]
        assert row["execution_mode"] == binding_row[1] == "SIMULATION"
        # 无名会话用确定性标题（首条目标，≤30 字）。
        assert row["display_name"] == "我想画五角星"
        assert row["title_source"] == "auto"
        await service.close()

    async def test_search_and_archive(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        _write_session(tmp_path, "pi_1", "", "我想画五角星")
        from rosclaw.agentd.session_catalog import SessionCatalog

        catalog = SessionCatalog(service._store.connection)
        catalog.refresh(tmp_path)
        hits = catalog.search("五角星")
        assert len(hits) == 1
        catalog.archive("pi_1")
        assert catalog.list() == [], "归档后默认列表仍可见"
        assert len(catalog.list(include_archived=True)) == 1
        await service.close()

    async def test_user_rename_wins(self, tmp_path: Path) -> None:
        """用户重命名永远优先（title_source=user 不被自动规则覆盖）。"""
        service, mission = await _setup(tmp_path)
        _write_session(tmp_path, "pi_1", "", "我想画五角星")
        from rosclaw.agentd.session_catalog import SessionCatalog

        catalog = SessionCatalog(service._store.connection)
        catalog.refresh(tmp_path)
        catalog.rename("pi_1", "我的五角星")
        # JSONL 里首条消息变了——refresh 不得覆盖用户标题。
        _write_session(tmp_path, "pi_1", "", "完全不同的新目标")
        catalog.refresh(tmp_path)
        row = catalog.list()[0]
        assert row["display_name"] == "我的五角星"
        assert row["title_source"] == "user"
        await service.close()


class TestCatalogPerformance:
    async def test_10k_sessions_list_search_slo(self, tmp_path: Path) -> None:
        """10,000 会话：list p95 < 300ms、搜索 < 150ms（总纲 §WP-P0-2）。"""
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.session_catalog import SessionCatalog

        catalog = SessionCatalog(service._store.connection)
        now = "2026-08-10T10:00:00Z"
        for i in range(10_000):
            catalog.upsert(
                session_id=f"sess_{i:05d}",
                pi_session_path=f"/tmp/{i}.jsonl",
                display_name=f"任务 {i}",
                mission_id=mission.mission_id,
                body_id="sim/ur5e",
                execution_mode="SIMULATION",
                lifecycle_state="COMPLETED",
                search_text=f"任务 {i} 巡检 五角星",
                created_at=now,
                last_active_at=now,
            )
        start = time.monotonic()
        rows = catalog.list()
        list_ms = (time.monotonic() - start) * 1000
        assert len(rows) == 10_000
        assert list_ms < 300, f"list {list_ms:.0f}ms 超 300ms SLO"
        start = time.monotonic()
        hits = catalog.search("五角星")
        search_ms = (time.monotonic() - start) * 1000
        assert hits and search_ms < 150, f"search {search_ms:.0f}ms 超 150ms SLO"
        await service.close()


class TestProductFieldsNotInBindings:
    def test_bindings_schema_unchanged(self) -> None:
        """产品字段不得塞回 pi_session_bindings（安全关系表）。"""
        migration = Path(
            "src/rosclaw/storage/migrations/014_pi_session_bindings_sqlite.sql"
        ).read_text(encoding="utf-8")
        for field in ("display_name", "summary", "cost_microunits", "archived_at"):
            assert field not in migration, (
                f"产品字段 {field} 混入了安全 binding 表"
            )
