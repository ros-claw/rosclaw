"""十二审 PR-12.2 红测试：事件分页零丢失 + seq O(1) + schema v2。

红测试先行——修复前必须红：
1. tail 旧语义（取最后 N 条）在两次轮询间新增 >N 条时永久丢中间
   事件；新 tail_page 返回最早 N 条 + next_cursor + has_more；
2. 10,000 条事件分页读取无重复、无缺口（审计 PR-12.2 退出条件）；
3. seq 由单写者内存计数（不再每写一次重读整个文件）；
4. 事件带 schema v2。
"""

from __future__ import annotations

from pathlib import Path

from rosclaw.agentd.workers.event_store import WorkerEventStore


class TestPagination:
    def test_earliest_n_with_cursor_and_has_more(self, tmp_path: Path) -> None:
        store = WorkerEventStore(tmp_path)
        for i in range(250):
            store.append_event("wo_page0001", "att_1", "tick", {"i": i})
        page1 = store.tail_page("wo_page0001", after_seq=0, limit=100)
        assert len(page1["events"]) == 100
        assert page1["events"][0]["seq"] == 1  # 最早，不是最后 100 条
        assert page1["has_more"] is True
        page2 = store.tail_page("wo_page0001", after_seq=page1["next_cursor"], limit=100)
        assert page2["events"][0]["seq"] == 101
        assert page2["has_more"] is True
        page3 = store.tail_page("wo_page0001", after_seq=page2["next_cursor"], limit=100)
        assert len(page3["events"]) == 50
        assert page3["has_more"] is False
        seqs = [e["seq"] for e in page1["events"] + page2["events"] + page3["events"]]
        assert seqs == list(range(1, 251)), "分页有缺口或重复"

    def test_ten_thousand_events_no_gap(self, tmp_path: Path) -> None:
        """审计退出条件：10,000 事件分页零丢失零重复。"""
        store = WorkerEventStore(tmp_path)
        total = 10_000
        for i in range(total):
            store.append_event("wo_page0002", "att_1", "tick", {"i": i})
        seen: list[int] = []
        cursor = 0
        while True:
            page = store.tail_page("wo_page0002", after_seq=cursor, limit=997)
            seen.extend(e["seq"] for e in page["events"])
            cursor = page["next_cursor"]
            if not page["has_more"]:
                break
        assert seen == list(range(1, total + 1))

    def test_seq_monotonic_across_store_instances(self, tmp_path: Path) -> None:
        """重启语义：新实例从文件行数续写（不重置、不冲突）。"""
        store = WorkerEventStore(tmp_path)
        store.append_event("wo_seq00001", "", "a", {})
        store.append_event("wo_seq00001", "", "b", {})
        store2 = WorkerEventStore(tmp_path)
        store2.append_event("wo_seq00001", "", "c", {})
        seqs = [e["seq"] for e in store2.tail("wo_seq00001")]
        assert seqs == [1, 2, 3]

    def test_event_schema_v2(self, tmp_path: Path) -> None:
        store = WorkerEventStore(tmp_path)
        store.append_event("wo_v2000001", "", "tick", {})
        event = store.tail("wo_v2000001")[0]
        assert event.get("v") == 2

    def test_concurrent_appends_unique_seq(self, tmp_path: Path) -> None:
        """同一进程内交替写两个 work order——各自 seq 独立连续。"""
        store = WorkerEventStore(tmp_path)
        for i in range(50):
            store.append_event("wo_multi001", "", "tick", {"i": i})
            store.append_event("wo_multi002", "", "tick", {"i": i})
        a = [e["seq"] for e in store.tail("wo_multi001", limit=100)]
        b = [e["seq"] for e in store.tail("wo_multi002", limit=100)]
        assert a == list(range(1, 51))
        assert b == list(range(1, 51))
