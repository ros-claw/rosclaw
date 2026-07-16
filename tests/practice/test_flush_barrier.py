"""Tests for PracticeRecorder/PracticeCatalog durability watermarks and flush barrier."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from rosclaw.practice.recorder import PracticeRecorder
from rosclaw.practice.storage.catalog import PracticeCatalog
from rosclaw.runtime.bus import RuntimeBus
from rosclaw.runtime.event import RuntimeEvent


def _make_event(event_type: str, payload: dict | None = None) -> RuntimeEvent:
    import datetime

    return RuntimeEvent(
        id=f"evt-{event_type}-{time.monotonic_ns()}",
        timestamp=datetime.datetime.now(datetime.UTC),
        source="test",
        robot="test_bot",
        type=event_type,
        payload=payload or {},
    )


def _start_recorder(tmp: str) -> tuple[RuntimeBus, PracticeRecorder]:
    bus = RuntimeBus()
    recorder = PracticeRecorder(bus, data_root=tmp, publish_to_event_bus=False)
    recorder.initialize()
    recorder.start()
    recorder.on_event(
        _make_event("practice.start", {"practice_id": "prac_test_barrier", "robot_id": "test_bot"})
    )
    return bus, recorder


def test_flush_until_waits_for_batch_commit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        catalog = PracticeCatalog(
            Path(tmp) / "catalog.sqlite",
            event_batch_size=500,
            event_flush_ms=50.0,
        )
        for i in range(1, 11):
            catalog.insert_event({"event_id": f"e{i}", "practice_id": "p1", "_wm": i})
            catalog.insert_event_index(
                {"event_id": f"e{i}", "session_id": "s1", "summary": {}, "_wm": i}
            )
        result = catalog.flush_until(10, timeout_sec=5.0)
        assert result["ok"]
        assert result["events_watermark"] >= 10
        assert result["event_index_watermark"] >= 10
        assert catalog.count_events("p1") == 10
        catalog.close()


def test_flush_until_times_out_for_missing_sequence() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        catalog = PracticeCatalog(
            Path(tmp) / "catalog.sqlite",
            event_batch_size=500,
            event_flush_ms=50.0,
        )
        catalog.insert_event({"event_id": "e1", "practice_id": "p1", "_wm": 1})
        catalog.insert_event_index({"event_id": "e1", "session_id": "s1", "summary": {}, "_wm": 1})
        start = time.monotonic()
        result = catalog.flush_until(5, timeout_sec=0.3)
        assert not result["ok"]
        assert time.monotonic() - start < 2.0
        catalog.close()


def test_advance_event_index_watermark_skips_missing_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        catalog = PracticeCatalog(
            Path(tmp) / "catalog.sqlite",
            event_batch_size=500,
            event_flush_ms=50.0,
        )
        catalog.insert_event({"event_id": "e1", "practice_id": "p1", "_wm": 1})
        # No index row for sequence 1 (e.g. JSONL write failed upstream).
        catalog.advance_event_index_watermark(1)
        result = catalog.flush_until(1, timeout_sec=5.0)
        assert result["ok"]
        catalog.close()


def test_watermark_key_is_not_persisted() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        catalog = PracticeCatalog(
            Path(tmp) / "catalog.sqlite",
            event_batch_size=1,  # synchronous flush path
        )
        catalog.insert_event({"event_id": "e1", "practice_id": "p1", "_wm": 7})
        row = catalog._conn.execute("SELECT * FROM events WHERE event_id = 'e1'").fetchone()
        assert row is not None
        assert "_wm" not in row.keys()
        assert catalog.watermarks()["events"] >= 7
        catalog.close()


def test_flush_barrier_satisfied_after_events() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bus, recorder = _start_recorder(tmp)
        try:
            for i in range(20):
                recorder.on_event(_make_event("skill.invoke", {"seq": i}))
            report = recorder.flush_barrier(timeout_sec=10.0)
            assert report["ok"], report
            assert report["target"] == 20
            assert report["jsonl_watermark"] == 20
            assert report["events_watermark"] >= 20
            assert report["event_index_watermark"] >= 20
        finally:
            recorder.stop()



def test_flush_barrier_called_on_finalize() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bus, recorder = _start_recorder(tmp)
        try:
            for i in range(5):
                recorder.on_event(_make_event("skill.invoke", {"seq": i}))
            recorder.on_event(_make_event("practice.stop", {"outcome": "SUCCESS"}))
            barrier = recorder.last_flush_barrier
            assert barrier is not None
            assert barrier["ok"], barrier
            assert barrier["target"] == 5

            # After a clean finalize the catalog agrees with the JSONL count.
            catalog = PracticeCatalog(Path(tmp) / "indexes" / "practice_catalog.sqlite")
            try:
                assert catalog.count_events("prac_test_barrier") == 5
            finally:
                catalog.close()
        finally:
            recorder.stop()

