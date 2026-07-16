"""Fault injection: disk full — recorder keeps the session alive, barrier reports."""

from __future__ import annotations

import datetime
import errno
import tempfile

from rosclaw.practice.recorder import PracticeRecorder
from rosclaw.runtime.bus import RuntimeBus
from rosclaw.runtime.event import RuntimeEvent


def _evt(event_type: str, payload: dict | None = None) -> RuntimeEvent:
    import time

    return RuntimeEvent(
        id=f"evt-{event_type}-{time.monotonic_ns()}",
        timestamp=datetime.datetime.now(datetime.UTC),
        source="test",
        robot="test_bot",
        type=event_type,
        payload=payload or {},
    )


def test_disk_full_keeps_session_alive_and_flags_barrier() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bus = RuntimeBus()
        recorder = PracticeRecorder(bus, data_root=tmp, publish_to_event_bus=False)
        recorder.initialize()
        recorder.start()
        try:
            recorder.on_event(
                _evt("practice.start", {"practice_id": "prac_enospc", "robot_id": "test_bot"})
            )
            recorder.on_event(_evt("skill.invoke", {"seq": 1}))

            # Simulate ENOSPC on the JSONL path for the next event only.
            original_write = recorder._writer.write

            def _failing_write(record: dict) -> None:
                raise OSError(errno.ENOSPC, "No space left on device")

            recorder._writer.write = _failing_write  # type: ignore[method-assign]
            recorder.on_event(_evt("skill.invoke", {"seq": 2}))
            recorder._writer.write = original_write  # type: ignore[method-assign]
            recorder.on_event(_evt("skill.invoke", {"seq": 3}))

            # The session is still alive (robot task not blocked by disk).
            assert recorder.session is not None
            assert recorder._event_count == 3

            report = recorder.flush_barrier(timeout_sec=10.0)
            # Event 2 never hit the JSONL file; later events did, so the
            # watermark advanced but the gap is tracked explicitly.
            assert report["jsonl_watermark"] == 3
            assert not report["ok"]
            assert report["failed_sequences"]["jsonl"] == [2]
            # But the catalog batch path still committed everything, and the
            # index watermark advanced past the skipped sequence.
            assert report["events_watermark"] >= 3
            assert report["event_index_watermark"] >= 3

            # Finalize must not raise; the CRITICAL log surfaces the gap.
            recorder.on_event(_evt("practice.stop", {"outcome": "UNKNOWN"}))
            assert recorder.last_flush_barrier is not None
        finally:
            recorder.stop()
