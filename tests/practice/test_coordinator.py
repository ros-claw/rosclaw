"""Tests for PracticeCoordinator."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from rosclaw.core.event_bus import EventBus
from rosclaw.practice.config import PracticeConfig, SourceConfig
from rosclaw.practice.coordinator import PracticeCoordinator


def test_coordinator_mock_session_writes_manifest_and_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PracticeConfig(
            robot_id="test_bot",
            task_name="pick cup",
            data_root=tmp,
            sources=SourceConfig(agent=True, runtime=True),
            mock=True,
            publish_to_event_bus=False,
        )
        coord = PracticeCoordinator(cfg)
        coord.initialize()
        coord.start()
        time.sleep(0.4)
        coord.stop()

        summary = coord.summary
        assert summary is not None
        assert summary.outcome == "SUCCESS"
        assert summary.event_count >= 4

        session_dir = Path(tmp) / "sessions" / summary.practice_id
        assert (session_dir / "manifest.yaml").exists()
        assert (session_dir / "raw" / "events.jsonl").exists()

        events_jsonl = session_dir / "raw" / "events.jsonl"
        lines = events_jsonl.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == summary.event_count
        first = json.loads(lines[0])
        assert first["schema_version"] == "practice.event.v1"
        assert first["practice_id"] == summary.practice_id


def test_coordinator_publishes_session_events_on_event_bus():
    with tempfile.TemporaryDirectory() as tmp:
        bus = EventBus()
        received = []
        bus.subscribe("practice.session_started", lambda e: received.append(("started", e.payload)))
        bus.subscribe(
            "practice.session_finished", lambda e: received.append(("finished", e.payload))
        )

        cfg = PracticeConfig(
            robot_id="test_bot",
            task_name="pick cup",
            data_root=tmp,
            sources=SourceConfig(agent=True),
            mock=True,
            event_bus=bus,
        )
        coord = PracticeCoordinator(cfg)
        coord.initialize()
        coord.start()
        time.sleep(0.2)
        coord.stop()

        assert any(t == "started" for t, _ in received)
        assert any(t == "finished" for t, _ in received)


def test_coordinator_catalog_indexed():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PracticeConfig(
            robot_id="test_bot",
            task_name="pick cup",
            data_root=tmp,
            sources=SourceConfig(agent=True),
            mock=True,
            publish_to_event_bus=False,
        )
        coord = PracticeCoordinator(cfg)
        coord.initialize()
        coord.start()
        time.sleep(0.2)
        coord.stop()

        record = coord.catalog.get_practice(coord.summary.practice_id)
        assert record is not None
        assert record["outcome"] == "SUCCESS"
        # event_count is tracked in the summary, not the practices table
        assert coord.summary.event_count > 0
        assert (
            coord.catalog.count_source_events(coord.summary.practice_id)
            == coord._source_event_count
        )
        assert coord.catalog.count_events(coord.summary.practice_id) == coord.summary.event_count


def owned_catalog_workers(catalog):
    return tuple(
        writer._thread
        for writer in (catalog._event_writer, catalog._event_index_writer)
        if writer is not None
    )


def assert_workers_terminated(workers):
    for worker in workers:
        worker.join(timeout=1.0)
        assert not worker.is_alive()


def ownership_config(tmp_path):
    return PracticeConfig(
        robot_id="ownership_bot",
        data_root=str(tmp_path),
        sources=SourceConfig(agent=True, runtime=True),
        mock=True,
        publish_to_event_bus=False,
    )


def test_coordinator_owned_catalog_stop_restart_and_queries(tmp_path):
    import sqlite3

    from rosclaw.core.lifecycle import LifecycleState

    coord = PracticeCoordinator(ownership_config(tmp_path))
    catalog = coord.catalog
    generations = []
    summaries = []
    try:
        for _cycle in range(2):
            coord.initialize()
            workers = owned_catalog_workers(catalog)
            assert len(workers) == 2
            assert all(worker.is_alive() for worker in workers)
            assert not any(worker is old for worker in workers for old in generations)
            generations.extend(workers)
            coord.start()
            coord.tick()
            poller = coord._poll_thread
            assert poller.is_alive()
            coord.stop()
            assert coord.state is LifecycleState.STOPPED
            assert_workers_terminated((*workers, poller))
            assert owned_catalog_workers(catalog) == ()
            assert coord.catalog is catalog
            summary = coord.summary
            summaries.append(summary)
            assert summary.event_count > 0
            assert catalog.get_practice(summary.practice_id)["outcome"] == "SUCCESS"
            assert catalog.count_events(summary.practice_id) == summary.event_count
            assert catalog.count_source_events(summary.practice_id) == coord._source_event_count
            assert (summary.artifact_dir / "manifest.yaml").is_file()
            lines = (summary.artifact_dir / "raw" / "events.jsonl").read_text().splitlines()
            assert len(lines) == summary.event_count
            with sqlite3.connect(str(catalog._db_path)) as reader:
                assert (
                    reader.execute(
                        "SELECT COUNT(*) FROM events WHERE practice_id = ?",
                        (summary.practice_id,),
                    ).fetchone()[0]
                    == summary.event_count
                )
            coord.stop()
            assert coord.summary is summary
            catalog.flush()
            assert owned_catalog_workers(catalog) == ()
        assert summaries[0].practice_id != summaries[1].practice_id
        for summary in summaries:
            assert catalog.count_events(summary.practice_id) == summary.event_count
        assert_workers_terminated(generations)
    finally:
        coord.stop()
        catalog.close()
        catalog.close()


def test_coordinator_stop_before_start_obeys_public_lifecycle(tmp_path):
    from rosclaw.core.lifecycle import LifecycleState

    coord = PracticeCoordinator(ownership_config(tmp_path))
    initial = owned_catalog_workers(coord.catalog)
    try:
        # LifecycleMixin deliberately makes stop a no-op before initialize.
        coord.stop()
        assert coord.state is LifecycleState.UNINITIALIZED
        assert all(worker.is_alive() for worker in initial)
        coord.initialize()
        coord.stop()
        assert coord.state is LifecycleState.STOPPED
        assert coord.summary is None
        assert_workers_terminated(initial)
        assert owned_catalog_workers(coord.catalog) == ()
        coord.stop()
        coord.initialize()
        replacement = owned_catalog_workers(coord.catalog)
        assert len(replacement) == 2
        assert all(worker.is_alive() for worker in replacement)
        assert not any(worker is old for worker in replacement for old in initial)
        coord.stop()
        assert_workers_terminated(replacement)
        assert owned_catalog_workers(coord.catalog) == ()
        assert coord.catalog.list_practices() == []
    finally:
        coord.stop()
        coord.catalog.close()


def test_coordinator_recorder_backend_preserves_separate_ownership(tmp_path):
    from rosclaw.practice.config import RecorderConfig
    from rosclaw.practice.recorder import PracticeRecorder
    from rosclaw.runtime.bus import RuntimeBus

    bus = RuntimeBus()
    recorder = PracticeRecorder(
        bus,
        data_root=tmp_path,
        publish_to_event_bus=False,
        config=RecorderConfig(telemetry_enabled=False),
    )
    recorder.initialize()
    recorder.start()
    coord = PracticeCoordinator(ownership_config(tmp_path), runtime_bus=bus, recorder=recorder)
    catalog = coord.catalog
    workers = owned_catalog_workers(catalog)
    recorder_catalog = None
    try:
        coord.initialize()
        coord.start()
        coord.tick()
        recorder_catalog = recorder._catalog
        assert recorder_catalog is not None
        assert recorder_catalog is not catalog
        recorder_workers = owned_catalog_workers(recorder_catalog)
        assert len(recorder_workers) == 2
        assert all(worker.is_alive() for worker in recorder_workers)
        assert all(worker.is_alive() for worker in workers)
        coord.stop()
        assert_workers_terminated(workers)
        assert owned_catalog_workers(catalog) == ()
        practice_id = coord.summary.practice_id
        assert catalog.get_practice(practice_id) is not None
        assert catalog.count_events(practice_id) > 0
        assert (coord.summary.artifact_dir / "manifest.yaml").is_file()
        coord.stop()
        coord.initialize()
        replacement = owned_catalog_workers(catalog)
        assert len(replacement) == 2
        assert not any(worker is old for worker in replacement for old in workers)
        coord.start()
        coord.tick()
        coord.stop()
        assert_workers_terminated(replacement)
        assert owned_catalog_workers(catalog) == ()
        assert catalog.count_events(practice_id) > 0
    finally:
        coord.stop()
        catalog.close()
        recorder.stop()
        # The recorder owns its catalog, never the coordinator. Explicitly
        # finalize that separate owner even if its stop already released it.
        if recorder._catalog is not None:
            recorder._catalog.close()
        if recorder_catalog is not None:
            recorder_catalog.close()


def test_coordinator_constructor_failure_closes_acquired_catalog(tmp_path):
    import sqlite3
    import threading
    from unittest.mock import patch

    import pytest

    import rosclaw.practice.coordinator as module

    connections = []
    workers = []
    connect = sqlite3.connect
    start = threading.Thread.start

    def track_connection(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    def track_start(worker):
        result = start(worker)
        if worker.name.startswith("catalog-batch-"):
            workers.append(worker)
        return result

    def fail_bus(*args, **kwargs):
        assert len(workers) == 2
        assert all(worker.is_alive() for worker in workers)
        raise OSError("bus assembly injected")

    with (
        patch.object(sqlite3, "connect", side_effect=track_connection),
        patch.object(threading.Thread, "start", new=track_start),
        patch.object(module, "RuntimeBus", side_effect=fail_bus),
        pytest.raises(OSError, match="bus assembly injected"),
    ):
        PracticeCoordinator(ownership_config(tmp_path))
    assert len(connections) == 1
    assert len(workers) == 2
    assert_workers_terminated(workers)
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")


def test_coordinator_start_assembly_failure_retires_owned_workers(tmp_path):
    from unittest.mock import patch

    import pytest

    import rosclaw.practice.coordinator as module

    coord = PracticeCoordinator(ownership_config(tmp_path))
    workers = owned_catalog_workers(coord.catalog)
    try:
        coord.initialize()
        assert all(worker.is_alive() for worker in workers)
        with (
            patch.object(module, "JsonlWriter", side_effect=OSError("writer assembly injected")),
            pytest.raises(OSError, match="writer assembly injected"),
        ):
            coord.start()
        assert_workers_terminated(workers)
        assert owned_catalog_workers(coord.catalog) == ()
        assert coord.catalog.list_practices() == []
        coord.stop()
        coord.initialize()
        replacement = owned_catalog_workers(coord.catalog)
        assert all(worker.is_alive() for worker in replacement)
        coord.start()
        coord.tick()
        coord.stop()
        assert_workers_terminated(replacement)
        assert coord.catalog.count_events(coord.summary.practice_id) == coord.summary.event_count
    finally:
        coord.stop()
        coord.catalog.close()


def test_coordinator_initialize_failure_retires_workers_and_can_retry(tmp_path):
    from unittest.mock import patch

    import pytest

    from rosclaw.core.lifecycle import LifecycleState

    coord = PracticeCoordinator(ownership_config(tmp_path))
    workers = owned_catalog_workers(coord.catalog)
    try:
        with (
            patch.object(
                coord.layout,
                "ensure_directories",
                side_effect=OSError("directory creation injected"),
            ),
            pytest.raises(OSError, match="directory creation injected"),
        ):
            coord.initialize()
        assert coord.state is LifecycleState.ERROR
        assert_workers_terminated(workers)
        assert owned_catalog_workers(coord.catalog) == ()
        coord.stop()  # ERROR is also a no-op in the existing LifecycleMixin.
        coord.initialize()
        replacement = owned_catalog_workers(coord.catalog)
        assert len(replacement) == 2
        assert all(worker.is_alive() for worker in replacement)
        coord.stop()
        assert_workers_terminated(replacement)
        assert coord.catalog.list_practices() == []
    finally:
        coord.stop()
        coord.catalog.close()
