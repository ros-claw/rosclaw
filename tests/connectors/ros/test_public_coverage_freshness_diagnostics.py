"""Original age guards remain failures; diagnostics separate capture and processing."""

from dataclasses import replace

import pytest

from tests.connectors.ros.test_public_coverage_execution import setup_executor


def test_original_source_receive_age_is_reported_without_waiver():
    _, executor, port, _ = setup_executor()
    sample = port.snapshot()
    sample = replace(
        sample,
        sources=tuple(
            replace(fact, received_monotonic_s=sample.captured_monotonic_s - 2)
            if fact.role == "base_tf"
            else fact
            for fact in sample.sources
        ),
    )
    with pytest.raises(ValueError, match="base_tf; receive_age_s=2.000000; capture_age_s=0.000000"):
        executor._validate(sample, historical=True)


def test_processing_delay_remains_rejected_and_is_not_relabelled_as_source_outage(monkeypatch):
    _, executor, port, _ = setup_executor()
    sample = port.snapshot()
    sample = replace(
        sample,
        sources=tuple(
            replace(fact, received_monotonic_s=sample.captured_monotonic_s)
            for fact in sample.sources
        ),
    )
    monkeypatch.setattr(
        "rosclaw.connectors.ros.mission.public_coverage.time.monotonic",
        lambda: sample.captured_monotonic_s + 2,
    )
    with pytest.raises(ValueError, match="receive_age_s=2.000000; capture_age_s=2.000000"):
        executor._validate(sample)


def test_latched_work_fault_is_preserved_when_freshness_is_the_first_failed_guard():
    _, executor, port, _ = setup_executor()
    sample = port.snapshot()
    sample = replace(
        sample,
        work_source_fault="map_tf: original source header age; roles=map_tf",
        sources=tuple(
            replace(fact, received_monotonic_s=sample.captured_monotonic_s - 2)
            if fact.role == "base_tf"
            else fact
            for fact in sample.sources
        ),
    )
    with pytest.raises(
        ValueError, match="work_source_fault='map_tf: original source header age; roles=map_tf'"
    ):
        executor._validate(sample, historical=True)
