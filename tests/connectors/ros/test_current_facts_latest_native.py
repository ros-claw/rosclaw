"""A single predecision refresh must revalidate originals and never forgive failure."""

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def collector(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("simulation_current_facts")
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: 1_000_000_000)
    facts = object.__new__(module.CurrentSimulationFacts)
    facts.failed_reason = None
    facts.directory = tmp_path / "evidence"
    facts.assets_directory = tmp_path / "assets"
    for p in [facts.directory, facts.assets_directory]:
        p.mkdir()
    for name in ["robot.urdf", "world.sdf"]:
        (facts.assets_directory / name).write_bytes(b"original")
    raw = b"{}"
    (facts.directory / "execution-facts-enrollment.json").write_bytes(raw)
    facts.enrollment_sha256 = hashlib.sha256(raw).hexdigest()
    facts._directory_identity = (facts.directory.stat().st_dev, facts.directory.stat().st_ino)
    facts._assets_identity = (
        facts.assets_directory.stat().st_dev,
        facts.assets_directory.stat().st_ino,
    )
    facts.enrollment = {
        "urdf_sha256": hashlib.sha256(b"original").hexdigest(),
        "world_sha256": hashlib.sha256(b"original").hexdigest(),
        "request": {},
        "policy": {},
        "policy_hash": "policy",
    }
    facts.reader = SimpleNamespace(
        poll=lambda: {"continuous": True},
        last_raw=json.dumps({"observer_monotonic_ns": 990_000_000}),
    )
    first = {
        "file": "original-native-001.json",
        "sha256": "a" * 64,
        "captured_at": "2026-10-18T00:00:00Z",
        "received_monotonic_ns": 980_000_000,
    }
    latest = {
        **first,
        "file": "original-native-002.json",
        "sha256": "b" * 64,
        "captured_at": "2026-10-18T00:00:00.100Z",
        "received_monotonic_ns": 990_000_000,
    }
    seen = []
    facts._load_native_model = lambda pointer: seen.append(pointer) or pointer["file"]
    reads = iter([first, latest])
    monkeypatch.setattr(
        module, "read_current_native_pointer", lambda _: (next(reads), 1_000_000_000)
    )
    evaluated = []
    monkeypatch.setattr(
        module,
        "review_current_sim_navigation_health",
        lambda *a, **kw: evaluated.append(kw) or {"status": "HEALTHY_SIM_OBSERVATION_ONLY"},
    )
    return module, facts, first, latest, seen, evaluated


def test_refresh_only_once_before_health_checks_latest_original(collector):
    _, facts, first, latest, seen, evaluated = collector
    result = facts.collect(freshness_horizon_ms=100)
    assert seen == [first, latest]
    assert len(evaluated) == 1
    assert evaluated[0]["model"] == latest["file"]
    assert evaluated[0]["freshness_margin_ms"] == 100
    assert result["native_pointer"] == latest
    assert result["superseded_native_pointer_before_health"] == first
    assert result["valid_until_monotonic_ns"] == 1_100_000_000


@pytest.mark.parametrize(
    "mutation",
    [
        {"file": "original-native-000.json"},
        {"file": "original-native-001.json"},
        {"received_monotonic_ns": 970_000_000},
        {"captured_at": "2026-10-17T00:00:00Z"},
    ],
)
def test_changed_or_rolled_back_pointer_latches_before_health(collector, mutation):
    _, facts, _, latest, _, evaluated = collector
    latest.update(mutation)
    with pytest.raises(ValueError, match="rolled back"):
        facts.collect(freshness_horizon_ms=100)
    assert evaluated == []
    with pytest.raises(ValueError, match="permanently latched"):
        facts.collect()


def test_latest_snapshot_still_stale_is_terminal_not_retried(collector, monkeypatch):
    module, facts, *_ = collector
    checks = []

    def stale(*a, **kw):
        checks.append(kw)
        return {"status": "UNKNOWN", "unknown_reasons": ["CURRENT_LIVE_STREAM_AGE_EXCEEDED"]}

    monkeypatch.setattr(module, "review_current_sim_navigation_health", stale)
    with pytest.raises(ValueError, match="expired or rejected"):
        facts.collect(freshness_horizon_ms=100)
    assert len(checks) == 1
    with pytest.raises(ValueError, match="permanently latched"):
        facts.collect()
