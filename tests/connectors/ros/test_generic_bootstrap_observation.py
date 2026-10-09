"""Real file checkpoints with fake read-only observations; no SDK or World."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def observation(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("generic_bootstrap_observation")


def window(observation, output, *, fail_at=None, snapshot=None):
    clock = [100.0]
    spins = []

    def spin(seconds):
        assert 0 < seconds <= 0.1
        spins.append(seconds)
        clock[0] += seconds
        if fail_at is not None and clock[0] >= fail_at:
            raise KeyboardInterrupt("outer supervisor interruption")

    observation.observe_window(
        seconds=3,
        output=output,
        discover=lambda: None,
        spin_once=spin,
        snapshot=snapshot or (lambda: {"received_messages": len(spins)}),
        monotonic=lambda: clock[0],
    )
    return clock, spins


def checkpoint(output):
    raw = output.read_bytes()
    meta = json.loads(output.with_suffix(".observation.json").read_text())
    assert meta["snapshot_sha256"] == hashlib.sha256(raw).hexdigest()
    assert meta["controller_activation"] is False
    assert meta["Body_admitted"] is False
    assert meta["physical_acceptance"] == "NOT_EVALUATED"
    return json.loads(raw), meta


def test_complete_window_preserves_fixed_deadline_and_original_snapshot(observation, tmp_path):
    output = tmp_path / "discovery.json"
    clock, spins = window(observation, output)
    original, meta = checkpoint(output)
    assert clock[0] == 103
    assert sum(spins) == pytest.approx(3)
    assert original == {"received_messages": len(spins)}
    assert meta["deadline_monotonic_sec"] == 103
    assert meta["started_monotonic_sec"] == 100
    assert meta["checkpoint_count"] == 3
    assert meta["window_status"] == "COMPLETE"
    assert not list(tmp_path.glob("*.tmp"))
    history = output.with_suffix(".checkpoints")
    previous_counts = []
    for index in range(1, 4):
        original, recorded = checkpoint(history / f"{index:03d}.json")
        previous_counts.append(original["received_messages"])
        assert recorded["checkpoint_count"] == index
        assert recorded["deadline_monotonic_sec"] == 103
        assert recorded["window_status"] == ("COMPLETE" if index == 3 else "IN_PROGRESS")
    assert previous_counts == sorted(set(previous_counts))


def test_interruption_retains_last_hash_verified_incomplete_window(observation, tmp_path):
    output = tmp_path / "discovery.json"
    with pytest.raises(KeyboardInterrupt):
        window(observation, output, fail_at=102.5)
    _, meta = checkpoint(output)
    assert meta["window_status"] == "IN_PROGRESS"
    assert 2 <= meta["observed_duration_sec"] < 3
    assert meta["deadline_monotonic_sec"] == 103


def test_snapshot_failure_preserves_previous_checkpoint(observation, tmp_path):
    output = tmp_path / "discovery.json"
    calls = 0

    def snapshot():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError("invalid sensor value")
        return {"original": 1}

    with pytest.raises(ValueError, match="sensor"):
        window(observation, output, snapshot=snapshot)
    original, meta = checkpoint(output)
    assert original == {"original": 1}
    assert meta["checkpoint_count"] == 1
    assert meta["window_status"] == "IN_PROGRESS"


@pytest.mark.parametrize("seconds", [True, False, 0, -1, 61, "3", float("nan"), float("inf")])
def test_invalid_window_refused_before_any_observation(observation, tmp_path, seconds):
    def forbidden(*args):
        pytest.fail("invalid observation must not start")

    with pytest.raises(ValueError):
        observation.observe_window(
            seconds=seconds,
            output=tmp_path / "out.json",
            discover=forbidden,
            spin_once=forbidden,
            snapshot=forbidden,
            monotonic=forbidden,
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("existing", ["snapshot", "sidecar", "symlink", "history"])
def test_existing_evidence_cannot_be_overwritten(observation, tmp_path, existing):
    output = tmp_path / "discovery.json"
    if existing == "history":
        output.with_suffix(".checkpoints").mkdir()
        with pytest.raises(ValueError, match="fresh"):
            window(observation, output)
        return
    target = output if existing == "snapshot" else output.with_suffix(".observation.json")
    if existing == "symlink":
        target.symlink_to(tmp_path / "absent.json")
    else:
        target.write_text("retained original")
    with pytest.raises(ValueError, match="fresh"):
        window(observation, output)
    if existing != "symlink":
        assert target.read_text() == "retained original"


@pytest.mark.parametrize("path", ["relative.json", "/evidence/out.json"])
def test_source_and_relative_output_refused(observation, path):
    with pytest.raises(ValueError):
        observation.observation_output(path)


def test_symlink_parent_cannot_alias_immutable_or_other_output(observation, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError):
        observation.observation_output(alias / "out.json")
