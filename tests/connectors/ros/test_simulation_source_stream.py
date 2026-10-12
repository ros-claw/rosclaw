"""Append-only, gap, budget and epoch contracts for the passive SDK reader."""

import hashlib
import importlib
import json

import pytest

from tests.connectors.ros.test_simulation_source_continuity import advance, setup_guard


def stream_fixture(tmp_path, monkeypatch, *, samples=3, segments=4):
    _, sample, _, kwargs = setup_guard(tmp_path, monkeypatch)
    module = importlib.import_module("simulation_source_stream")
    pins = module.SourceStreamPins("b" * 32, 123, samples, 1_000_000, segments)
    sample["sample"] = 0
    sample["source_stream"] = pins.header(0, 0, "")
    raw = encode(sample)
    (tmp_path / "ecm.jsonl").write_bytes(raw)
    kwargs = {k: v for k, v in kwargs.items() if k not in {"expected_sha256", "now_monotonic_ns"}}
    reader = module.SimulationSourceStreamReader(
        tmp_path,
        "ecm.jsonl",
        pins=pins,
        accepted_prefix=raw,
        expected_prefix_sha256=hashlib.sha256(raw).hexdigest(),
        guard_inputs=kwargs,
        now_monotonic_ns=sample["observer_monotonic_ns"] + 1,
    )
    monkeypatch.setattr(reader, "_test_sample", sample, raising=False)
    return reader, sample, module


def encode(sample):
    return json.dumps(sample).encode() + b"\n"


def append(path, sample):
    with path.open("ab") as f:
        f.write(encode(sample))


def poll(reader, sample, extra=1):
    return reader.poll(now_monotonic_ns=sample["observer_monotonic_ns"] + extra)


def test_append_partial_eof_rollover_and_sealed_sha_chain(tmp_path, monkeypatch):
    reader, first, _ = stream_fixture(tmp_path, monkeypatch)
    try:
        second = advance(first)
        raw = encode(second)
        with (tmp_path / "ecm.jsonl").open("ab") as f:
            f.write(raw[:100])
        assert poll(reader, first)["continuous"]
        with (tmp_path / "ecm.jsonl").open("ab") as f:
            f.write(raw[100:])
        assert poll(reader, second)["continuous"]
        third = advance(second)
        append(tmp_path / "ecm.jsonl", third)
        assert poll(reader, third)["continuous"]
        previous = hashlib.sha256((tmp_path / "ecm.jsonl").read_bytes()).hexdigest()
        fourth = advance(third)
        fourth["source_stream"] = reader.pins.header(1, 3, previous)
        (tmp_path / "ecm.jsonl.segment-000001.jsonl").write_bytes(encode(fourth))
        report = poll(reader, fourth)
        assert report["continuous"] and report["segment_index"] == 1
        assert report["segment_chain"][0]["sha256"] == previous
        assert report["execution_authority"] == "NONE" and report["L2"] == "NOT_RUN"
        assert reader.capacity_covers(duration_ms=100, max_sample_bytes=100_000)
        assert not reader.capacity_covers(duration_ms=10_000, max_sample_bytes=500_000)
    finally:
        reader.close()


@pytest.mark.parametrize(
    "fault",
    [
        "stale",
        "bad_line",
        "sample_gap",
        "epoch",
        "pid",
        "budget",
        "hash",
        "truncated",
        "rewritten",
        "replaced",
        "symlink",
        "gap",
        "overflow",
        "partial_overflow",
        "clock_gap",
        "parent",
        "capacity",
        "sealed_mutation",
        "rollover_partial",
        "closed",
    ],
)
def test_every_stream_fault_latches_without_rearm(tmp_path, monkeypatch, fault):
    reader, first, _ = stream_fixture(
        tmp_path, monkeypatch, segments=1 if fault == "capacity" else 4
    )
    sample = advance(first)
    path = tmp_path / "ecm.jsonl"
    now = sample["observer_monotonic_ns"] + 1
    if fault == "stale":
        now += 500_000_000
    elif fault == "bad_line":
        with path.open("ab") as f:
            f.write(b"{broken}\n")
    elif fault == "sample_gap":
        sample["sample"] += 1
    elif fault in {"epoch", "pid", "budget", "hash"}:
        key, value = {
            "epoch": ("observer_epoch", "c" * 32),
            "pid": ("observer_pid", 124),
            "budget": ("max_segments", 5),
            "hash": ("previous_segment_sha256", "f" * 64),
        }[fault]
        sample["source_stream"][key] = value
    elif fault == "truncated":
        path.write_bytes(b"")
    elif fault == "rewritten":
        data = path.read_bytes()
        path.write_bytes(data.replace(b'"sample": 0', b'"sample": 9'))
    elif fault in {"replaced", "symlink"}:
        path.rename(tmp_path / "original")
        if fault == "replaced":
            path.write_bytes((tmp_path / "original").read_bytes())
        else:
            path.symlink_to(tmp_path / "original")
    elif fault == "gap":
        (tmp_path / "ecm.jsonl.segment-000002.jsonl").write_bytes(b"")
    elif fault == "overflow":
        path.write_bytes(path.read_bytes() + b" " * 1_000_001)
    elif fault == "partial_overflow":
        with path.open("ab") as f:
            f.write(b" " * 500_001)
    elif fault == "clock_gap":
        sample["observer_monotonic_ns"] += 500_000_000
        now = sample["observer_monotonic_ns"] + 1
    elif fault == "parent":
        sample["collisions"][0]["ancestry"][0]["entity"] += 10
    elif fault == "capacity":
        append(path, sample)
        sample = advance(sample)
        now = sample["observer_monotonic_ns"] + 1
    elif fault in {"sealed_mutation", "rollover_partial"}:
        previous = hashlib.sha256(path.read_bytes()).hexdigest()
        sample["source_stream"] = reader.pins.header(1, 1, previous)
        (tmp_path / "ecm.jsonl.segment-000001.jsonl").write_bytes(encode(sample))
        if fault == "sealed_mutation":
            assert poll(reader, sample)["continuous"]
            with path.open("ab") as f:
                f.write(b" ")
        else:
            with path.open("ab") as f:
                f.write(b"{partial")
    elif fault == "closed":
        reader.close()
    if fault in {"sample_gap", "epoch", "pid", "budget", "hash", "clock_gap", "parent", "capacity"}:
        append(path, sample)
    report = reader.poll(now_monotonic_ns=now)
    assert report["terminal"] and not report["continuous"], (fault, report)
    assert reader.poll(now_monotonic_ns=first["observer_monotonic_ns"])["terminal"]
    assert not reader.capacity_covers(duration_ms=100, max_sample_bytes=1000)
    reader.close()


def test_pin_and_prefix_validation(tmp_path, monkeypatch):
    reader, _, module = stream_fixture(tmp_path, monkeypatch)
    try:
        with pytest.raises(ValueError):
            module.SourceStreamPins("invalid", 123, 3, 1_000_000, 4)
        with pytest.raises(ValueError):
            module.SimulationSourceStreamReader(
                tmp_path,
                "../ecm.jsonl",
                pins=reader.pins,
                accepted_prefix=b"{}\n",
                expected_prefix_sha256="0" * 64,
                guard_inputs={},
                now_monotonic_ns=0,
            )
    finally:
        reader.close()
