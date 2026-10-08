"""Live-prefix orchestration tests do not establish dynamic physical acceptance."""

import importlib.util
import json
from pathlib import Path

import pytest

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest

ROOT = Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance"
spec = importlib.util.spec_from_file_location("audit_cursor", ROOT / "audit_cursor.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(sequence, previous=None):
    value = {
        "schema_version": "rosclaw.coverage_audit_event.v1",
        "run_id": "run",
        "sequence": sequence,
        "previous_hash": previous,
        "kind": "physics_snapshot_received",
        "payload": {"complete": True},
    }
    value["artifact_sha256"] = digest(value)
    return value


def encoded(value):
    return (json.dumps(value) + "\n").encode()


def test_fragmented_append_preserves_contiguous_chain_without_final_claim(tmp_path):
    path = tmp_path / "live.jsonl"
    first = row(1)
    second = row(2, first["artifact_sha256"])
    path.write_bytes(encoded(first) + encoded(second)[:15])
    cursor = module.AuditCursor(path, run_id="run")
    assert cursor.poll() == [first]
    assert cursor.poll() == []
    with path.open("ab") as stream:
        stream.write(encoded(second)[15:])
    assert cursor.poll() == [second]
    assert cursor.status()["evidence_role"] == "LIVE_PREFIX_NOT_FINAL_AUDIT"
    assert cursor.status()["physical_acceptance"] == "NOT_RUN"


@pytest.mark.parametrize(
    "fault", ["json", "hash", "run", "gap", "repeat", "truncated", "replaced", "oversized"]
)
def test_invalid_completed_or_replaced_source_latches_after_good_row(tmp_path, fault):
    path = tmp_path / "live.jsonl"
    first = row(1)
    path.write_bytes(encoded(first))
    cursor = module.AuditCursor(path, run_id="run", read_limit=1024)
    assert cursor.poll() == [first]
    second = row(2, first["artifact_sha256"])
    if fault == "json":
        addition = b"broken\n"
    elif fault == "hash":
        second["payload"]["complete"] = False
        addition = encoded(second)
    elif fault == "run":
        second["run_id"] = "other"
        addition = encoded(second)
    elif fault == "gap":
        addition = encoded(row(3, first["artifact_sha256"]))
    elif fault == "repeat":
        addition = encoded(first)
    elif fault == "oversized":
        addition = b"x" * 2048
    else:
        addition = b""
    if fault == "truncated":
        path.write_bytes(b"")
    elif fault == "replaced":
        path.rename(tmp_path / "old")
        path.write_bytes(encoded(first))
    else:
        with path.open("ab") as stream:
            stream.write(addition)
    with pytest.raises(ValueError):
        cursor.poll()
    with path.open("ab") as stream:
        stream.write(encoded(second))
    with pytest.raises(ValueError, match="latched"):
        cursor.poll()
