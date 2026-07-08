"""Tests for ArtifactStore."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from rosclaw.practice.artifact_store import ArtifactRecord, ArtifactStore


@pytest.fixture
def tmp_store(tmp_path: Path):
    return ArtifactStore(tmp_path)


def test_write_jsonl_creates_file_and_manifest(tmp_store: ArtifactStore, tmp_path: Path):
    records = [{"event_id": "evt_1", "type": "a"}, {"event_id": "evt_2", "type": "b"}]
    record = tmp_store.write_jsonl(
        "art_events_1", records, session_id="sess_1", episode_id="ep_1"
    )
    assert isinstance(record, ArtifactRecord)
    assert record.artifact_type == "events"
    assert record.session_id == "sess_1"
    assert record.episode_id == "ep_1"
    assert Path(record.path).exists()

    manifest_path = tmp_path / "sessions" / "sess_1" / "episodes" / "ep_1" / "artifact_manifest.yaml"
    assert manifest_path.exists()
    with open(manifest_path, encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    assert manifest["artifacts"]["art_events_1"]["sha256"] == record.sha256


def test_write_yaml_snapshot(tmp_store: ArtifactStore):
    data = {"body_id": "body_rh56_left", "force_model": {"thumb": 50.0}}
    record = tmp_store.write_yaml(
        "art_body_snapshot",
        data,
        session_id="sess_1",
        episode_id="ep_1",
        artifact_type="body_snapshot",
    )
    assert record.artifact_type == "body_snapshot"
    path = Path(record.path)
    assert path.suffix == ".yaml"
    with open(path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert loaded["body_id"] == "body_rh56_left"


def test_write_parquet_from_records(tmp_store: ArtifactStore):
    pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    records = [
        {"timestamp": 1.0, "thumb_force": 100.0, "meta": {"ok": True}},
        {"timestamp": 2.0, "thumb_force": 120.0, "meta": {"ok": False}},
    ]
    record = tmp_store.write_parquet(
        "art_summary",
        records,
        session_id="sess_1",
        episode_id="ep_1",
        artifact_type="summary",
    )
    assert Path(record.path).exists()
    assert record.artifact_type == "summary"
    table = pq.read_table(record.path)
    assert table.num_rows == 2


def test_idempotent_write_skips_manifest_update(tmp_store: ArtifactStore):
    records = [{"event_id": "evt_1"}]
    r1 = tmp_store.write_jsonl("art_idem", records, session_id="sess_1")
    r2 = tmp_store.write_jsonl("art_idem", records, session_id="sess_1")
    assert r1.sha256 == r2.sha256
    assert r1.created_at == r2.created_at


def test_list_artifacts(tmp_store: ArtifactStore):
    tmp_store.write_jsonl("a1", [{"x": 1}], session_id="sess_1", episode_id="ep_1")
    tmp_store.write_yaml("a2", {"y": 2}, session_id="sess_1", episode_id="ep_1")
    artifacts = tmp_store.list_artifacts("sess_1", "ep_1")
    assert len(artifacts) == 2
    assert {a.artifact_id for a in artifacts} == {"a1", "a2"}


def test_verify_artifact_passes_and_fails_on_tamper(tmp_store: ArtifactStore):
    tmp_store.write_jsonl("a1", [{"x": 1}], session_id="sess_1", episode_id="ep_1")
    ok, msg = tmp_store.verify_artifact("a1", "sess_1", "ep_1")
    assert ok is True
    assert msg == "ok"

    path = tmp_store.get_artifact("a1", "sess_1", "ep_1").path
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"tamper": True}) + "\n")

    ok, msg = tmp_store.verify_artifact("a1", "sess_1", "ep_1")
    assert ok is False
    assert "sha256 mismatch" in msg


def test_manifest_schema_version(tmp_store: ArtifactStore):
    tmp_store.write_jsonl("a1", [{"x": 1}], session_id="sess_1")
    manifest_path = tmp_store.manifest_path("sess_1")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    assert manifest["schema_version"] == ArtifactStore.SCHEMA_VERSION
    assert manifest["session_id"] == "sess_1"


def test_get_artifact_missing_returns_none(tmp_store: ArtifactStore):
    assert tmp_store.get_artifact("missing", "sess_1") is None
