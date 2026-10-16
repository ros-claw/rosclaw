"""Budget semantics and private-path safeguards for original-record auditing."""

import json

import pytest

from benchmarks.ros_expert_ablation.forensics.audit import audit, shown_result
from benchmarks.ros_expert_ablation.forensics.seal import seal


def test_offload_replays_character_limit_not_utf8_byte_limit():
    value = {"text": "中" * 7000}
    raw = json.dumps(value, ensure_ascii=False)
    assert len(raw) < 16000 and len(raw.encode()) > 16000
    assert shown_result({"turn": 2, "tool_result": value}) == value
    larger = {"text": "x" * 16001}
    assert shown_result({"turn": 2, "tool_result": larger})["file"] == "tool-result-02.json"


def test_audit_rejects_output_inside_originals_before_touching_them(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    marker = original / "marker"
    marker.write_text("immutable")
    with pytest.raises(ValueError, match="outside"):
        audit(original, original / "analysis")
    assert marker.read_text() == "immutable" and not (original / "analysis").exists()


def test_public_seal_refuses_credentials(tmp_path):
    study = tmp_path / "study"
    study.mkdir()
    (study / "auth.json").write_text("not-public")
    with pytest.raises(ValueError, match="credential"):
        seal(study, tmp_path, tmp_path / "public")
    assert not list((tmp_path / "public/blobs").iterdir())


def test_public_seal_refuses_symlinked_original(tmp_path):
    study = tmp_path / "study"
    study.mkdir()
    secret = tmp_path / "private"
    secret.write_text("not-public")
    (study / "events.json").symlink_to(secret)
    with pytest.raises(ValueError, match="linked"):
        seal(study, tmp_path, tmp_path / "public")
    assert not list((tmp_path / "public/blobs").iterdir())
