"""Current Native journeys use append logs rather than overwritten snapshots."""

import importlib.util
from pathlib import Path

import pytest


def load(monkeypatch):
    root = Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location(
        "native_acceptance_paths", root / "cleaning_acceptance.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_append_only_native_evidence_does_not_require_removed_snapshot_files(tmp_path, monkeypatch):
    module = load(monkeypatch)
    stream = tmp_path / "plan-events-123.jsonl"
    stream.write_text('{"kind":"path"}\n')
    assert module.native_path_sources(tmp_path) == [stream]
    assert not (tmp_path / "coverage_path.json").exists()
    assert not (tmp_path / "navigation_path.json").exists()


def test_missing_or_partial_legacy_evidence_cannot_be_silently_accepted(tmp_path, monkeypatch):
    module = load(monkeypatch)
    with pytest.raises(RuntimeError, match="missing"):
        module.native_path_sources(tmp_path)
    (tmp_path / "coverage_path.json").write_text("{}")
    with pytest.raises(RuntimeError, match="missing"):
        module.native_path_sources(tmp_path)
    (tmp_path / "navigation_path.json").write_text("{}")
    assert len(module.native_path_sources(tmp_path)) == 2
