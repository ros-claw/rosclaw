"""Exact upstream source patch and refusal contracts, never SDK execution."""

import gzip
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def source(tmp_path):
    script = ROOT / "integrations/ros_probe/acceptance/control_metadata/patch_import_metadata.py"
    spec = importlib.util.spec_from_file_location("control_metadata_patch", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = gzip.decompress(
        (ROOT / "tests/fixtures/ros/control_metadata/resource_manager_4.48.1.cpp.gz").read_bytes()
    )
    path = tmp_path / "resource_manager.cpp"
    path.write_bytes(raw)
    return module, path, raw


def test_metadata_patch_can_be_undone_to_the_exact_unmodified_control_implementation(source):
    module, path, original = source
    report = module.patch_source(path)
    patched = path.read_bytes()
    assert patched.count(module.METADATA.encode()) == 1
    assert patched.replace(module.METADATA.encode(), b"") == original
    assert report["original_sha256"] == module.ORIGINAL_SHA256
    assert report["patched_sha256"] != report["original_sha256"]


def test_repeated_patch_refused_without_changing_already_patched_source(source):
    module, path, _ = source
    module.patch_source(path)
    patched = path.read_bytes()
    with pytest.raises(ValueError, match="exact original"):
        module.patch_source(path)
    assert path.read_bytes() == patched


def test_different_upstream_source_refused_without_modification(source):
    module, path, original = source
    altered = original + b"\n// different source revision\n"
    path.write_bytes(altered)
    with pytest.raises(ValueError, match="exact original"):
        module.patch_source(path)
    assert path.read_bytes() == altered


def test_symlink_cannot_redirect_patch_to_another_source_file(source, tmp_path):
    module, path, original = source
    alias = tmp_path / "alias.cpp"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="exact original"):
        module.patch_source(alias)
    assert path.read_bytes() == original
