"""Tests for ProviderLoader and GenericProvider."""

import pytest
from pathlib import Path

from rosclaw.provider.core.registry import ProviderRegistry
from rosclaw.provider.loader import ProviderLoader
from rosclaw.provider.adapters.generic import GenericProvider


def test_provider_loader_scan_empty_directory(tmp_path):
    """Scanning an empty directory returns no providers."""
    registry = ProviderRegistry()
    loader = ProviderLoader(registry)
    loaded = loader.scan_directory(tmp_path)
    assert loaded == []


def test_provider_loader_load_from_yaml(tmp_path):
    """Load a provider from a valid provider.yaml file."""
    registry = ProviderRegistry()
    loader = ProviderLoader(registry)

    # Create a test provider directory
    provider_dir = tmp_path / "test_vlm"
    provider_dir.mkdir()
    yaml_path = provider_dir / "provider.yaml"
    yaml_path.write_text("""
name: test_http_vlm
version: "0.1.0"
type: vlm
capabilities:
  - vlm.object_grounding
  - vlm.scene_understanding
modalities:
  input: [image, text]
  output: [object_list]
runtime:
  backend: http
  endpoint: http://localhost:11434/api/generate
  device: cpu
safety:
  executable: false
  requires_guard: false
""")

    loaded = loader.load_file(yaml_path)
    assert loaded == "test_http_vlm"
    assert "test_http_vlm" in registry.list_providers()

    manifest = registry.get_manifest("test_http_vlm")
    assert manifest.type == "vlm"
    assert "vlm.object_grounding" in manifest.capabilities
    assert manifest.runtime.backend == "http"
    assert manifest.runtime.endpoint == "http://localhost:11434/api/generate"


def test_provider_loader_scan_directory(tmp_path):
    """Scan a directory tree with multiple providers."""
    registry = ProviderRegistry()
    loader = ProviderLoader(registry)

    # Provider 1
    d1 = tmp_path / "vlm"
    d1.mkdir()
    (d1 / "provider.yaml").write_text("""
name: scan_vlm
version: "0.1.0"
type: vlm
capabilities: [vlm.scene_understanding]
""")

    # Provider 2 (nested)
    d2 = tmp_path / "skills" / "grasp"
    d2.mkdir(parents=True)
    (d2 / "provider.yaml").write_text("""
name: scan_skill
version: "0.1.0"
type: skill
capabilities: [skill.grasp]
""")

    loaded = loader.scan_directory(tmp_path)
    assert set(loaded) == {"scan_vlm", "scan_skill"}


def test_provider_loader_duplicate_skipped(tmp_path):
    """Duplicate provider names are skipped."""
    registry = ProviderRegistry()
    loader = ProviderLoader(registry)

    yaml_path = tmp_path / "provider.yaml"
    yaml_path.write_text("""
name: dup_test
version: "0.1.0"
type: vlm
capabilities: [vlm.scene_understanding]
""")

    loader.load_file(yaml_path)
    # Second load should be skipped
    result = loader.load_file(yaml_path)
    assert result is None


def test_provider_loader_unload(tmp_path):
    """Unload removes provider from registry."""
    registry = ProviderRegistry()
    loader = ProviderLoader(registry)

    yaml_path = tmp_path / "provider.yaml"
    yaml_path.write_text("""
name: unload_test
version: "0.1.0"
type: vlm
capabilities: [vlm.scene_understanding]
""")

    loader.load_file(yaml_path)
    assert "unload_test" in registry.list_providers()

    loader.unload("unload_test")
    assert "unload_test" not in registry.list_providers()


@pytest.mark.asyncio
async def test_generic_provider_http_runtime_creation():
    """GenericProvider creates HTTPRuntime from manifest."""
    from rosclaw.provider.core.manifest import ProviderManifest

    manifest = ProviderManifest.from_dict({
        "name": "generic_http",
        "version": "0.1.0",
        "type": "vlm",
        "capabilities": ["vlm.object_grounding"],
        "runtime": {
            "backend": "http",
            "endpoint": "http://localhost:11434/api/generate",
            "env": {"timeout_sec": "5.0", "retries": "2"},
        },
    })

    provider = GenericProvider(manifest)
    assert provider.name == "generic_http"
    assert provider._runtime is not None
    assert provider._runtime.name == "generic_http"


@pytest.mark.asyncio
async def test_generic_provider_python_runtime_creation():
    """GenericProvider creates PythonRuntime from manifest."""
    from rosclaw.provider.core.manifest import ProviderManifest

    manifest = ProviderManifest.from_dict({
        "name": "generic_python",
        "version": "0.1.0",
        "type": "skill",
        "capabilities": ["skill.grasp"],
        "runtime": {"backend": "python"},
    })

    provider = GenericProvider(manifest)
    assert provider.name == "generic_python"
    assert provider._runtime is not None


@pytest.mark.asyncio
async def test_generic_provider_no_runtime():
    """GenericProvider handles manifest with no runtime backend."""
    from rosclaw.provider.core.manifest import ProviderManifest

    manifest = ProviderManifest.from_dict({
        "name": "meta_only",
        "version": "0.1.0",
        "type": "critic",
        "capabilities": ["critic.success_detection"],
    })

    provider = GenericProvider(manifest)
    assert provider._runtime is None

    # infer() should fail gracefully
    from rosclaw.provider.core.request import ProviderRequest
    req = ProviderRequest(request_id="r1", capability="critic.success_detection", inputs={})
    with pytest.raises(Exception):
        await provider.infer(req)


def test_provider_loader_custom_class_fallback(tmp_path):
    """Invalid provider_class falls back to GenericProvider."""
    registry = ProviderRegistry()
    loader = ProviderLoader(registry)

    yaml_path = tmp_path / "provider.yaml"
    yaml_path.write_text("""
name: bad_class
version: "0.1.0"
type: vlm
capabilities: [vlm.scene_understanding]
extra:
  provider_class: "nonexistent.module.Class"
""")

    # Should not raise; falls back to GenericProvider
    loaded = loader.load_file(yaml_path)
    assert loaded == "bad_class"
