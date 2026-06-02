"""Tests for ProviderLoader and GenericProvider."""

import pytest

from rosclaw.provider.core.registry import ProviderRegistry
from rosclaw.provider.loader import ProviderLoader
from rosclaw.provider.adapters.generic import GenericProvider
from rosclaw.provider.core.provider import Provider


class MyTestProvider(Provider):
    """Custom provider subclass for loader testing."""
    name = "my_test"
    version = "0.1.0"
    capabilities = ["vlm.test"]

    async def infer(self, request):
        from rosclaw.provider.core.response import ProviderResponse
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            result={},
            status="ok",
        )


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


def test_provider_loader_scan_nonexistent_directory():
    """Scanning a non-existent directory returns empty list."""
    registry = ProviderRegistry()
    loader = ProviderLoader(registry)
    loaded = loader.scan_directory("/definitely/not/a/real/path")
    assert loaded == []


def test_provider_loader_list_loaded(tmp_path):
    """list_loaded returns name -> path mapping."""
    registry = ProviderRegistry()
    loader = ProviderLoader(registry)

    yaml_path = tmp_path / "provider.yaml"
    yaml_path.write_text("""
name: list_test
version: "0.1.0"
type: vlm
capabilities: [vlm.scene_understanding]
""")

    loader.load_file(yaml_path)
    loaded = loader.list_loaded()
    assert "list_test" in loaded
    assert str(yaml_path) in loaded["list_test"]


def test_provider_loader_manifest_parse_error(tmp_path, caplog):
    """Invalid YAML is handled gracefully."""
    import logging
    registry = ProviderRegistry()
    loader = ProviderLoader(registry)

    yaml_path = tmp_path / "provider.yaml"
    yaml_path.write_text("not: valid: yaml: [")

    with caplog.at_level(logging.WARNING, logger="rosclaw.provider.loader"):
        result = loader.load_file(yaml_path)
    assert result is None
    assert "Failed to parse" in caplog.text


def test_provider_loader_register_exception(tmp_path, caplog):
    """Registry.register exception is handled gracefully."""
    import logging
    from unittest.mock import patch
    registry = ProviderRegistry()
    loader = ProviderLoader(registry)

    yaml_path = tmp_path / "provider.yaml"
    yaml_path.write_text("""
name: reg_fail
version: "0.1.0"
type: vlm
capabilities: [vlm.scene_understanding]
""")

    with caplog.at_level(logging.WARNING, logger="rosclaw.provider.loader"):
        with patch.object(registry, "register", side_effect=RuntimeError("register boom")):
            result = loader.load_file(yaml_path)
    assert result is None
    assert "Failed to register" in caplog.text


def test_provider_loader_custom_class_not_provider_subclass():
    """provider_class that is not a Provider subclass falls back."""
    from rosclaw.provider.core.manifest import ProviderManifest
    manifest = ProviderManifest.from_dict({
        "name": "not_provider",
        "version": "0.1.0",
        "type": "vlm",
        "capabilities": ["vlm.scene_understanding"],
    })
    manifest.extra = {"provider_class": "builtins.str"}
    cls = ProviderLoader._resolve_provider_class(manifest)
    assert cls is GenericProvider


def test_provider_loader_custom_class_import_error():
    """provider_class that cannot be imported falls back."""
    from rosclaw.provider.core.manifest import ProviderManifest
    manifest = ProviderManifest.from_dict({
        "name": "bad_class",
        "version": "0.1.0",
        "type": "vlm",
        "capabilities": ["vlm.scene_understanding"],
    })
    manifest.extra = {"provider_class": "nonexistent.module.Class"}
    cls = ProviderLoader._resolve_provider_class(manifest)
    assert cls is GenericProvider


def test_provider_loader_custom_class_valid(monkeypatch):
    """provider_class that is a valid Provider subclass is used."""
    from rosclaw.provider.core.manifest import ProviderManifest
    # Allow test to run when tests module is importable OR when running directly
    try:
        import tests
        import tests.test_provider_loader  # noqa: F401
    except ImportError:
        pytest.skip("tests module not importable")
    # Temporarily allow tests. prefix for provider class loading
    monkeypatch.setattr(
        ProviderLoader, "_ALLOWED_MODULE_PREFIXES", ("rosclaw.", "tests.")
    )
    manifest = ProviderManifest.from_dict({
        "name": "custom_provider",
        "version": "0.1.0",
        "type": "vlm",
        "capabilities": ["vlm.test"],
    })
    manifest.extra = {"provider_class": "tests.test_provider_loader.MyTestProvider"}
    cls = ProviderLoader._resolve_provider_class(manifest)
    assert cls is not GenericProvider
    assert cls.__name__ == "MyTestProvider"
