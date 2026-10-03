"""No sockets or ROS client: snapshot provenance cannot silently select default rosbridge."""

import asyncio
import json

import pytest

from rosclaw.connectors.ros.compiler import CapabilityManifestCompiler
from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot
from rosclaw.connectors.ros.provider import RosCapabilityProvider
from rosclaw.provider.core.manifest import ProviderManifest


def _snapshot(endpoint):
    return RosGraphSnapshot(
        ros_version="ros2",
        distro="jazzy",
        endpoint=endpoint,
        topics=[],
        services=[],
        actions=[],
        nodes=[],
        params=[],
        captured_at="2026-10-03T22:00:00Z",
    )


@pytest.mark.parametrize(
    "endpoint", ["fixture://saved-domain-123", "", "dds://localhost/domain-123"]
)
def test_non_rosbridge_snapshot_is_offline_and_never_invents_live_endpoint(endpoint):
    compiled = CapabilityManifestCompiler(robot_id="fixture").compile(_snapshot(endpoint=endpoint))
    assert compiled.endpoint["transport"] == "offline"
    assert compiled.endpoint["execution_eligible"] is False
    assert "host" not in compiled.endpoint and "port" not in compiled.endpoint
    assert compiled.ros["endpoint"] == endpoint


def test_explicit_rosbridge_snapshot_preserves_real_host_and_port():
    compiled = CapabilityManifestCompiler().compile(
        _snapshot(endpoint="wss://example.invalid:9443")
    )
    assert compiled.endpoint["host"] == "example.invalid"
    assert compiled.endpoint["port"] == 9443
    assert compiled.endpoint["scheme"] == "wss"


@pytest.mark.parametrize("runtime_endpoint", [None, "ws://127.0.0.1:9090"])
def test_offline_static_provider_load_and_health_do_not_create_transport(
    tmp_path, monkeypatch, runtime_endpoint
):
    compiled = CapabilityManifestCompiler().compile(
        _snapshot(endpoint="fixture://saved-domain-123")
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(compiled.to_dict()))
    runtime = {} if runtime_endpoint is None else {"endpoint": runtime_endpoint}
    provider = RosCapabilityProvider(
        ProviderManifest(
            name="ros_capability_provider",
            version="0.1.0",
            type="ros",
            runtime=runtime,
            extra={"manifest_path": str(path), "auto_discover": False},
        )
    )
    import rosclaw.connectors.ros.provider.ros_capability_provider as module

    monkeypatch.setattr(
        module,
        "RosbridgeTransport",
        lambda **_: pytest.fail("offline snapshot opened a live transport"),
    )
    asyncio.run(provider.load())
    assert provider._manifest.ros["endpoint"].startswith("fixture://")
    assert provider._transport is None
    health = asyncio.run(provider.health())
    assert not health["ok"]
    assert "METADATA_ONLY" in health["load_error"]
    with pytest.raises(ValueError, match="OFFLINE"):
        provider._create_transport()


def test_no_runtime_endpoint_is_not_implicit_localhost_transport():
    provider = RosCapabilityProvider(
        ProviderManifest(
            name="ros_capability_provider",
            version="0.1.0",
            type="ros",
            runtime={},
            extra={},
        )
    )
    with pytest.raises(ValueError, match="ENDPOINT_REQUIRED"):
        provider._create_transport()


def test_explicit_endpoint_path_still_constructs_expected_transport(monkeypatch):
    provider = RosCapabilityProvider(
        ProviderManifest(
            name="ros_capability_provider",
            version="0.1.0",
            type="ros",
            runtime={"endpoint": "ws://example.invalid:19090"},
            extra={},
        )
    )
    import rosclaw.connectors.ros.provider.ros_capability_provider as module

    monkeypatch.setattr(module, "RosbridgeTransport", lambda *, endpoint: endpoint)
    constructed = provider._create_transport()
    assert constructed.host == "example.invalid" and constructed.port == 19090


def test_pre_fix_fixture_manifest_cannot_promote_fake_default_rosbridge(tmp_path, monkeypatch):
    compiled = CapabilityManifestCompiler().compile(_snapshot("fixture://saved-domain-123"))
    compiled.endpoint = {"transport": "rosbridge", "host": "127.0.0.1", "port": 9090}
    provider = RosCapabilityProvider(
        ProviderManifest(
            name="ros_capability_provider",
            version="0.1.0",
            type="ros",
            runtime={"endpoint": "ws://127.0.0.1:9090"},
            extra={},
        )
    )
    provider._manifest = compiled
    import rosclaw.connectors.ros.provider.ros_capability_provider as module

    monkeypatch.setattr(
        module,
        "RosbridgeTransport",
        lambda **_: pytest.fail("legacy fixture touched live transport"),
    )
    with pytest.raises(ValueError, match="OFFLINE"):
        provider._create_transport()


def test_runtime_endpoint_cannot_silently_retarget_saved_graph(monkeypatch):
    provider = RosCapabilityProvider(
        ProviderManifest(
            name="ros_capability_provider",
            version="0.1.0",
            type="ros",
            runtime={"endpoint": "ws://other.invalid:9090"},
            extra={},
        )
    )
    provider._manifest = CapabilityManifestCompiler().compile(
        _snapshot("ws://original.invalid:9090")
    )
    import rosclaw.connectors.ros.provider.ros_capability_provider as module

    monkeypatch.setattr(
        module, "RosbridgeTransport", lambda **_: pytest.fail("wrong bound graph transport created")
    )
    with pytest.raises(ValueError, match="BINDING_MISMATCH"):
        provider._create_transport()


def test_matching_explicit_static_binding_still_loads_transport(tmp_path, monkeypatch):
    compiled = CapabilityManifestCompiler().compile(_snapshot("ws://example.invalid:19090"))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(compiled.to_dict()))
    provider = RosCapabilityProvider(
        ProviderManifest(
            name="ros_capability_provider",
            version="0.1.0",
            type="ros",
            runtime={"endpoint": "ws://example.invalid:19090"},
            extra={"auto_discover": False, "manifest_path": str(path)},
        )
    )
    import rosclaw.connectors.ros.provider.ros_capability_provider as module

    monkeypatch.setattr(module, "RosbridgeTransport", lambda *, endpoint: endpoint)
    asyncio.run(provider.load())
    assert provider._transport.host == "example.invalid"
    assert provider._transport.port == 19090
