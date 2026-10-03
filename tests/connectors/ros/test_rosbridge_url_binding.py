"""Proxy websocket routes are preserved and bound; no actual transport connection."""

import pytest

from rosclaw.connectors.ros.compiler import CapabilityManifestCompiler
from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot
from rosclaw.connectors.ros.provider import RosCapabilityProvider
from rosclaw.connectors.ros.transport import RosbridgeEndpoint
from rosclaw.provider.core.manifest import ProviderManifest


@pytest.mark.parametrize(
    "url",
    [
        "wss://example.invalid:9443/robot/rosbridge",
        "ws://example.invalid:19090/rosbridge?domain=fixture",
        "ws://[::1]:19090/rosbridge",
    ],
)
def test_endpoint_parser_roundtrips_complete_websocket_route(url):
    assert RosbridgeEndpoint.from_url(url).url == url


def _provider(url, source):
    snapshot = RosGraphSnapshot(
        ros_version="ros2",
        distro="jazzy",
        endpoint=source,
        topics=[],
        services=[],
        actions=[],
        nodes=[],
        params=[],
        captured_at="fixture",
    )
    provider = RosCapabilityProvider(
        ProviderManifest(
            name="ros_capability_provider",
            version="0.1.0",
            type="ros",
            runtime={"endpoint": url},
            extra={},
        )
    )
    provider._manifest = CapabilityManifestCompiler().compile(snapshot)
    return provider


def test_matching_proxy_route_is_preserved_in_manifest_and_transport(monkeypatch):
    url = "wss://example.invalid:9443/robot/rosbridge?domain=fixture"
    provider = _provider(url, url)
    assert provider._manifest.endpoint["path"] == "/robot/rosbridge"
    assert provider._manifest.endpoint["query"] == "domain=fixture"
    import rosclaw.connectors.ros.provider.ros_capability_provider as module

    monkeypatch.setattr(module, "RosbridgeTransport", lambda *, endpoint: endpoint)
    assert provider._create_transport().url == url


@pytest.mark.parametrize(
    "requested",
    [
        "ws://example.invalid:19090/robot-b",
        "ws://example.invalid:19090/robot-a?domain=other",
    ],
)
def test_same_host_port_different_route_or_query_cannot_retarget_graph(requested, monkeypatch):
    provider = _provider(requested, "ws://example.invalid:19090/robot-a?domain=fixture")
    import rosclaw.connectors.ros.provider.ros_capability_provider as module

    monkeypatch.setattr(
        module, "RosbridgeTransport", lambda **_: pytest.fail("wrong route created transport")
    )
    with pytest.raises(ValueError, match="BINDING_MISMATCH"):
        provider._create_transport()


@pytest.mark.parametrize(
    "url",
    [
        "fixture://offline",
        "http://example.invalid",
        "ws://",
        "ws://host:0",
        "ws://user:password@example.invalid:9090",
    ],
)
def test_invalid_or_credential_embedded_transport_urls_rejected(url):
    with pytest.raises(ValueError):
        RosbridgeEndpoint.from_url(url)
