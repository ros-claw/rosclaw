"""Offline tests for the actual pinned proxy, never a ROS/provider invocation.

The full captured proxy is embedded below (including its original BSD license)
so these tests need no host ROS, network or external history packet. Its full
SHA is checked before it is copied into an owned temporary image fixture.
"""

from __future__ import annotations

import hashlib
import importlib.util
import stat
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "humble_action_compat_installer", ROOT / "docker/rosapi_humble_action_type_compat.py"
)
assert SPEC is not None and SPEC.loader is not None
compat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compat)

# Captured installed proxy, SHA7042b78b...; NOT a reimplemented get_action_type.
PINNED_PROXY = b'''# Software License Agreement (BSD License)
#
# Copyright (c) 2012, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

from typing import TYPE_CHECKING

from ros2action.api import get_action_names_and_types
from ros2interface.api import type_completer
from ros2node.api import (
    get_node_names,
    get_publisher_info,
    get_service_server_info,
    get_subscriber_info,
)
from ros2service.api import get_service_names, get_service_names_and_types
from ros2topic.api import get_topic_names, get_topic_names_and_types

from .glob_helper import any_match, filter_globs

if TYPE_CHECKING:
    from collections.abc import Callable

    from rclpy.node import Node

_node: Node | None = None


def init(node: Node) -> None:
    """
    Initialize proxy module with a rclpy.node.Node for further use.

    This function has to be called before any other for the module to work.

    :param node: The rclpy node to use for service calls.
    :type node: Node
    """
    global _node
    _node = node


def get_topics(
    topics_pub_glob: list[str] | None,
    topics_sub_glob: list[str] | None,
    include_hidden: bool = False,
) -> list[str]:
    """Return a list of all the active topics in the ROS system matching pub or sub globs."""
    topic_names = get_topic_names(node=_node, include_hidden_topics=include_hidden)

    # Determine overall topic visibility
    if topics_pub_glob is None or topics_sub_glob is None:
        combined_globs = None
    else:
        combined_globs = list(set(topics_pub_glob + topics_sub_glob))

    # Sort for deterministic output.
    return sorted(filter_globs(combined_globs, topic_names))


def get_interfaces() -> list[str]:
    """Return a list of all the types in the ROS system."""
    return type_completer()


def get_topics_and_types(
    topics_pub_glob: list[str] | None,
    topics_sub_glob: list[str] | None,
    include_hidden: bool = False,
) -> tuple[list[str], list[str]]:
    """Return filtered allowed topics and their types."""
    if topics_pub_glob is None or topics_sub_glob is None:
        combined_globs = None
    else:
        combined_globs = list(set(topics_pub_glob + topics_sub_glob))

    return get_publications_and_types(
        combined_globs, get_topic_names_and_types, include_hidden_topics=include_hidden
    )


def get_topics_for_type(
    topic_type: str,
    topics_pub_glob: list[str] | None,
    topics_sub_glob: list[str] | None,
    include_hidden: bool = False,
) -> list[str]:
    topic_names_and_types = get_topic_names_and_types(
        node=_node, include_hidden_topics=include_hidden
    )
    # topic[0] has the topic name and topic[1] has the type wrapped in a list.
    topics_for_type = [topic[0] for topic in topic_names_and_types if topic[1][0] == topic_type]

    if topics_pub_glob is None or topics_sub_glob is None:
        combined_globs = None
    else:
        combined_globs = list(set(topics_pub_glob + topics_sub_glob))

    return filter_globs(combined_globs, topics_for_type)


def get_services(services_glob: list[str] | None, include_hidden: bool = False) -> list[str]:
    """Return a list of all the services advertised in the ROS system."""
    # Filter the list of services by whether they are public before returning.
    service_names = get_service_names(node=_node, include_hidden_services=include_hidden)
    return filter_globs(services_glob, service_names)


def get_services_and_types(
    services_glob: list[str] | None, include_hidden: bool = False
) -> tuple[list[str], list[str]]:
    return get_publications_and_types(
        services_glob,
        get_service_names_and_types,
        include_hidden_services=include_hidden,
    )


def get_services_for_type(
    service_type: str, services_glob: list[str] | None, include_hidden: bool = False
) -> list[str]:
    """Return a list of services as specific service type."""
    # Filter the list of services by whether they are public before returning.
    services_names_and_types = get_service_names_and_types(
        node=_node, include_hidden_services=include_hidden
    )
    # service[0] has the topic name and service[1] has the type wrapped in a list.
    services_for_type = [
        service[0] for service in services_names_and_types if service[1][0] == service_type
    ]
    return filter_globs(services_glob, services_for_type)


def get_publications_and_types(
    glob: list[str] | None,
    getter_function: Callable,
    **include_hidden_publications: bool,
) -> tuple[list[str], list[str]]:
    """
    Get a list of topic or service publications and their types.

    :param glob: A glob pattern to filter the publications.
    :type glob: list[str] | None
    :param getter_function: A function to get the names and types of publications.
    :type getter_function: Callable
    :param include_hidden_publications: Keyword arguments to specify whether to include hidden publications.
    :type include_hidden_publications: dict
    :return: A tuple of filtered publication names and their types.
    :rtype: tuple
    """
    publication_names_and_types = getter_function(node=_node, **include_hidden_publications)
    # publication[0] has the publication name and publication[1] has the type wrapped in a list.
    all_publications = [publication[0] for publication in publication_names_and_types]
    filtered_publications = filter_globs(glob, all_publications)
    filtered_publication_types = [
        publication[1][0]
        for publication in publication_names_and_types
        if publication[0] in filtered_publications
    ]
    return filtered_publications, filtered_publication_types


def get_nodes(include_hidden: bool = False) -> list[str]:
    """Return a list of all the nodes registered in the ROS system."""
    node_names = get_node_names(node=_node, include_hidden_nodes=include_hidden)
    return [node_name.full_name for node_name in node_names]


def get_node_info(
    node_name: str, include_hidden: bool = False
) -> tuple[list[str], list[str], list[str]] | None:
    node_names = get_node_names(node=_node, include_hidden_nodes=include_hidden)
    if node_name in [n.full_name for n in node_names]:
        # Only the name of each item is required as output.
        subscribers = get_node_subscriptions(node_name)
        publishers = get_node_publications(node_name)
        services = get_node_services(node_name)

        return subscribers, publishers, services
    return None


def get_node_publications(node_name: str) -> list[str]:
    """Return a list of topic names that are being published by the specified node."""
    publishers = get_publisher_info(node=_node, remote_node_name=node_name)
    return [publisher.name for publisher in publishers]


def get_node_subscriptions(node_name: str) -> list[str]:
    """Return a list of topic names that are being subscribed by the specified node."""
    subscribers = get_subscriber_info(node=_node, remote_node_name=node_name)
    return [subscriber.name for subscriber in subscribers]


def get_node_services(node_name: str) -> list[str]:
    """Return a list of service names that are being hosted by the specified node."""
    services = get_service_server_info(node=_node, remote_node_name=node_name)
    return [service.name for service in services]


def get_node_service_types(node_name: str) -> list[str]:
    """Return a list of service types that are being hosted by the specified node."""
    services = get_service_server_info(node=_node, remote_node_name=node_name)
    return [service.types[0] for service in services]


def get_topic_type(
    topic: str, topics_pub_glob: list[str] | None, topics_sub_glob: list[str] | None
) -> str:
    """Return the type of the specified ROS topic."""
    # Note: this doesn't consider hidden topics.
    topics, types = get_topics_and_types(topics_pub_glob, topics_sub_glob)
    try:
        return types[topics.index(topic)]
    except ValueError:
        # Return empty string if the topic is not present.
        return ""


def filter_action_servers(topics: list[str]) -> list[str]:
    """Return a list of action servers."""
    # Note(@jubeira): filtering by topic should be enough; services can be taken into account as well.
    action_servers = []
    possible_action_server = ""
    possibility = [0, 0]

    action_topics = ["feedback", "status"]
    for topic in sorted(topics):
        split = topic.split("/")
        if len(split) >= 4:
            topic_name = split.pop()
            action_prefix = split.pop()
            if action_prefix != "_action":
                continue

            namespace = "/".join(split)
            if possible_action_server != namespace:
                possible_action_server = namespace
                possibility = [0, 0]
            if possible_action_server == namespace and topic_name in action_topics:
                possibility[action_topics.index(topic_name)] = 1
            if all(p == 1 for p in possibility):
                action_servers.append(possible_action_server)
                possibility = [0, 0]

    return action_servers


def get_service_type(service: str, services_glob: list[str] | None) -> str:
    """Return the type of the specified ROS service,."""
    # Note: this doesn't consider hidden services.
    services, types = get_services_and_types(services_glob)
    try:
        return types[services.index(service)]
    except ValueError:
        # Return empty string if the service is not present.
        return ""


def get_channel_info(
    channel: str,
    channels_glob: list[str] | None,
    getter_function: Callable,
    include_hidden: bool = False,
) -> list[str]:
    """
    Return a list of node names that are publishing on a specified channel.

    A channel may include topic publishers, subscribers, or service providers.

    :param channel: The name of the channel to query.
    :type channel: str
    :param channels_glob: A list of glob patterns to filter the channels.
    :type channels_glob: list[str] | None
    :param getter_function: A function to get the channel information for a given node.
    :type getter_function: Callable
    :param include_hidden: Whether to include hidden nodes in the search.
    :type include_hidden: bool
    :return: A list of node names that match the specified channel.
    :rtype: list[str]
    """
    if any_match(str(channel), channels_glob):
        channel_info_list = []
        node_list = get_nodes(include_hidden)
        for node in node_list:
            channel_info = getter_function(node)
            if channel in channel_info:
                channel_info_list.append(node)
        return channel_info_list
    return []


def get_publishers(
    topic: str, topics_pub_glob: list[str] | None, include_hidden: bool = False
) -> list[str]:
    """Return a list of node names that are publishing the specified topic."""
    return get_channel_info(
        topic, topics_pub_glob, get_node_publications, include_hidden=include_hidden
    )


def get_subscribers(
    topic: str, topics_sub_glob: list[str] | None, include_hidden: bool = False
) -> list[str]:
    """Return a list of node names that are subscribing to the specified topic."""
    return get_channel_info(
        topic, topics_sub_glob, get_node_subscriptions, include_hidden=include_hidden
    )


def get_service_providers(
    queried_type: str, services_glob: list[str] | None, include_hidden: bool = False
) -> list[str]:
    """Return a list of node names that are advertising a service with the specified type."""
    return get_channel_info(
        queried_type,
        services_glob,
        get_node_service_types,
        include_hidden=include_hidden,
    )


def get_service_node(
    queried_type: str, services_glob: list[str] | None, include_hidden: bool = False
) -> str:
    """Return the name of the node that is providing the given service, or empty string."""
    node_name = get_channel_info(
        queried_type, services_glob, get_node_services, include_hidden=include_hidden
    )
    if node_name:
        return node_name[0]
    return ""


def get_action_type(action_name: str) -> str:
    """
    Return the type of the specified ROS action.

    If the action does not exist, an empty string is returned.
    """
    names_and_types = get_action_names_and_types(node=_node)

    for name, types in names_and_types:
        if name == action_name and types:
            return types[0]

    return ""
'''


@pytest.fixture
def image(tmp_path):
    assert len(PINNED_PROXY) == 13935
    assert hashlib.sha256(PINNED_PROXY).hexdigest() == compat.ORIGINAL_SHA256
    target = tmp_path / compat.TARGET
    target.parent.mkdir(parents=True)
    target.write_bytes(PINNED_PROXY)
    target.chmod(0o640)
    status = tmp_path / compat.STATUS
    status.parent.mkdir(parents=True)
    status.write_text(
        "\n\n".join(
            f"Package: {package}\nStatus: install ok installed\nVersion: {version}"
            for package, version in compat.VERSIONS.items()
        )
        + "\n\n",
        encoding="utf-8",
    )
    return tmp_path


def _unexpected(*args, **kwargs):
    raise AssertionError("unrelated dependency API must not be invoked")


def _module(monkeypatch, name, **members):
    module = types.ModuleType(name)
    for key, value in members.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_proxy(monkeypatch, target, standalone, legacy):
    # Populate every import used by the actual proxy before executing it.
    # No import can reach an installed ROS package or a provider.
    modules = {
        "ros2action.api": {"get_action_names_and_types": legacy},
        "rclpy.action": {"get_action_names_and_types": standalone},
        "ros2interface.api": {"type_completer": _unexpected},
        "ros2node.api": dict.fromkeys(
            [
                "get_node_names",
                "get_publisher_info",
                "get_service_server_info",
                "get_subscriber_info",
            ],
            _unexpected,
        ),
        "ros2service.api": dict.fromkeys(
            ["get_service_names", "get_service_names_and_types"], _unexpected
        ),
        "ros2topic.api": dict.fromkeys(
            ["get_topic_names", "get_topic_names_and_types"], _unexpected
        ),
        "rosapi.glob_helper": {"any_match": _unexpected, "filter_globs": _unexpected},
    }
    for name, members in modules.items():
        parent = name.split(".")[0]
        if parent not in modules:
            package = _module(monkeypatch, parent)
            package.__path__ = []
        _module(monkeypatch, name, **members)
    spec = importlib.util.spec_from_file_location("rosapi.proxy", target)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "rosapi.proxy", module)
    # Compile exact file bytes, bypassing timestamp/size-based pyc cache reuse.
    exec(compile(target.read_bytes(), str(target), "exec"), module.__dict__)
    return module


def test_actual_proxy_mismatch_then_standalone_lookup(image, monkeypatch):
    target = image / compat.TARGET
    calls = []

    class Handle:
        active = False

        def __enter__(self):
            self.active = True
            return self

        def __exit__(self, *args):
            self.active = False

        def get_action_names_and_types(self):
            assert self.active
            calls.append("handle")
            return [
                ("/turtle1/rotate_absolute", ["turtlesim/action/RotateAbsolute"]),
                ("/empty", []),
                ("/other", ["example/action/First", "example/action/Second"]),
            ]

    node = types.SimpleNamespace(handle=Handle())
    assert not hasattr(node, "get_action_names_and_types")

    def legacy(*, node):
        calls.append("legacy")
        return node.get_action_names_and_types()

    def standalone(node):
        # Exact official 3.3.21 API shape/handle semantics, inert dependency.
        with node.handle:
            return node.handle.get_action_names_and_types()

    original = _load_proxy(monkeypatch, target, standalone, legacy)
    original.init(node)
    with pytest.raises(AttributeError, match="get_action_names_and_types"):
        original.get_action_type("/turtle1/rotate_absolute")
    assert calls == ["legacy"]
    assert compat.install(image) == "applied"
    patched = _load_proxy(monkeypatch, target, standalone, legacy)
    patched.init(node)
    assert patched.get_action_names_and_types is standalone
    action_name = "/turtle1/rotate_absolute"
    action_type = patched.get_action_type(action_name)
    assert action_type == "turtlesim/action/RotateAbsolute"
    # Inert metadata provenance, not a compiler/profile bypass or provider call.
    metadata = {"action_name": action_name, "action_type": action_type}
    assert metadata == {
        "action_name": "/turtle1/rotate_absolute",
        "action_type": "turtlesim/action/RotateAbsolute",
    }
    assert patched.get_action_type("/absent") == ""
    assert patched.get_action_type("/empty") == ""
    assert patched.get_action_type("/other") == "example/action/First"
    assert calls == ["legacy", "handle", "handle", "handle", "handle"]
    assert node.handle.active is False


def test_exact_delta_and_safe_idempotence(image):
    target = image / compat.TARGET
    before = target.stat()
    assert compat.install(image) == "applied"
    expected = PINNED_PROXY.replace(compat.OLD_IMPORT, compat.NEW_IMPORT, 1)
    assert target.read_bytes() == expected
    assert expected.replace(compat.NEW_IMPORT, compat.OLD_IMPORT, 1) == PINNED_PROXY
    after = target.stat()
    assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode)
    assert (after.st_uid, after.st_gid) == (before.st_uid, before.st_gid)
    assert compat.install(image) == "already-applied"
    assert target.stat().st_ino == after.st_ino
    assert target.stat().st_mtime_ns == after.st_mtime_ns
    assert target.read_bytes() == expected
    assert not list(target.parent.glob(".rosapi-compat-*"))


@pytest.mark.parametrize("kind", ["hash", "import", "layout", "already-altered"])
def test_reject_unrecognized_proxy_without_mutation(image, kind):
    target = image / compat.TARGET
    bad = {
        "hash": PINNED_PROXY + b"# unrelated change\n",
        "import": PINNED_PROXY.replace(compat.OLD_IMPORT, b"from other import lookup\n"),
        "layout": PINNED_PROXY.replace(compat.CALL, compat.CALL + compat.CALL),
        "already-altered": PINNED_PROXY.replace(compat.OLD_IMPORT, compat.NEW_IMPORT)
        + b"# unrelated change\n",
    }[kind]
    target.write_bytes(bad)
    before = target.stat()
    with pytest.raises(compat.CompatibilityError):
        compat.install(image)
    assert target.read_bytes() == bad
    assert target.stat().st_ino == before.st_ino
    assert target.stat().st_mtime_ns == before.st_mtime_ns
    assert not list(target.parent.glob(".rosapi-compat-*"))


@pytest.mark.parametrize("kind", ["rosapi", "rclpy", "missing", "status", "duplicate"])
def test_reject_unsupported_version_before_write(image, kind):
    status = image / compat.STATUS
    data = status.read_text()
    if kind in {"rosapi", "rclpy"}:
        version = compat.VERSIONS[f"ros-humble-{kind}"]
        data = data.replace(version, "unsupported")
    elif kind == "missing":
        data = ""
    elif kind == "status":
        data = data.replace("install ok installed", "deinstall ok config-files")
    else:
        data += data
    status.write_text(data)
    target = image / compat.TARGET
    before = target.stat()
    with pytest.raises(compat.CompatibilityError, match="package|version"):
        compat.install(image)
    assert target.read_bytes() == PINNED_PROXY
    assert target.stat().st_ino == before.st_ino
    assert target.stat().st_mtime_ns == before.st_mtime_ns
    assert status.read_text() == data


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "parent", "status-link", "directory"])
def test_reject_links_and_nonregular_targets(image, kind):
    target = image / compat.TARGET
    outside = image / "outside.py"
    outside.write_bytes(PINNED_PROXY)
    if kind == "symlink":
        target.unlink()
        target.symlink_to(outside)
    elif kind == "hardlink":
        target.unlink()
        target.hardlink_to(outside)
    elif kind == "parent":
        parent = target.parent
        moved = parent.with_name("rosapi-moved")
        parent.rename(moved)
        parent.symlink_to(moved, target_is_directory=True)
    elif kind == "status-link":
        status = image / compat.STATUS
        saved = image / "saved-status"
        status.rename(saved)
        status.symlink_to(saved)
    else:
        target.unlink()
        target.mkdir()
    with pytest.raises(compat.CompatibilityError):
        compat.install(image)
    assert outside.read_bytes() == PINNED_PROXY
    if kind != "directory":
        assert target.read_bytes() == PINNED_PROXY
    else:
        assert list(target.iterdir()) == []
    assert not list(target.parent.glob(".rosapi-compat-*"))


@pytest.mark.parametrize("relative", [compat.TARGET, compat.STATUS])
def test_missing_image_file_is_refused(image, relative):
    (image / relative).unlink()
    with pytest.raises(compat.CompatibilityError, match="file operation failed"):
        compat.install(image)
    if relative != compat.TARGET:
        assert (image / compat.TARGET).read_bytes() == PINNED_PROXY


def test_atomic_replace_failure_keeps_original_and_cleans_up(image, monkeypatch):
    def refuse_replace(*args):
        raise OSError("fixture replacement refused")

    monkeypatch.setattr(compat.os, "replace", refuse_replace)
    with pytest.raises(compat.CompatibilityError, match="replacement refused"):
        compat.install(image)
    target = image / compat.TARGET
    assert target.read_bytes() == PINNED_PROXY
    assert not list(target.parent.glob(".rosapi-compat-*"))


def test_cli_fixed_target_success_refusal_and_argument_rejection(image, monkeypatch, capsys):
    actual_install = compat.install
    roots = []

    def owned_install(root):
        roots.append(root)
        return actual_install(image)

    monkeypatch.setattr(compat, "install", owned_install)
    assert compat.main([]) == 0
    assert "applied" in capsys.readouterr().out
    assert compat.main([]) == 0
    assert "already-applied" in capsys.readouterr().out
    assert roots == [Path("/"), Path("/")]
    assert compat.main(["--target", "/elsewhere"]) == 2
    assert "no arguments" in capsys.readouterr().err
    assert len(roots) == 2
    (image / compat.STATUS).write_text("")
    before = (image / compat.TARGET).read_bytes()
    assert compat.main([]) == 1
    assert "refused" in capsys.readouterr().err
    assert (image / compat.TARGET).read_bytes() == before
